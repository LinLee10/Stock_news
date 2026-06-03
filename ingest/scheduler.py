"""
Scheduler and orchestration for the ingestion system.

This module handles job queuing, worker coordination, and task scheduling
while respecting all compliance requirements.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

import structlog
import yaml

from .models import (
    IngestionJob, IngestionResult, DomainPolicy, FetchStrategy,
    ExtractedArticle, QualityDecision
)
from .browsing_session import CompliantBrowsingSession
from .extractor import FinancialContentExtractor
from .quality import ContentQualityFilter
from .storage import RedisStorage, PostgreSQLStorage

logger = structlog.get_logger(__name__)


class ConfigManager:
    """Manages configuration loading and domain policies"""
    
    def __init__(self, config_path: str = "ingest/config.yaml"):
        self.config_path = config_path
        self.config = {}
        self.domain_policies: Dict[str, DomainPolicy] = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            # Parse domain policies
            self._parse_domain_policies()
            
            logger.info("Configuration loaded", config_path=self.config_path)
            return self.config
            
        except Exception as e:
            logger.error("Failed to load config", path=self.config_path, error=str(e))
            raise
    
    def _parse_domain_policies(self):
        """Parse domain policies from config"""
        domains_config = self.config.get('domains', {})
        
        for domain, policy_data in domains_config.items():
            self.domain_policies[domain] = DomainPolicy(
                domain=domain,
                allowed=policy_data.get('allowed', True),
                crawl_delay_seconds=policy_data.get('crawl_delay_seconds', 12.0),
                max_concurrency=policy_data.get('max_concurrency', 1),
                credibility_score=policy_data.get('credibility_score', 0.5),
                preferred_strategy=FetchStrategy(policy_data.get('preferred_strategy', 'playwright')),
                custom_selectors=policy_data.get('custom_selectors', {})
            )
    
    def get_domain_policy(self, domain: str) -> DomainPolicy:
        """Get domain policy, falling back to defaults"""
        if domain in self.domain_policies:
            return self.domain_policies[domain]
        
        # Default policy
        return DomainPolicy(
            domain=domain,
            allowed=True,
            crawl_delay_seconds=self.config.get('rate_limiting', {}).get('default_delay_seconds', 12.0),
            max_concurrency=self.config.get('rate_limiting', {}).get('max_concurrent_per_domain', 1),
            credibility_score=0.5,
            preferred_strategy=FetchStrategy.PLAYWRIGHT
        )


class JobQueue:
    """Manages ingestion job queues"""
    
    def __init__(self, storage: RedisStorage, config: Dict[str, Any]):
        self.storage = storage
        self.config = config
        self.queue_names = config.get('queues', {
            'pending': 'jobs:pending',
            'inflight': 'jobs:inflight', 
            'deadletter': 'jobs:deadletter'
        })
    
    async def enqueue_job(self, url: str, priority: int = 1, metadata: Dict[str, Any] = None) -> str:
        """Add a job to the pending queue"""
        domain = urlparse(url).netloc
        
        job = IngestionJob(
            job_id=str(uuid.uuid4()),
            url=url,
            domain=domain,
            priority=priority,
            metadata=metadata or {}
        )
        
        job_data = job.model_dump_json()
        success = await self.storage.push_job(self.queue_names['pending'], job_data)
        
        if success:
            logger.info("Job enqueued", job_id=job.job_id, url=url, domain=domain)
            return job.job_id
        else:
            raise RuntimeError(f"Failed to enqueue job for {url}")
    
    async def dequeue_job(self, timeout: int = 1) -> Optional[IngestionJob]:
        """Get next job from the pending queue"""
        job_data = await self.storage.pop_job(self.queue_names['pending'], timeout)
        
        if job_data:
            try:
                job = IngestionJob.model_validate_json(job_data)
                
                # Move to inflight queue
                await self.storage.push_job(self.queue_names['inflight'], job_data)
                
                return job
            except Exception as e:
                logger.error("Failed to parse job data", error=str(e))
                return None
        
        return None
    
    async def complete_job(self, job_id: str, result: IngestionResult):
        """Mark job as completed and remove from inflight"""
        # Note: In a full implementation, we'd need to track job_id -> job_data mapping
        # For simplicity, we're just logging completion here
        logger.info("Job completed", job_id=job_id, success=result.success)
    
    async def fail_job(self, job: IngestionJob, error_message: str):
        """Handle failed job"""
        job.attempts += 1
        
        if job.attempts >= job.max_attempts:
            # Send to dead letter queue
            await self.storage.push_job(self.queue_names['deadletter'], job.model_dump_json())
            logger.warning("Job moved to dead letter queue", 
                         job_id=job.job_id, 
                         attempts=job.attempts,
                         error=error_message)
        else:
            # Retry - put back in pending queue
            await self.storage.push_job(self.queue_names['pending'], job.model_dump_json())
            logger.info("Job requeued for retry", 
                       job_id=job.job_id, 
                       attempt=job.attempts,
                       error=error_message)
    
    async def get_queue_stats(self) -> Dict[str, int]:
        """Get queue length statistics"""
        stats = {}
        for queue_type, queue_name in self.queue_names.items():
            stats[queue_type] = await self.storage.queue_length(queue_name)
        return stats


class IngestionOrchestrator:
    """
    Main orchestrator that coordinates the ingestion pipeline.
    
    Handles job processing, worker management, and ensures compliance
    with all rate limiting and policy requirements.
    """
    
    def __init__(self, config_path: str = "ingest/config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        
        # Initialize storage
        redis_config = self.config.get('redis', {})
        self.redis_storage = RedisStorage(redis_config.get('url', 'redis://localhost:6379/0'))
        
        # Optional PostgreSQL
        self.pg_storage = None
        if self.config.get('postgresql', {}).get('enabled', False):
            self.pg_storage = PostgreSQLStorage(
                self.config['postgresql']['connection_string']
            )
        
        # Initialize components
        self.job_queue = JobQueue(self.redis_storage, self.config)
        self.extractor = FinancialContentExtractor()
        self.quality_filter = ContentQualityFilter(
            self.redis_storage, 
            self.config.get('quality', {})
        )
        
        # Worker control
        self._workers_running = False
        self._worker_tasks: List[asyncio.Task] = []
        
        # Metrics
        self.metrics = {
            'jobs_processed': 0,
            'jobs_successful': 0,
            'jobs_failed': 0,
            'articles_extracted': 0,
            'articles_kept': 0,
            'articles_rejected': 0
        }
    
    async def initialize(self):
        """Initialize all components"""
        await self.redis_storage.connect()
        
        if self.pg_storage:
            await self.pg_storage.connect()
            await self.pg_storage.create_tables()
        
        await self.quality_filter.duplication_detector.initialize()
        
        logger.info("Ingestion orchestrator initialized")
    
    async def shutdown(self):
        """Shutdown all components"""
        self._workers_running = False
        
        # Wait for workers to finish
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        await self.redis_storage.disconnect()
        
        if self.pg_storage:
            await self.pg_storage.disconnect()
        
        logger.info("Ingestion orchestrator shutdown complete")
    
    async def enqueue_urls(self, urls: List[str], priority: int = 1) -> List[str]:
        """Enqueue multiple URLs for processing"""
        job_ids = []
        
        for url in urls:
            try:
                job_id = await self.job_queue.enqueue_job(url, priority)
                job_ids.append(job_id)
            except Exception as e:
                logger.error("Failed to enqueue URL", url=url, error=str(e))
        
        logger.info("URLs enqueued", count=len(job_ids), total_urls=len(urls))
        return job_ids
    
    async def start_workers(self, num_workers: int = 3):
        """Start worker processes"""
        if self._workers_running:
            logger.warning("Workers already running")
            return
        
        self._workers_running = True
        self._worker_tasks = []
        
        for worker_id in range(num_workers):
            task = asyncio.create_task(self._worker_loop(worker_id))
            self._worker_tasks.append(task)
        
        logger.info("Workers started", count=num_workers)
    
    async def stop_workers(self):
        """Stop all workers"""
        self._workers_running = False
        
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            self._worker_tasks = []
        
        logger.info("All workers stopped")
    
    async def _worker_loop(self, worker_id: int):
        """Main worker loop"""
        logger.info("Worker started", worker_id=worker_id)
        
        browser_config = self.config.get('browser', {})
        
        async with CompliantBrowsingSession(
            storage=self.redis_storage,
            user_agent=browser_config.get('user_agent'),
            contact_email=browser_config.get('contact_email')
        ) as session:
            
            while self._workers_running:
                try:
                    # Get next job
                    job = await self.job_queue.dequeue_job(timeout=5)
                    
                    if not job:
                        await asyncio.sleep(1)
                        continue
                    
                    # Process job
                    result = await self._process_job(session, job)
                    
                    # Update metrics
                    self.metrics['jobs_processed'] += 1
                    if result.success:
                        self.metrics['jobs_successful'] += 1
                        if result.article:
                            self.metrics['articles_extracted'] += 1
                            if result.quality_decision and result.quality_decision.keep:
                                self.metrics['articles_kept'] += 1
                            else:
                                self.metrics['articles_rejected'] += 1
                    else:
                        self.metrics['jobs_failed'] += 1
                    
                    # Complete or fail job
                    if result.success:
                        await self.job_queue.complete_job(job.job_id, result)
                    else:
                        await self.job_queue.fail_job(job, result.error_message or "Unknown error")
                    
                except Exception as e:
                    logger.error("Worker error", worker_id=worker_id, error=str(e))
                    await asyncio.sleep(5)  # Back off on error
        
        logger.info("Worker stopped", worker_id=worker_id)
    
    async def _process_job(self, session: CompliantBrowsingSession, job: IngestionJob) -> IngestionResult:
        """Process a single ingestion job"""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Check domain policy
            domain_policy = self.config_manager.get_domain_policy(job.domain)
            
            if not domain_policy.allowed:
                return IngestionResult(
                    job_id=job.job_id,
                    url=job.url,
                    success=False,
                    error_message="Domain not allowed by policy",
                    processing_time_ms=0
                )
            
            # Fetch content
            logger.info("Processing job", job_id=job.job_id, url=job.url)
            
            render_result = await session.fetch_html(job.url)
            
            if not render_result.success:
                return IngestionResult(
                    job_id=job.job_id,
                    url=job.url,
                    success=False,
                    error_message=render_result.error_message,
                    processing_time_ms=render_result.render_time_ms,
                    strategy_used=render_result.strategy
                )
            
            # Extract content
            article = self.extractor.extract_article(render_result)
            
            # Quality assessment
            quality_decision = await self.quality_filter.evaluate_quality(article)
            
            # Store if keeping
            if quality_decision.keep and self.pg_storage:
                article_data = article.model_dump()
                await self.pg_storage.save_article(article_data)
                
                quality_data = quality_decision.model_dump()
                quality_data['url'] = article.url
                await self.pg_storage.save_quality_decision(quality_data)
            
            processing_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            
            return IngestionResult(
                job_id=job.job_id,
                url=job.url,
                success=True,
                article=article,
                quality_decision=quality_decision,
                processing_time_ms=processing_time,
                strategy_used=render_result.strategy,
                completed_at=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            processing_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            
            logger.error("Job processing failed", 
                        job_id=job.job_id, 
                        url=job.url, 
                        error=str(e))
            
            return IngestionResult(
                job_id=job.job_id,
                url=job.url,
                success=False,
                error_message=str(e),
                processing_time_ms=processing_time
            )
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        queue_stats = await self.job_queue.get_queue_stats()
        
        return {
            **self.metrics,
            'queues': queue_stats,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }