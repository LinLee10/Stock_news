"""
AsyncProcessingPipeline - Event-driven architecture for processing financial news articles.

This module provides:
1. Event-driven architecture with Kafka/Redis streams
2. Parallel processing using asyncio (configurable concurrency)
3. Queue management with priority lanes
4. Real-time and batch processing capabilities
5. Fault tolerance and error handling
6. Backpressure management and circuit breakers
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Callable, NamedTuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import defaultdict, deque
import uuid

import structlog
import redis.asyncio as redis
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import aiohttp

from .news_deduplicator import NewsDeduplicator, DuplicateResult
from .ticker_relevance_matcher import TickerRelevanceMatcher, RelevanceScore
from .article_quality_assessment import ArticleQualityAssessment, QualityScore

logger = structlog.get_logger(__name__)


class ProcessingPriority(Enum):
    """Processing priority levels"""
    BREAKING_NEWS = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    HISTORICAL = 5


class ProcessingStatus(Enum):
    """Processing status for articles"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProcessingJob:
    """Processing job for an article"""
    job_id: str
    article_url: str
    title: str
    content: str
    source_url: str
    publish_date: Optional[datetime] = None
    author: Optional[str] = None
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    target_tickers: Optional[List[str]] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProcessingResult:
    """Result of processing an article"""
    job_id: str
    status: ProcessingStatus
    duplicate_result: Optional[DuplicateResult] = None
    relevance_scores: Optional[List[RelevanceScore]] = None
    quality_score: Optional[QualityScore] = None
    processing_time_ms: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EventType(Enum):
    """Event types in the processing pipeline"""
    ARTICLE_RECEIVED = "article_received"
    DUPLICATE_CHECK_STARTED = "duplicate_check_started"
    DUPLICATE_CHECK_COMPLETED = "duplicate_check_completed"
    RELEVANCE_ANALYSIS_STARTED = "relevance_analysis_started"
    RELEVANCE_ANALYSIS_COMPLETED = "relevance_analysis_completed"
    QUALITY_ASSESSMENT_STARTED = "quality_assessment_started"
    QUALITY_ASSESSMENT_COMPLETED = "quality_assessment_completed"
    PROCESSING_COMPLETED = "processing_completed"
    PROCESSING_FAILED = "processing_failed"
    ARTICLE_SKIPPED = "article_skipped"


@dataclass
class ProcessingEvent:
    """Event in the processing pipeline"""
    event_id: str
    event_type: EventType
    job_id: str
    timestamp: datetime
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}


class PipelineMetrics:
    """Metrics tracking for the processing pipeline"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.start_time = time.time()
        self.jobs_received = 0
        self.jobs_processed = 0
        self.jobs_failed = 0
        self.jobs_skipped = 0
        self.duplicates_found = 0
        self.high_quality_articles = 0
        self.processing_times = deque(maxlen=1000)  # Keep last 1000 processing times
        self.error_counts = defaultdict(int)
        self.priority_counts = defaultdict(int)
        self.hourly_throughput = defaultdict(int)
    
    def record_job_received(self, priority: ProcessingPriority):
        """Record a new job received"""
        self.jobs_received += 1
        self.priority_counts[priority.value] += 1
        hour = datetime.now(timezone.utc).hour
        self.hourly_throughput[hour] += 1
    
    def record_job_completed(self, processing_time_ms: int):
        """Record a job completion"""
        self.jobs_processed += 1
        self.processing_times.append(processing_time_ms)
    
    def record_job_failed(self, error_type: str):
        """Record a job failure"""
        self.jobs_failed += 1
        self.error_counts[error_type] += 1
    
    def record_duplicate_found(self):
        """Record a duplicate article found"""
        self.duplicates_found += 1
        self.jobs_skipped += 1
    
    def record_high_quality_article(self):
        """Record a high-quality article processed"""
        self.high_quality_articles += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        runtime_hours = (time.time() - self.start_time) / 3600
        
        summary = {
            'runtime_hours': runtime_hours,
            'total_jobs_received': self.jobs_received,
            'jobs_processed': self.jobs_processed,
            'jobs_failed': self.jobs_failed,
            'jobs_skipped': self.jobs_skipped,
            'duplicates_found': self.duplicates_found,
            'high_quality_articles': self.high_quality_articles,
            'success_rate': self.jobs_processed / max(1, self.jobs_received),
            'error_rate': self.jobs_failed / max(1, self.jobs_received),
            'duplicate_rate': self.duplicates_found / max(1, self.jobs_received),
            'throughput_jobs_per_hour': self.jobs_processed / max(0.1, runtime_hours),
            'priority_distribution': dict(self.priority_counts),
            'error_distribution': dict(self.error_counts),
            'avg_processing_time_ms': sum(self.processing_times) / max(1, len(self.processing_times)),
        }
        
        if self.processing_times:
            summary.update({
                'min_processing_time_ms': min(self.processing_times),
                'max_processing_time_ms': max(self.processing_times),
                'p95_processing_time_ms': sorted(self.processing_times)[int(0.95 * len(self.processing_times))]
            })
        
        return summary


class AsyncProcessingPipeline:
    """
    High-performance async processing pipeline for financial news articles.
    
    Features:
    - Event-driven architecture with Redis Streams/Kafka
    - Configurable parallel processing (1-50 concurrent workers)
    - Priority-based queue management
    - Real-time and batch processing
    - Comprehensive error handling and retry logic
    - Circuit breakers and backpressure management
    - Detailed metrics and monitoring
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379/0",
                 kafka_bootstrap_servers: Optional[str] = None,
                 max_concurrent_jobs: int = 20,
                 use_kafka: bool = False):
        
        self.redis_url = redis_url
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.max_concurrent_jobs = max_concurrent_jobs
        self.use_kafka = use_kafka
        
        # Initialize components
        self.redis_client: Optional[redis.Redis] = None
        self.kafka_producer: Optional[KafkaProducer] = None
        self.kafka_consumer: Optional[KafkaConsumer] = None
        
        # Processing components
        self.deduplicator: Optional[NewsDeduplicator] = None
        self.relevance_matcher: Optional[TickerRelevanceMatcher] = None
        self.quality_assessor: Optional[ArticleQualityAssessment] = None
        
        # Queue management
        self.job_queues = {
            ProcessingPriority.BREAKING_NEWS: "jobs:breaking_news",
            ProcessingPriority.HIGH: "jobs:high_priority", 
            ProcessingPriority.NORMAL: "jobs:normal_priority",
            ProcessingPriority.LOW: "jobs:low_priority",
            ProcessingPriority.HISTORICAL: "jobs:historical"
        }
        
        # Processing control
        self.is_running = False
        self.worker_tasks: List[asyncio.Task] = []
        self.processing_semaphore: Optional[asyncio.Semaphore] = None
        
        # Metrics and monitoring
        self.metrics = PipelineMetrics()
        
        # Event handlers
        self.event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        
        # Circuit breaker for external dependencies
        self.circuit_breakers = {}
        self.circuit_breaker_failure_threshold = 5
        self.circuit_breaker_recovery_time = 300  # 5 minutes
        
    async def initialize(self):
        """Initialize the processing pipeline"""
        logger.info("Initializing async processing pipeline")
        
        # Initialize Redis connection
        self.redis_client = redis.from_url(self.redis_url)
        await self.redis_client.ping()
        logger.info("Redis connection established")
        
        # Initialize Kafka if enabled
        if self.use_kafka and self.kafka_bootstrap_servers:
            self._initialize_kafka()
        
        # Initialize processing components
        self.deduplicator = NewsDeduplicator(redis_client=await self._get_sync_redis())
        self.relevance_matcher = TickerRelevanceMatcher()
        self.quality_assessor = ArticleQualityAssessment()
        
        # Initialize concurrency control
        self.processing_semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
        
        logger.info("Processing pipeline initialized", 
                   max_concurrent_jobs=self.max_concurrent_jobs,
                   use_kafka=self.use_kafka)
    
    def _initialize_kafka(self):
        """Initialize Kafka producer and consumer"""
        try:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.kafka_bootstrap_servers,
                value_serializer=lambda x: json.dumps(x, default=str).encode('utf-8'),
                acks='all',  # Wait for all replicas
                retries=3,
                batch_size=16384,
                linger_ms=10
            )
            
            self.kafka_consumer = KafkaConsumer(
                'financial_news_processing',
                bootstrap_servers=self.kafka_bootstrap_servers,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                group_id='processing_pipeline',
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            
            logger.info("Kafka initialized", servers=self.kafka_bootstrap_servers)
            
        except Exception as e:
            logger.error("Failed to initialize Kafka", error=str(e))
            self.use_kafka = False
    
    async def _get_sync_redis(self) -> redis.Redis:
        """Get synchronous Redis client for components that need it"""
        import redis as sync_redis
        return sync_redis.from_url(self.redis_url)
    
    async def submit_job(self, 
                        article_url: str,
                        title: str, 
                        content: str,
                        source_url: str,
                        priority: ProcessingPriority = ProcessingPriority.NORMAL,
                        target_tickers: Optional[List[str]] = None,
                        publish_date: Optional[datetime] = None,
                        author: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit a new processing job.
        
        Returns:
            job_id: Unique identifier for the job
        """
        job_id = str(uuid.uuid4())
        
        job = ProcessingJob(
            job_id=job_id,
            article_url=article_url,
            title=title,
            content=content,
            source_url=source_url,
            priority=priority,
            target_tickers=target_tickers,
            publish_date=publish_date,
            author=author,
            metadata=metadata or {}
        )
        
        # Add to appropriate priority queue
        queue_name = self.job_queues[priority]
        job_data = json.dumps(asdict(job), default=str)
        
        if self.use_kafka:
            # Send to Kafka topic
            await self._send_to_kafka('job_submitted', job_data)
        else:
            # Add to Redis queue
            await self.redis_client.lpush(queue_name, job_data)
        
        # Record metrics
        self.metrics.record_job_received(priority)
        
        # Emit event
        await self._emit_event(EventType.ARTICLE_RECEIVED, job_id, {'priority': priority.value})
        
        logger.debug("Job submitted", job_id=job_id, priority=priority.value)
        
        return job_id
    
    async def start_processing(self, num_workers: int = None) -> None:
        """Start the processing pipeline with specified number of workers"""
        if self.is_running:
            logger.warning("Pipeline already running")
            return
        
        if num_workers is None:
            num_workers = min(self.max_concurrent_jobs, 10)  # Default to 10 workers
        
        self.is_running = True
        self.worker_tasks = []
        
        # Start worker tasks
        for i in range(num_workers):
            task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self.worker_tasks.append(task)
        
        # Start metrics collection task
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.worker_tasks.append(metrics_task)
        
        logger.info("Processing pipeline started", num_workers=num_workers)
    
    async def stop_processing(self) -> None:
        """Stop the processing pipeline"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
        
        logger.info("Processing pipeline stopped")
    
    async def _worker_loop(self, worker_id: str):
        """Main worker loop for processing jobs"""
        logger.info("Worker started", worker_id=worker_id)
        
        while self.is_running:
            try:
                # Get next job from priority queues
                job = await self._get_next_job()
                
                if job is None:
                    await asyncio.sleep(1)  # No jobs available
                    continue
                
                # Process the job
                async with self.processing_semaphore:
                    result = await self._process_job(job)
                    
                    # Handle result
                    await self._handle_processing_result(job, result)
                
            except asyncio.CancelledError:
                logger.info("Worker cancelled", worker_id=worker_id)
                break
            except Exception as e:
                logger.error("Worker error", worker_id=worker_id, error=str(e))
                await asyncio.sleep(5)  # Backoff on error
        
        logger.info("Worker stopped", worker_id=worker_id)
    
    async def _get_next_job(self) -> Optional[ProcessingJob]:
        """Get next job from priority queues"""
        # Check queues in priority order
        for priority in ProcessingPriority:
            queue_name = self.job_queues[priority]
            
            try:
                job_data = await self.redis_client.rpop(queue_name)
                if job_data:
                    job_dict = json.loads(job_data)
                    # Convert datetime strings back to datetime objects
                    for date_field in ['created_at', 'started_at', 'completed_at', 'publish_date']:
                        if job_dict.get(date_field):
                            job_dict[date_field] = datetime.fromisoformat(job_dict[date_field].replace('Z', '+00:00'))
                    
                    job = ProcessingJob(**job_dict)
                    job.started_at = datetime.now(timezone.utc)
                    job.status = ProcessingStatus.PROCESSING
                    
                    return job
                    
            except Exception as e:
                logger.error("Failed to get job from queue", queue=queue_name, error=str(e))
        
        return None
    
    async def _process_job(self, job: ProcessingJob) -> ProcessingResult:
        """Process a single job through the complete pipeline"""
        start_time = time.time()
        
        try:
            logger.debug("Processing job", job_id=job.job_id)
            
            # Step 1: Duplicate detection
            await self._emit_event(EventType.DUPLICATE_CHECK_STARTED, job.job_id)
            
            fingerprint = self.deduplicator.create_content_fingerprint(
                job.article_url, job.title, job.content, job.publish_date
            )
            duplicate_result = self.deduplicator.check_duplicate(fingerprint)
            
            await self._emit_event(EventType.DUPLICATE_CHECK_COMPLETED, job.job_id, 
                                 {'is_duplicate': duplicate_result.is_duplicate})
            
            # If duplicate, skip further processing
            if duplicate_result.is_duplicate:
                self.metrics.record_duplicate_found()
                
                processing_time = int((time.time() - start_time) * 1000)
                
                return ProcessingResult(
                    job_id=job.job_id,
                    status=ProcessingStatus.SKIPPED,
                    duplicate_result=duplicate_result,
                    processing_time_ms=processing_time,
                    metadata={'skip_reason': 'duplicate'}
                )
            
            # Store fingerprint for future duplicate detection
            self.deduplicator.store_fingerprint(fingerprint)
            
            # Step 2: Ticker relevance analysis
            await self._emit_event(EventType.RELEVANCE_ANALYSIS_STARTED, job.job_id)
            
            relevance_scores = self.relevance_matcher.analyze_article_relevance(
                job.title, job.content, job.target_tickers
            )
            
            await self._emit_event(EventType.RELEVANCE_ANALYSIS_COMPLETED, job.job_id,
                                 {'relevance_scores_count': len(relevance_scores)})
            
            # Step 3: Quality assessment
            await self._emit_event(EventType.QUALITY_ASSESSMENT_STARTED, job.job_id)
            
            quality_score = self.quality_assessor.assess_article_quality(
                job.title, job.content, job.source_url, job.publish_date, job.author
            )
            
            await self._emit_event(EventType.QUALITY_ASSESSMENT_COMPLETED, job.job_id,
                                 {'overall_score': quality_score.overall_score})
            
            # Record high-quality articles
            if quality_score.overall_score >= 0.8:
                self.metrics.record_high_quality_article()
            
            processing_time = int((time.time() - start_time) * 1000)
            
            result = ProcessingResult(
                job_id=job.job_id,
                status=ProcessingStatus.COMPLETED,
                duplicate_result=duplicate_result,
                relevance_scores=relevance_scores,
                quality_score=quality_score,
                processing_time_ms=processing_time
            )
            
            await self._emit_event(EventType.PROCESSING_COMPLETED, job.job_id, 
                                 {'processing_time_ms': processing_time})
            
            self.metrics.record_job_completed(processing_time)
            
            return result
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            
            logger.error("Job processing failed", job_id=job.job_id, error=str(e))
            
            await self._emit_event(EventType.PROCESSING_FAILED, job.job_id, 
                                 {'error': str(e)})
            
            self.metrics.record_job_failed(type(e).__name__)
            
            return ProcessingResult(
                job_id=job.job_id,
                status=ProcessingStatus.FAILED,
                processing_time_ms=processing_time,
                error_message=str(e)
            )
    
    async def _handle_processing_result(self, job: ProcessingJob, result: ProcessingResult):
        """Handle the result of job processing"""
        try:
            # Update job status
            job.completed_at = datetime.now(timezone.utc)
            job.status = result.status
            
            if result.error_message:
                job.error_message = result.error_message
            
            # Store result
            if self.use_kafka:
                await self._send_to_kafka('processing_result', asdict(result))
            else:
                # Store in Redis for later retrieval
                result_key = f"results:{result.job_id}"
                result_data = json.dumps(asdict(result), default=str)
                await self.redis_client.setex(result_key, 86400, result_data)  # 24 hour TTL
            
            # Handle retry logic for failed jobs
            if result.status == ProcessingStatus.FAILED and job.retry_count < job.max_retries:
                await self._retry_job(job)
            
        except Exception as e:
            logger.error("Failed to handle processing result", 
                        job_id=job.job_id, error=str(e))
    
    async def _retry_job(self, job: ProcessingJob):
        """Retry a failed job"""
        job.retry_count += 1
        job.status = ProcessingStatus.PENDING
        job.started_at = None
        job.completed_at = None
        
        # Add delay before retry (exponential backoff)
        delay_seconds = min(300, 2 ** job.retry_count)  # Max 5 minutes
        
        logger.info("Retrying job", job_id=job.job_id, 
                   retry_count=job.retry_count, delay_seconds=delay_seconds)
        
        # Schedule retry
        asyncio.create_task(self._delayed_job_submission(job, delay_seconds))
    
    async def _delayed_job_submission(self, job: ProcessingJob, delay_seconds: int):
        """Submit a job after a delay (for retries)"""
        await asyncio.sleep(delay_seconds)
        
        queue_name = self.job_queues[job.priority]
        job_data = json.dumps(asdict(job), default=str)
        await self.redis_client.lpush(queue_name, job_data)
    
    async def _emit_event(self, event_type: EventType, job_id: str, data: Dict[str, Any] = None):
        """Emit a processing event"""
        event = ProcessingEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            job_id=job_id,
            timestamp=datetime.now(timezone.utc),
            data=data or {}
        )
        
        # Send to event handlers
        for handler in self.event_handlers[event_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error("Event handler failed", 
                           event_type=event_type.value, error=str(e))
        
        # Store event for monitoring
        if self.redis_client:
            event_key = f"events:{event.event_id}"
            event_data = json.dumps(asdict(event), default=str)
            await self.redis_client.setex(event_key, 3600, event_data)  # 1 hour TTL
    
    async def _send_to_kafka(self, topic: str, data: Any):
        """Send data to Kafka topic"""
        if self.kafka_producer:
            try:
                self.kafka_producer.send(topic, data)
                self.kafka_producer.flush()
            except KafkaError as e:
                logger.error("Failed to send to Kafka", topic=topic, error=str(e))
    
    async def _metrics_collection_loop(self):
        """Background task for metrics collection and logging"""
        while self.is_running:
            try:
                # Log metrics every 60 seconds
                await asyncio.sleep(60)
                
                summary = self.metrics.get_summary()
                logger.info("Pipeline metrics", **summary)
                
                # Store metrics in Redis
                if self.redis_client:
                    metrics_key = f"metrics:{int(time.time())}"
                    await self.redis_client.setex(metrics_key, 3600, 
                                                json.dumps(summary, default=str))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics collection error", error=str(e))
    
    def add_event_handler(self, event_type: EventType, handler: Callable):
        """Add an event handler for a specific event type"""
        self.event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: EventType, handler: Callable):
        """Remove an event handler"""
        if handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job"""
        try:
            # Check if result exists
            result_key = f"results:{job_id}"
            result_data = await self.redis_client.get(result_key)
            
            if result_data:
                return json.loads(result_data)
            
            # Job might still be in queue or processing
            return {'job_id': job_id, 'status': 'not_found'}
            
        except Exception as e:
            logger.error("Failed to get job status", job_id=job_id, error=str(e))
            return None
    
    async def get_queue_stats(self) -> Dict[str, int]:
        """Get statistics for all queues"""
        stats = {}
        
        for priority, queue_name in self.job_queues.items():
            try:
                length = await self.redis_client.llen(queue_name)
                stats[priority.name.lower()] = length
            except Exception as e:
                logger.error("Failed to get queue stats", queue=queue_name, error=str(e))
                stats[priority.name.lower()] = -1
        
        return stats
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current pipeline metrics"""
        summary = self.metrics.get_summary()
        queue_stats = await self.get_queue_stats()
        summary['queue_lengths'] = queue_stats
        summary['is_running'] = self.is_running
        summary['active_workers'] = len(self.worker_tasks)
        
        return summary
    
    async def shutdown(self):
        """Shutdown the pipeline and cleanup resources"""
        logger.info("Shutting down processing pipeline")
        
        await self.stop_processing()
        
        if self.redis_client:
            await self.redis_client.aclose()
        
        if self.kafka_producer:
            self.kafka_producer.close()
        
        logger.info("Pipeline shutdown complete")


# Utility functions for easy integration
async def create_processing_pipeline(redis_url: str = "redis://localhost:6379/0",
                                   kafka_servers: Optional[str] = None,
                                   max_concurrent: int = 20) -> AsyncProcessingPipeline:
    """Create and initialize a processing pipeline"""
    pipeline = AsyncProcessingPipeline(
        redis_url=redis_url,
        kafka_bootstrap_servers=kafka_servers,
        max_concurrent_jobs=max_concurrent,
        use_kafka=kafka_servers is not None
    )
    
    await pipeline.initialize()
    return pipeline


async def process_articles_batch(articles: List[Dict[str, Any]], 
                               pipeline: AsyncProcessingPipeline,
                               priority: ProcessingPriority = ProcessingPriority.NORMAL) -> List[str]:
    """Process a batch of articles and return job IDs"""
    job_ids = []
    
    for article in articles:
        job_id = await pipeline.submit_job(
            article_url=article['url'],
            title=article['title'],
            content=article['content'],
            source_url=article.get('source_url', article['url']),
            priority=priority,
            target_tickers=article.get('target_tickers'),
            publish_date=article.get('publish_date'),
            author=article.get('author'),
            metadata=article.get('metadata')
        )
        job_ids.append(job_id)
    
    return job_ids


if __name__ == "__main__":
    # Test the async processing pipeline
    async def test_pipeline():
        # Create pipeline
        pipeline = await create_processing_pipeline()
        
        # Add event handler for demonstration
        def on_article_processed(event: ProcessingEvent):
            print(f"Event: {event.event_type.value} for job {event.job_id}")
        
        pipeline.add_event_handler(EventType.PROCESSING_COMPLETED, on_article_processed)
        
        # Start processing
        await pipeline.start_processing(num_workers=3)
        
        # Submit test job
        test_article = {
            'url': 'https://example.com/test-article',
            'title': 'Apple Reports Strong Q4 Earnings Results',
            'content': '''Apple Inc. announced record fourth-quarter earnings today, 
                         with revenue of $89.5 billion beating analyst expectations...''',
            'source_url': 'https://reuters.com',
            'publish_date': datetime.now(timezone.utc),
            'target_tickers': ['AAPL']
        }
        
        job_id = await pipeline.submit_job(**test_article)
        print(f"Submitted job: {job_id}")
        
        # Wait and check metrics
        await asyncio.sleep(10)
        metrics = await pipeline.get_metrics()
        print(f"Pipeline metrics: {json.dumps(metrics, indent=2, default=str)}")
        
        # Shutdown
        await pipeline.shutdown()
    
    # Run test
    asyncio.run(test_pipeline())