"""
Intelligent News Processor - Integration layer for the enhanced news processing pipeline.

This module integrates all the intelligent processing components:
- NewsDeduplicator for duplicate detection
- TickerRelevanceMatcher for ticker matching
- ArticleQualityAssessment for quality scoring
- AsyncProcessingPipeline for scalable processing
- MonitoringSystem for real-time monitoring

Provides a unified interface for the existing enhanced_news_scraper.py
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import structlog
import redis.asyncio as redis

from .news_deduplicator import NewsDeduplicator, DuplicateResult
from .ticker_relevance_matcher import TickerRelevanceMatcher, RelevanceScore
from .article_quality_assessment import ArticleQualityAssessment, QualityScore
from .async_processing_pipeline import AsyncProcessingPipeline, ProcessingPriority, ProcessingResult
from .monitoring_alerting import MonitoringSystem

logger = structlog.get_logger(__name__)


@dataclass
class ProcessedArticle:
    """Complete processed article with all analysis results"""
    # Original article data
    title: str
    url: str
    content: str
    source_domain: str
    publish_date: Optional[datetime]
    authors: List[str]
    
    # Processing results
    is_duplicate: bool
    duplicate_of: Optional[str]
    relevance_scores: List[RelevanceScore]
    quality_score: QualityScore
    processing_time_ms: int
    
    # Derived insights
    best_ticker_match: Optional[str]
    overall_relevance: float
    recommended_action: str  # "publish", "review", "reject"
    confidence_score: float
    
    # Metadata
    processed_at: datetime
    processor_version: str = "1.0.0"


class IntelligentNewsProcessor:
    """
    Unified intelligent news processing system.
    
    Integrates all processing components into a single, easy-to-use interface
    that can be integrated with the existing enhanced_news_scraper.py
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379/0",
                 enable_monitoring: bool = True,
                 max_concurrent_jobs: int = 20):
        
        self.redis_url = redis_url
        self.enable_monitoring = enable_monitoring
        self.max_concurrent_jobs = max_concurrent_jobs
        
        # Core components
        self.deduplicator: Optional[NewsDeduplicator] = None
        self.relevance_matcher: Optional[TickerRelevanceMatcher] = None
        self.quality_assessor: Optional[ArticleQualityAssessment] = None
        self.processing_pipeline: Optional[AsyncProcessingPipeline] = None
        self.monitoring_system: Optional[MonitoringSystem] = None
        
        # Redis client
        self.redis_client: Optional[redis.Redis] = None
        
        # Processing statistics
        self.stats = {
            'articles_processed': 0,
            'duplicates_filtered': 0,
            'high_quality_articles': 0,
            'processing_time_total': 0,
            'start_time': time.time()
        }
        
        # Configuration
        self.config = {
            'quality_threshold': 0.7,
            'relevance_threshold': 0.5,
            'duplicate_threshold': 0.85,
            'enable_async_processing': True,
            'max_articles_per_batch': 100,
            'processing_timeout_seconds': 300
        }
    
    async def initialize(self) -> bool:
        """Initialize all processing components"""
        try:
            logger.info("Initializing intelligent news processor")
            
            # Initialize Redis
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            # Initialize processing components
            sync_redis = await self._get_sync_redis()
            self.deduplicator = NewsDeduplicator(sync_redis)
            self.relevance_matcher = TickerRelevanceMatcher()
            self.quality_assessor = ArticleQualityAssessment()
            
            # Initialize async processing pipeline
            if self.config['enable_async_processing']:
                self.processing_pipeline = AsyncProcessingPipeline(
                    redis_url=self.redis_url,
                    max_concurrent_jobs=self.max_concurrent_jobs
                )
                await self.processing_pipeline.initialize()
            
            # Initialize monitoring system
            if self.enable_monitoring:
                self.monitoring_system = MonitoringSystem(self.redis_client)
                await self.monitoring_system.start_monitoring()
            
            logger.info("Intelligent news processor initialized successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to initialize intelligent news processor", error=str(e))
            return False
    
    async def _get_sync_redis(self):
        """Get synchronous Redis client for components that need it"""
        import redis as sync_redis
        return sync_redis.from_url(self.redis_url)
    
    async def process_articles_enhanced(self, articles_data: List[Dict[str, Any]], 
                                      target_tickers: List[str] = None) -> Dict[str, Any]:
        """
        Process articles with full intelligent analysis.
        
        This is the main integration point with enhanced_news_scraper.py
        
        Args:
            articles_data: List of article dictionaries from enhanced_news_scraper
            target_tickers: Optional list of tickers to focus analysis on
            
        Returns:
            Enhanced results dictionary compatible with existing format
        """
        start_time = time.time()
        
        try:
            logger.info("Starting intelligent article processing", 
                       article_count=len(articles_data),
                       target_tickers=target_tickers)
            
            # Process articles in batches
            processed_articles = []
            batch_size = min(self.config['max_articles_per_batch'], len(articles_data))
            
            for i in range(0, len(articles_data), batch_size):
                batch = articles_data[i:i + batch_size]
                
                if self.config['enable_async_processing']:
                    batch_results = await self._process_batch_async(batch, target_tickers)
                else:
                    batch_results = await self._process_batch_sync(batch, target_tickers)
                
                processed_articles.extend(batch_results)
            
            # Generate enhanced results
            enhanced_results = self._generate_enhanced_results(processed_articles, target_tickers)
            
            processing_time = time.time() - start_time
            self.stats['processing_time_total'] += processing_time
            
            logger.info("Intelligent processing completed", 
                       processed_count=len(processed_articles),
                       processing_time=f"{processing_time:.3f}s")
            
            return enhanced_results
            
        except Exception as e:
            logger.error("Intelligent article processing failed", error=str(e))
            return self._generate_error_results(articles_data, str(e))
    
    async def _process_batch_async(self, batch: List[Dict[str, Any]], 
                                 target_tickers: List[str]) -> List[ProcessedArticle]:
        """Process batch of articles using async pipeline"""
        if not self.processing_pipeline:
            # Fallback to sync processing
            return await self._process_batch_sync(batch, target_tickers)
        
        # Start pipeline if not running
        if not self.processing_pipeline.is_running:
            await self.processing_pipeline.start_processing(num_workers=5)
        
        # Submit jobs
        job_ids = []
        for article in batch:
            try:
                job_id = await self.processing_pipeline.submit_job(
                    article_url=article.get('url', ''),
                    title=article.get('title', ''),
                    content=article.get('full_content', article.get('content', '')),
                    source_url=article.get('source_domain', ''),
                    priority=ProcessingPriority.HIGH,
                    target_tickers=target_tickers,
                    publish_date=self._parse_date(article.get('date')),
                    author=', '.join(article.get('authors', [])),
                    metadata={'original_article': article}
                )
                job_ids.append((job_id, article))
                
            except Exception as e:
                logger.error("Failed to submit job", article_url=article.get('url'), error=str(e))
        
        # Wait for results with timeout
        processed_articles = []
        timeout = self.config['processing_timeout_seconds']
        
        for job_id, original_article in job_ids:
            result = await self._wait_for_job_result(job_id, timeout)
            
            if result:
                processed_article = self._create_processed_article(original_article, result)
                processed_articles.append(processed_article)
            else:
                # Create fallback result
                fallback_article = self._create_fallback_processed_article(original_article)
                processed_articles.append(fallback_article)
        
        return processed_articles
    
    async def _process_batch_sync(self, batch: List[Dict[str, Any]], 
                                target_tickers: List[str]) -> List[ProcessedArticle]:
        """Process batch of articles synchronously"""
        processed_articles = []
        
        for article in batch:
            try:
                processed_article = await self._process_single_article(article, target_tickers)
                processed_articles.append(processed_article)
                
            except Exception as e:
                logger.error("Failed to process article", 
                           article_url=article.get('url'), error=str(e))
                # Create fallback result
                fallback_article = self._create_fallback_processed_article(article)
                processed_articles.append(fallback_article)
        
        return processed_articles
    
    async def _process_single_article(self, article: Dict[str, Any], 
                                    target_tickers: List[str]) -> ProcessedArticle:
        """Process a single article through all analysis stages"""
        start_time = time.time()
        
        # Extract article data
        title = article.get('title', '')
        content = article.get('full_content', article.get('content', ''))
        url = article.get('url', '')
        source_domain = article.get('source_domain', '')
        publish_date = self._parse_date(article.get('date'))
        authors = article.get('authors', [])
        
        # Step 1: Duplicate detection
        fingerprint = self.deduplicator.create_content_fingerprint(url, title, content, publish_date)
        duplicate_result = self.deduplicator.check_duplicate(fingerprint)
        
        if not duplicate_result.is_duplicate:
            self.deduplicator.store_fingerprint(fingerprint)
        else:
            self.stats['duplicates_filtered'] += 1
        
        # Step 2: Ticker relevance analysis
        relevance_scores = self.relevance_matcher.analyze_article_relevance(
            title, content, target_tickers
        )
        
        # Step 3: Quality assessment
        quality_score = self.quality_assessor.assess_article_quality(
            title, content, url, publish_date, ', '.join(authors)
        )
        
        # Calculate derived insights
        best_ticker_match = None
        overall_relevance = 0.0
        
        if relevance_scores:
            best_score = max(relevance_scores, key=lambda x: x.relevance_score)
            best_ticker_match = best_score.ticker
            overall_relevance = best_score.relevance_score
        
        # Determine recommended action
        recommended_action = self._determine_recommendation(
            duplicate_result.is_duplicate, quality_score.overall_score, overall_relevance
        )
        
        # Calculate confidence
        confidence_score = self._calculate_confidence(quality_score, relevance_scores, duplicate_result)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Update stats
        self.stats['articles_processed'] += 1
        if quality_score.overall_score >= self.config['quality_threshold']:
            self.stats['high_quality_articles'] += 1
        
        return ProcessedArticle(
            title=title,
            url=url,
            content=content,
            source_domain=source_domain,
            publish_date=publish_date,
            authors=authors,
            is_duplicate=duplicate_result.is_duplicate,
            duplicate_of=duplicate_result.original_url,
            relevance_scores=relevance_scores,
            quality_score=quality_score,
            processing_time_ms=processing_time,
            best_ticker_match=best_ticker_match,
            overall_relevance=overall_relevance,
            recommended_action=recommended_action,
            confidence_score=confidence_score,
            processed_at=datetime.now(timezone.utc)
        )
    
    async def _wait_for_job_result(self, job_id: str, timeout: int) -> Optional[ProcessingResult]:
        """Wait for async job result with timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = await self.processing_pipeline.get_job_status(job_id)
            
            if result and result.get('status') in ['completed', 'failed']:
                return ProcessingResult(**result) if result.get('status') == 'completed' else None
            
            await asyncio.sleep(1)  # Check every second
        
        logger.warning("Job result timeout", job_id=job_id)
        return None
    
    def _create_processed_article(self, original_article: Dict[str, Any], 
                                result: ProcessingResult) -> ProcessedArticle:
        """Create ProcessedArticle from async processing result"""
        
        # Extract best ticker match and relevance
        best_ticker_match = None
        overall_relevance = 0.0
        
        if result.relevance_scores:
            best_score = max(result.relevance_scores, key=lambda x: x.relevance_score)
            best_ticker_match = best_score.ticker
            overall_relevance = best_score.relevance_score
        
        # Determine recommendation
        is_duplicate = result.duplicate_result.is_duplicate if result.duplicate_result else False
        quality_score_val = result.quality_score.overall_score if result.quality_score else 0.0
        
        recommended_action = self._determine_recommendation(is_duplicate, quality_score_val, overall_relevance)
        confidence_score = self._calculate_confidence(result.quality_score, result.relevance_scores, result.duplicate_result)
        
        return ProcessedArticle(
            title=original_article.get('title', ''),
            url=original_article.get('url', ''),
            content=original_article.get('full_content', original_article.get('content', '')),
            source_domain=original_article.get('source_domain', ''),
            publish_date=self._parse_date(original_article.get('date')),
            authors=original_article.get('authors', []),
            is_duplicate=is_duplicate,
            duplicate_of=result.duplicate_result.original_url if result.duplicate_result else None,
            relevance_scores=result.relevance_scores or [],
            quality_score=result.quality_score,
            processing_time_ms=result.processing_time_ms,
            best_ticker_match=best_ticker_match,
            overall_relevance=overall_relevance,
            recommended_action=recommended_action,
            confidence_score=confidence_score,
            processed_at=datetime.now(timezone.utc)
        )
    
    def _create_fallback_processed_article(self, article: Dict[str, Any]) -> ProcessedArticle:
        """Create fallback ProcessedArticle when processing fails"""
        return ProcessedArticle(
            title=article.get('title', ''),
            url=article.get('url', ''),
            content=article.get('full_content', article.get('content', '')),
            source_domain=article.get('source_domain', ''),
            publish_date=self._parse_date(article.get('date')),
            authors=article.get('authors', []),
            is_duplicate=False,
            duplicate_of=None,
            relevance_scores=[],
            quality_score=None,  # Will need to handle None case
            processing_time_ms=0,
            best_ticker_match=None,
            overall_relevance=0.0,
            recommended_action="review",
            confidence_score=0.0,
            processed_at=datetime.now(timezone.utc)
        )
    
    def _parse_date(self, date_input: Any) -> Optional[datetime]:
        """Parse date from various input formats"""
        if isinstance(date_input, datetime):
            return date_input
        elif isinstance(date_input, str):
            try:
                return datetime.fromisoformat(date_input.replace('Z', '+00:00'))
            except:
                return None
        return None
    
    def _determine_recommendation(self, is_duplicate: bool, quality_score: float, 
                                relevance_score: float) -> str:
        """Determine recommended action based on analysis results"""
        if is_duplicate:
            return "reject"
        
        if quality_score < 0.4 or relevance_score < 0.3:
            return "reject"
        elif quality_score >= self.config['quality_threshold'] and relevance_score >= self.config['relevance_threshold']:
            return "publish"
        else:
            return "review"
    
    def _calculate_confidence(self, quality_score, relevance_scores, duplicate_result) -> float:
        """Calculate overall confidence in the processing results"""
        confidence_factors = []
        
        # Quality confidence
        if quality_score and hasattr(quality_score, 'confidence'):
            confidence_factors.append(quality_score.confidence)
        else:
            confidence_factors.append(0.5)
        
        # Relevance confidence
        if relevance_scores:
            avg_confidence = sum(score.confidence for score in relevance_scores) / len(relevance_scores)
            confidence_factors.append(avg_confidence)
        else:
            confidence_factors.append(0.5)
        
        # Duplicate detection confidence
        if duplicate_result and duplicate_result.similarity_score > 0:
            confidence_factors.append(duplicate_result.similarity_score)
        else:
            confidence_factors.append(0.8)  # High confidence when no duplicates
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _generate_enhanced_results(self, processed_articles: List[ProcessedArticle], 
                                 target_tickers: List[str]) -> Dict[str, Any]:
        """Generate enhanced results compatible with existing enhanced_news_scraper format"""
        
        # Group by ticker for compatibility
        ticker_results = defaultdict(lambda: {
            'enhanced_articles': [],
            'article_count': 0,
            'sentiment_analysis': {
                'average_sentiment': 0.0,
                'sentiment_volatility': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            },
            'intelligence_insights': {
                'duplicate_articles_filtered': 0,
                'high_quality_articles': 0,
                'average_quality_score': 0.0,
                'average_relevance_score': 0.0,
                'recommended_for_publication': 0,
                'recommended_for_review': 0,
                'recommended_for_rejection': 0,
                'average_confidence': 0.0
            },
            'quality_metrics': {
                'average_credibility': 0.0,
                'total_words_analyzed': 0,
                'sources_count': 0,
                'processing_time_total_ms': 0
            }
        })
        
        # Process results for each ticker
        all_tickers = set(target_tickers) if target_tickers else set()
        
        # Add all found tickers from relevance analysis
        for article in processed_articles:
            for relevance_score in article.relevance_scores:
                all_tickers.add(relevance_score.ticker)
        
        for ticker in all_tickers:
            relevant_articles = []
            quality_scores = []
            relevance_scores = []
            confidence_scores = []
            processing_times = []
            
            for article in processed_articles:
                # Check if article is relevant to this ticker
                ticker_relevance = None
                for score in article.relevance_scores:
                    if score.ticker == ticker:
                        ticker_relevance = score
                        break
                
                if ticker_relevance and ticker_relevance.relevance_score >= 0.3:
                    # Convert to enhanced_articles format
                    enhanced_article = {
                        'title': article.title,
                        'url': article.url,
                        'date': article.publish_date.date().isoformat() if article.publish_date else None,
                        'full_content': article.content[:1000],  # Truncate for storage
                        'authors': article.authors,
                        'source_domain': article.source_domain,
                        'sentiment': {
                            'label': article.quality_score.quality_strengths[0] if article.quality_score and article.quality_score.quality_strengths else 'neutral',
                            'confidence': ticker_relevance.confidence,
                            'score': (ticker_relevance.relevance_score - 0.5) * 2,  # Convert to -1 to 1 scale
                            'ensemble_score': (ticker_relevance.relevance_score - 0.5) * 2,
                            'uncertainty': 1.0 - ticker_relevance.confidence
                        },
                        'intelligence_analysis': {
                            'is_duplicate': article.is_duplicate,
                            'duplicate_of': article.duplicate_of,
                            'relevance_score': ticker_relevance.relevance_score,
                            'quality_score': article.quality_score.overall_score if article.quality_score else 0.0,
                            'recommended_action': article.recommended_action,
                            'confidence': article.confidence_score,
                            'best_ticker_match': article.best_ticker_match,
                            'processing_time_ms': article.processing_time_ms
                        },
                        'quality_metrics': {
                            'readability_score': article.quality_score.readability_score if article.quality_score else 0.0,
                            'credibility_score': article.quality_score.source_credibility if article.quality_score else 0.0,
                            'factual_density': article.quality_score.factual_density if article.quality_score else 0.0,
                            'word_count': len(article.content.split()) if article.content else 0
                        }
                    }
                    
                    relevant_articles.append(enhanced_article)
                    
                    if article.quality_score:
                        quality_scores.append(article.quality_score.overall_score)
                    relevance_scores.append(ticker_relevance.relevance_score)
                    confidence_scores.append(article.confidence_score)
                    processing_times.append(article.processing_time_ms)
            
            # Calculate aggregate metrics
            if relevant_articles:
                # Sentiment analysis (using relevance as proxy)
                positive_articles = [a for a in relevant_articles if a['sentiment']['score'] > 0.1]
                negative_articles = [a for a in relevant_articles if a['sentiment']['score'] < -0.1]
                neutral_articles = [a for a in relevant_articles if -0.1 <= a['sentiment']['score'] <= 0.1]
                
                avg_sentiment = sum(a['sentiment']['score'] for a in relevant_articles) / len(relevant_articles)
                sentiment_scores = [a['sentiment']['score'] for a in relevant_articles]
                sentiment_volatility = statistics.stdev(sentiment_scores) if len(sentiment_scores) > 1 else 0.0
                
                # Intelligence insights
                duplicate_count = sum(1 for a in relevant_articles if a['intelligence_analysis']['is_duplicate'])
                high_quality_count = sum(1 for a in relevant_articles if a['intelligence_analysis']['quality_score'] >= self.config['quality_threshold'])
                
                action_counts = defaultdict(int)
                for a in relevant_articles:
                    action_counts[a['intelligence_analysis']['recommended_action']] += 1
                
                # Quality metrics
                credibility_scores = [a['quality_metrics']['credibility_score'] for a in relevant_articles]
                word_counts = [a['quality_metrics']['word_count'] for a in relevant_articles]
                sources = set(a['source_domain'] for a in relevant_articles)
                
                # Update ticker results
                ticker_results[ticker].update({
                    'enhanced_articles': relevant_articles,
                    'article_count': len(relevant_articles),
                    'sentiment_analysis': {
                        'average_sentiment': avg_sentiment,
                        'sentiment_volatility': sentiment_volatility,
                        'positive_count': len(positive_articles),
                        'negative_count': len(negative_articles),
                        'neutral_count': len(neutral_articles)
                    },
                    'intelligence_insights': {
                        'duplicate_articles_filtered': duplicate_count,
                        'high_quality_articles': high_quality_count,
                        'average_quality_score': statistics.mean(quality_scores) if quality_scores else 0.0,
                        'average_relevance_score': statistics.mean(relevance_scores),
                        'recommended_for_publication': action_counts['publish'],
                        'recommended_for_review': action_counts['review'],
                        'recommended_for_rejection': action_counts['reject'],
                        'average_confidence': statistics.mean(confidence_scores)
                    },
                    'quality_metrics': {
                        'average_credibility': statistics.mean(credibility_scores) if credibility_scores else 0.0,
                        'total_words_analyzed': sum(word_counts),
                        'sources_count': len(sources),
                        'processing_time_total_ms': sum(processing_times)
                    }
                })
        
        # Add global processing statistics
        results = dict(ticker_results)
        results['_processing_stats'] = {
            'total_articles_processed': len(processed_articles),
            'unique_articles': len([a for a in processed_articles if not a.is_duplicate]),
            'duplicates_filtered': len([a for a in processed_articles if a.is_duplicate]),
            'high_quality_articles': len([a for a in processed_articles if a.quality_score and a.quality_score.overall_score >= self.config['quality_threshold']]),
            'articles_recommended_for_publication': len([a for a in processed_articles if a.recommended_action == 'publish']),
            'processing_time_total_seconds': sum(a.processing_time_ms for a in processed_articles) / 1000.0,
            'processor_version': processed_articles[0].processor_version if processed_articles else "1.0.0",
            'processing_completed_at': datetime.now(timezone.utc).isoformat()
        }
        
        return results
    
    def _generate_error_results(self, articles_data: List[Dict[str, Any]], error_message: str) -> Dict[str, Any]:
        """Generate error results when processing fails"""
        return {
            '_error': {
                'message': error_message,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'article_count': len(articles_data)
            }
        }
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        runtime = time.time() - self.stats['start_time']
        
        stats = self.stats.copy()
        stats.update({
            'runtime_seconds': runtime,
            'articles_per_second': self.stats['articles_processed'] / max(1, runtime),
            'duplicate_rate': self.stats['duplicates_filtered'] / max(1, self.stats['articles_processed']),
            'high_quality_rate': self.stats['high_quality_articles'] / max(1, self.stats['articles_processed']),
            'avg_processing_time_ms': self.stats['processing_time_total'] * 1000 / max(1, self.stats['articles_processed'])
        })
        
        # Add pipeline stats if available
        if self.processing_pipeline:
            pipeline_stats = await self.processing_pipeline.get_metrics()
            stats['pipeline_metrics'] = pipeline_stats
        
        return stats
    
    async def shutdown(self):
        """Shutdown the intelligent news processor"""
        logger.info("Shutting down intelligent news processor")
        
        if self.processing_pipeline:
            await self.processing_pipeline.shutdown()
        
        if self.monitoring_system:
            await self.monitoring_system.stop_monitoring()
        
        if self.redis_client:
            await self.redis_client.aclose()
        
        logger.info("Intelligent news processor shutdown complete")


# Integration function for enhanced_news_scraper.py
async def process_enhanced_articles_with_intelligence(articles_data: List[Dict[str, Any]], 
                                                    target_tickers: List[str] = None,
                                                    redis_url: str = "redis://localhost:6379/0",
                                                    enable_monitoring: bool = False) -> Dict[str, Any]:
    """
    Main integration function for enhanced_news_scraper.py
    
    This function can be called from enhanced_scrape_headlines() to add
    intelligent processing capabilities to the existing scraping system.
    
    Args:
        articles_data: Article data from enhanced_news_scraper
        target_tickers: List of tickers to analyze
        redis_url: Redis connection URL
        enable_monitoring: Whether to enable monitoring
        
    Returns:
        Enhanced results with intelligent analysis
    """
    processor = IntelligentNewsProcessor(
        redis_url=redis_url,
        enable_monitoring=enable_monitoring
    )
    
    try:
        if await processor.initialize():
            return await processor.process_articles_enhanced(articles_data, target_tickers)
        else:
            logger.error("Failed to initialize intelligent processor, returning original data")
            return {'_error': 'initialization_failed'}
            
    except Exception as e:
        logger.error("Intelligent processing failed", error=str(e))
        return {'_error': str(e)}
    
    finally:
        await processor.shutdown()


# Utility function for testing
async def test_intelligent_processing():
    """Test function for the intelligent processing system"""
    
    # Sample article data (format from enhanced_news_scraper.py)
    test_articles = [
        {
            'title': 'Apple Reports Record Q4 Earnings, Beats Wall Street Expectations',
            'url': 'https://example.com/apple-earnings',
            'content': 'Apple Inc. announced record fourth-quarter earnings today...',
            'full_content': '''Apple Inc. reported exceptional fourth-quarter financial results today, 
                             with revenue of $89.5 billion and earnings per share of $1.46, both exceeding 
                             analyst expectations. The iPhone maker's strong performance was driven by robust 
                             iPhone 15 sales and continued growth in the services segment...''',
            'authors': ['Jane Smith', 'Tech Reporter'],
            'source_domain': 'reuters.com',
            'date': datetime.now(timezone.utc).isoformat(),
            'sentiment': {'score': 0.8, 'label': 'positive'}
        },
        {
            'title': 'Tesla Stock Surges on Production Milestone',
            'url': 'https://example.com/tesla-production', 
            'content': 'Tesla shares jumped 8% after the company announced...',
            'full_content': '''Tesla Motors announced it has reached a significant production milestone, 
                             manufacturing its 5 millionth vehicle at the Fremont factory. The achievement 
                             comes as the electric vehicle maker continues to scale production globally...''',
            'authors': ['Mike Johnson'],
            'source_domain': 'cnbc.com',
            'date': (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            'sentiment': {'score': 0.6, 'label': 'positive'}
        }
    ]
    
    # Test processing
    results = await process_enhanced_articles_with_intelligence(
        test_articles, 
        target_tickers=['AAPL', 'TSLA'],
        enable_monitoring=True
    )
    
    print("Intelligent Processing Results:")
    print(json.dumps(results, indent=2, default=str))
    
    return results


if __name__ == "__main__":
    # Run test
    asyncio.run(test_intelligent_processing())