"""
Base Microservice Class with Common Infrastructure

Provides common functionality for all microservices including:
- Health checks, metrics, logging, configuration
- Event-driven messaging, circuit breakers, observability
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import aiohttp
import aioredis
import aiokafka
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from contextlib import asynccontextmanager
from pathlib import Path
import os

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.add_log_level,
        structlog.processors.add_logger_name,
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(30),  # INFO level
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class ServiceStatus(Enum):
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SHUTTING_DOWN = "shutting_down"

@dataclass
class ServiceConfig:
    """Common service configuration"""
    service_name: str
    service_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8080
    
    # Messaging configuration
    kafka_bootstrap_servers: str = "localhost:9092"
    redis_url: str = "redis://localhost:6379"
    
    # Observability configuration
    metrics_port: int = 9090
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    
    # Feature flags configuration
    feature_flag_service_url: str = "http://localhost:8081"
    
    # Circuit breaker configuration
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    circuit_breaker_expected_exception: tuple = (Exception,)

@dataclass
class HealthCheckResult:
    """Health check result"""
    service_name: str
    status: ServiceStatus
    timestamp: datetime
    checks: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    dependencies: Dict[str, bool] = field(default_factory=dict)

class CircuitBreaker:
    """Circuit breaker implementation for resilience"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == "OPEN":
            if (datetime.now() - self.last_failure_time).seconds > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e

class BaseService(ABC):
    """Base class for all microservices"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.status = ServiceStatus.STARTING
        self.correlation_id = str(uuid.uuid4())
        
        # Initialize logging with service context
        self.logger = structlog.get_logger().bind(
            service=config.service_name,
            version=config.service_version
        )
        
        # Initialize tracing
        self._setup_tracing()
        self.tracer = trace.get_tracer(config.service_name)
        
        # Initialize metrics
        self._setup_metrics()
        
        # Initialize messaging clients
        self.kafka_producer: Optional[aiokafka.AIOKafkaProducer] = None
        self.kafka_consumer: Optional[aiokafka.AIOKafkaConsumer] = None
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Initialize HTTP client
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        # Circuit breakers for external dependencies
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Background tasks
        self._background_tasks: set = set()

    def _setup_tracing(self):
        """Setup distributed tracing with Jaeger"""
        
        trace.set_tracer_provider(TracerProvider())
        
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Auto-instrument HTTP client
        AioHttpClientInstrumentor().instrument()

    def _setup_metrics(self):
        """Setup Prometheus metrics"""
        
        self.metrics = {
            'requests_total': Counter(
                'requests_total',
                'Total requests processed',
                ['service', 'method', 'endpoint', 'status']
            ),
            'request_duration': Histogram(
                'request_duration_seconds',
                'Request duration in seconds',
                ['service', 'method', 'endpoint']
            ),
            'active_connections': Gauge(
                'active_connections',
                'Number of active connections',
                ['service']
            ),
            'events_processed': Counter(
                'events_processed_total',
                'Total events processed',
                ['service', 'event_type', 'status']
            ),
            'circuit_breaker_state': Gauge(
                'circuit_breaker_state',
                'Circuit breaker state (0=CLOSED, 1=OPEN, 2=HALF_OPEN)',
                ['service', 'dependency']
            )
        }

    async def start(self):
        """Start the microservice"""
        
        with self.tracer.start_as_current_span("service_startup"):
            self.logger.info("Starting microservice", service=self.config.service_name)
            
            try:
                # Start metrics server
                start_http_server(self.config.metrics_port)
                
                # Initialize messaging
                await self._initialize_messaging()
                
                # Initialize HTTP client
                await self._initialize_http_client()
                
                # Service-specific initialization
                await self.initialize()
                
                # Start health checks
                asyncio.create_task(self._health_check_loop())
                
                # Start message consumers
                asyncio.create_task(self._message_consumer_loop())
                
                self.status = ServiceStatus.HEALTHY
                
                self.logger.info("Microservice started successfully")
                
            except Exception as e:
                self.status = ServiceStatus.UNHEALTHY
                self.logger.error("Failed to start microservice", error=str(e))
                raise

    async def _initialize_messaging(self):
        """Initialize Kafka and Redis messaging"""
        
        # Initialize Kafka producer
        self.kafka_producer = aiokafka.AIOKafkaProducer(
            bootstrap_servers=self.config.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        await self.kafka_producer.start()
        
        # Initialize Kafka consumer
        self.kafka_consumer = aiokafka.AIOKafkaConsumer(
            bootstrap_servers=self.config.kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id=f"{self.config.service_name}_group"
        )
        
        # Subscribe to topics (implemented by subclasses)
        topics = await self.get_subscribed_topics()
        if topics:
            self.kafka_consumer.subscribe(topics)
            await self.kafka_consumer.start()
        
        # Initialize Redis client
        self.redis_client = aioredis.from_url(self.config.redis_url)
        
        self.logger.info("Messaging initialized successfully")

    async def _initialize_http_client(self):
        """Initialize HTTP client with timeouts and retries"""
        
        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        self.http_session = aiohttp.ClientSession(timeout=timeout)

    async def _health_check_loop(self):
        """Background health check loop"""
        
        while self.status != ServiceStatus.SHUTTING_DOWN:
            try:
                health_result = await self.health_check()
                
                # Update service status based on health check
                if all(health_result.checks.values()) and all(health_result.dependencies.values()):
                    if self.status != ServiceStatus.HEALTHY:
                        self.status = ServiceStatus.HEALTHY
                elif any(not check for check in health_result.dependencies.values()):
                    self.status = ServiceStatus.DEGRADED
                else:
                    self.status = ServiceStatus.UNHEALTHY
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error("Health check failed", error=str(e))
                self.status = ServiceStatus.UNHEALTHY
                await asyncio.sleep(30)

    async def _message_consumer_loop(self):
        """Background message consumer loop"""
        
        if not self.kafka_consumer:
            return
            
        try:
            async for message in self.kafka_consumer:
                correlation_id = str(uuid.uuid4())
                
                with self.tracer.start_as_current_span("process_message") as span:
                    span.set_attribute("correlation_id", correlation_id)
                    span.set_attribute("topic", message.topic)
                    
                    start_time = datetime.now()
                    
                    try:
                        await self.process_message(message.topic, message.value, correlation_id)
                        
                        self.metrics['events_processed'].labels(
                            service=self.config.service_name,
                            event_type=message.topic,
                            status='success'
                        ).inc()
                        
                    except Exception as e:
                        self.logger.error(
                            "Message processing failed",
                            correlation_id=correlation_id,
                            topic=message.topic,
                            error=str(e)
                        )
                        
                        self.metrics['events_processed'].labels(
                            service=self.config.service_name,
                            event_type=message.topic,
                            status='error'
                        ).inc()
                        
                        # Send to dead letter queue
                        await self._send_to_dead_letter_queue(message, str(e))
                    
                    finally:
                        duration = (datetime.now() - start_time).total_seconds()
                        self.metrics['request_duration'].labels(
                            service=self.config.service_name,
                            method='CONSUME',
                            endpoint=message.topic
                        ).observe(duration)
                        
        except Exception as e:
            self.logger.error("Message consumer loop failed", error=str(e))

    async def _send_to_dead_letter_queue(self, message, error: str):
        """Send failed message to dead letter queue"""
        
        dead_letter_topic = f"{message.topic}.dead_letter"
        
        dead_letter_message = {
            'original_topic': message.topic,
            'original_message': message.value,
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'service': self.config.service_name
        }
        
        try:
            await self.kafka_producer.send(dead_letter_topic, dead_letter_message)
            self.logger.info("Message sent to dead letter queue", topic=dead_letter_topic)
        except Exception as e:
            self.logger.error("Failed to send to dead letter queue", error=str(e))

    async def publish_event(self, topic: str, event: Dict[str, Any], 
                          key: Optional[str] = None, correlation_id: Optional[str] = None):
        """Publish event to Kafka topic"""
        
        correlation_id = correlation_id or str(uuid.uuid4())
        
        # Add metadata to event
        event_with_metadata = {
            **event,
            'metadata': {
                'correlation_id': correlation_id,
                'timestamp': datetime.now().isoformat(),
                'service': self.config.service_name,
                'version': self.config.service_version
            }
        }
        
        with self.tracer.start_as_current_span("publish_event") as span:
            span.set_attribute("topic", topic)
            span.set_attribute("correlation_id", correlation_id)
            
            try:
                await self.kafka_producer.send(topic, event_with_metadata, key=key)
                
                self.logger.info(
                    "Event published",
                    topic=topic,
                    correlation_id=correlation_id,
                    event_type=event.get('event_type', 'unknown')
                )
                
            except Exception as e:
                self.logger.error(
                    "Failed to publish event",
                    topic=topic,
                    correlation_id=correlation_id,
                    error=str(e)
                )
                raise

    async def call_service(self, service_name: str, endpoint: str, 
                          method: str = "GET", data: Optional[Dict] = None,
                          timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Call another microservice with circuit breaker protection"""
        
        circuit_breaker = self.circuit_breakers.get(service_name)
        if not circuit_breaker:
            circuit_breaker = CircuitBreaker(
                self.config.circuit_breaker_failure_threshold,
                self.config.circuit_breaker_recovery_timeout
            )
            self.circuit_breakers[service_name] = circuit_breaker
        
        async def make_request():
            # Service discovery (simplified - would use proper service discovery)
            service_url = f"http://{service_name}:8080"
            url = f"{service_url}{endpoint}"
            
            async with self.http_session.request(method, url, json=data, timeout=timeout) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Service call failed: {response.status}")
        
        with self.tracer.start_as_current_span("service_call") as span:
            span.set_attribute("target_service", service_name)
            span.set_attribute("endpoint", endpoint)
            
            try:
                result = await circuit_breaker.call(make_request)
                
                # Update circuit breaker metrics
                self.metrics['circuit_breaker_state'].labels(
                    service=self.config.service_name,
                    dependency=service_name
                ).set(0 if circuit_breaker.state == "CLOSED" else 1 if circuit_breaker.state == "OPEN" else 2)
                
                return result
                
            except Exception as e:
                self.logger.error(
                    "Service call failed",
                    target_service=service_name,
                    endpoint=endpoint,
                    error=str(e)
                )
                
                # Update circuit breaker metrics
                self.metrics['circuit_breaker_state'].labels(
                    service=self.config.service_name,
                    dependency=service_name
                ).set(1 if circuit_breaker.state == "OPEN" else 2)
                
                return None

    async def get_feature_flag(self, flag_name: str, user_context: Dict[str, Any] = None) -> bool:
        """Get feature flag value from feature flag service"""
        
        try:
            response = await self.call_service(
                "feature-flag-service",
                f"/flags/{flag_name}",
                method="POST",
                data={'user_context': user_context or {}}
            )
            
            if response:
                return response.get('enabled', False)
            else:
                # Default to False if service is unavailable
                return False
                
        except Exception as e:
            self.logger.warning(
                "Feature flag check failed, using default",
                flag_name=flag_name,
                error=str(e)
            )
            return False

    # Abstract methods to be implemented by subclasses
    
    @abstractmethod
    async def initialize(self):
        """Initialize service-specific components"""
        pass

    @abstractmethod
    async def get_subscribed_topics(self) -> List[str]:
        """Return list of Kafka topics to subscribe to"""
        pass

    @abstractmethod
    async def process_message(self, topic: str, message: Dict[str, Any], correlation_id: str):
        """Process incoming message from Kafka"""
        pass

    @abstractmethod
    async def health_check(self) -> HealthCheckResult:
        """Perform health check and return result"""
        pass

    # Utility methods
    
    async def shutdown(self):
        """Graceful shutdown"""
        
        self.status = ServiceStatus.SHUTTING_DOWN
        self.logger.info("Shutting down microservice")
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Close messaging connections
        if self.kafka_producer:
            await self.kafka_producer.stop()
        
        if self.kafka_consumer:
            await self.kafka_consumer.stop()
        
        if self.redis_client:
            await self.redis_client.close()
        
        # Close HTTP session
        if self.http_session:
            await self.http_session.close()
        
        self.logger.info("Microservice shutdown complete")

    @asynccontextmanager
    async def correlation_context(self, correlation_id: Optional[str] = None):
        """Context manager for correlation ID tracking"""
        
        correlation_id = correlation_id or str(uuid.uuid4())
        
        # Bind correlation ID to logger
        bound_logger = self.logger.bind(correlation_id=correlation_id)
        
        # Start tracing span
        with self.tracer.start_as_current_span("operation") as span:
            span.set_attribute("correlation_id", correlation_id)
            
            try:
                yield bound_logger, correlation_id
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
            finally:
                span.set_status(trace.Status(trace.StatusCode.OK))

    def create_background_task(self, coro):
        """Create and track background task"""
        
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

# Event schemas for type safety and documentation

@dataclass
class BaseEvent:
    """Base event structure"""
    event_type: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class NewsArticleEvent(BaseEvent):
    """News article discovered event"""
    event_type: str = "news_article_discovered"
    article_id: str = ""
    title: str = ""
    url: str = ""
    source: str = ""
    symbols: List[str] = field(default_factory=list)
    published_at: Optional[str] = None

@dataclass
class PriceUpdateEvent(BaseEvent):
    """Price data update event"""
    event_type: str = "price_data_updated"
    symbol: str = ""
    price: float = 0.0
    change_percent: float = 0.0
    volume: int = 0
    market_cap: Optional[float] = None

@dataclass
class SentimentAnalysisEvent(BaseEvent):
    """Sentiment analysis completed event"""
    event_type: str = "sentiment_analysis_completed"
    article_id: str = ""
    symbol: str = ""
    sentiment_score: float = 0.0
    confidence: float = 0.0
    analysis_model: str = "finbert"

@dataclass
class RecommendationEvent(BaseEvent):
    """Investment recommendation generated event"""
    event_type: str = "recommendation_generated"
    symbol: str = ""
    action: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    price_target: Optional[float] = None

# Service registry for service discovery
class ServiceRegistry:
    """Simple service registry for development"""
    
    def __init__(self):
        self._services: Dict[str, Dict[str, Any]] = {}
    
    def register(self, service_name: str, host: str, port: int, health_endpoint: str = "/health"):
        """Register a service"""
        self._services[service_name] = {
            'host': host,
            'port': port,
            'health_endpoint': health_endpoint,
            'registered_at': datetime.now().isoformat()
        }
    
    def discover(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Discover a service"""
        return self._services.get(service_name)
    
    def list_services(self) -> Dict[str, Dict[str, Any]]:
        """List all registered services"""
        return self._services.copy()

# Global service registry instance
service_registry = ServiceRegistry()