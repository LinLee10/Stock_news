"""
Event-Driven Communication Infrastructure

Implements Kafka-based event bus with event sourcing, dead letter queues,
and comprehensive event schema management.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
import aiokafka
import aioredis
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer
import avro.schema
import avro.io
import io
import structlog

logger = structlog.get_logger(__name__)

class EventStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"

class EventPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class EventMetadata:
    """Event metadata for tracking and auditing"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: str = "1.0"
    source_service: str = ""
    priority: EventPriority = EventPriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class EventEnvelope:
    """Event envelope containing event data and metadata"""
    event_type: str
    data: Dict[str, Any]
    metadata: EventMetadata
    schema_version: str = "1.0"

class EventStore:
    """Event store for event sourcing"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.event_stream_prefix = "events:"
        self.snapshot_prefix = "snapshots:"
        
    async def append_event(self, stream_id: str, event: EventEnvelope) -> bool:
        """Append event to event stream"""
        
        try:
            stream_key = f"{self.event_stream_prefix}{stream_id}"
            event_data = json.dumps(asdict(event))
            
            # Use Redis Streams for event storage
            await self.redis.xadd(stream_key, {"event": event_data})
            
            # Set stream expiration (30 days)
            await self.redis.expire(stream_key, 30 * 24 * 3600)
            
            logger.info("Event appended to stream", stream_id=stream_id, event_id=event.metadata.event_id)
            return True
            
        except Exception as e:
            logger.error("Failed to append event to stream", error=str(e), stream_id=stream_id)
            return False
    
    async def read_stream(self, stream_id: str, from_id: str = "0") -> List[EventEnvelope]:
        """Read events from stream"""
        
        try:
            stream_key = f"{self.event_stream_prefix}{stream_id}"
            
            # Read from Redis Stream
            events = await self.redis.xrange(stream_key, min=from_id)
            
            event_envelopes = []
            for event_id, fields in events:
                event_data = json.loads(fields[b'event'].decode())
                
                # Reconstruct EventEnvelope
                metadata_data = event_data['metadata']
                metadata_data['priority'] = EventPriority(metadata_data['priority'])
                
                event_envelope = EventEnvelope(
                    event_type=event_data['event_type'],
                    data=event_data['data'],
                    metadata=EventMetadata(**metadata_data),
                    schema_version=event_data['schema_version']
                )
                
                event_envelopes.append(event_envelope)
            
            return event_envelopes
            
        except Exception as e:
            logger.error("Failed to read stream", error=str(e), stream_id=stream_id)
            return []
    
    async def create_snapshot(self, stream_id: str, snapshot_data: Dict[str, Any]) -> bool:
        """Create snapshot for event stream"""
        
        try:
            snapshot_key = f"{self.snapshot_prefix}{stream_id}"
            snapshot = {
                'data': snapshot_data,
                'timestamp': datetime.utcnow().isoformat(),
                'stream_id': stream_id
            }
            
            await self.redis.set(snapshot_key, json.dumps(snapshot), ex=30 * 24 * 3600)
            
            logger.info("Snapshot created", stream_id=stream_id)
            return True
            
        except Exception as e:
            logger.error("Failed to create snapshot", error=str(e), stream_id=stream_id)
            return False

class EventSchemaRegistry:
    """Event schema registry for managing event schemas"""
    
    def __init__(self):
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self._load_default_schemas()
    
    def _load_default_schemas(self):
        """Load default event schemas"""
        
        # News Article Event Schema
        self.schemas["news_article_discovered"] = {
            "version": "1.0",
            "schema": {
                "type": "record",
                "name": "NewsArticleEvent",
                "fields": [
                    {"name": "article_id", "type": "string"},
                    {"name": "title", "type": "string"},
                    {"name": "url", "type": "string"},
                    {"name": "source", "type": "string"},
                    {"name": "symbols", "type": {"type": "array", "items": "string"}},
                    {"name": "published_at", "type": ["null", "string"], "default": None},
                    {"name": "quality_score", "type": ["null", "double"], "default": None}
                ]
            }
        }
        
        # Price Update Event Schema
        self.schemas["price_data_updated"] = {
            "version": "1.0",
            "schema": {
                "type": "record",
                "name": "PriceUpdateEvent",
                "fields": [
                    {"name": "symbol", "type": "string"},
                    {"name": "price", "type": "double"},
                    {"name": "change_percent", "type": "double"},
                    {"name": "volume", "type": "long"},
                    {"name": "market_cap", "type": ["null", "double"], "default": None},
                    {"name": "timestamp", "type": "string"}
                ]
            }
        }
        
        # Sentiment Analysis Event Schema
        self.schemas["sentiment_analysis_completed"] = {
            "version": "1.0",
            "schema": {
                "type": "record",
                "name": "SentimentAnalysisEvent",
                "fields": [
                    {"name": "article_id", "type": "string"},
                    {"name": "symbol", "type": "string"},
                    {"name": "sentiment_score", "type": "double"},
                    {"name": "confidence", "type": "double"},
                    {"name": "analysis_model", "type": "string"},
                    {"name": "processing_time_ms", "type": ["null", "double"], "default": None}
                ]
            }
        }
        
        # Recommendation Event Schema
        self.schemas["recommendation_generated"] = {
            "version": "1.0",
            "schema": {
                "type": "record",
                "name": "RecommendationEvent",
                "fields": [
                    {"name": "symbol", "type": "string"},
                    {"name": "action", "type": "string"},
                    {"name": "confidence", "type": "double"},
                    {"name": "reasoning", "type": "string"},
                    {"name": "price_target", "type": ["null", "double"], "default": None},
                    {"name": "risk_score", "type": ["null", "double"], "default": None}
                ]
            }
        }
    
    def get_schema(self, event_type: str, version: str = "1.0") -> Optional[Dict[str, Any]]:
        """Get schema for event type"""
        return self.schemas.get(event_type)
    
    def register_schema(self, event_type: str, schema: Dict[str, Any], version: str = "1.0"):
        """Register new event schema"""
        self.schemas[event_type] = {
            "version": version,
            "schema": schema
        }
        logger.info("Schema registered", event_type=event_type, version=version)
    
    def validate_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """Validate event data against schema"""
        
        schema_def = self.get_schema(event_type)
        if not schema_def:
            logger.warning("No schema found for event type", event_type=event_type)
            return False
        
        try:
            # Simple validation (in production, use proper Avro validation)
            schema_fields = {field["name"]: field for field in schema_def["schema"]["fields"]}
            
            for field_name, field_def in schema_fields.items():
                if field_name not in event_data:
                    if "default" not in field_def:
                        logger.error("Required field missing", field=field_name, event_type=event_type)
                        return False
            
            return True
            
        except Exception as e:
            logger.error("Schema validation failed", error=str(e), event_type=event_type)
            return False

class EventBus:
    """Event bus for publish/subscribe messaging"""
    
    def __init__(self, kafka_bootstrap_servers: str = "localhost:9092",
                 redis_url: str = "redis://localhost:6379"):
        
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.redis_url = redis_url
        
        # Kafka clients
        self.producer: Optional[aiokafka.AIOKafkaProducer] = None
        self.consumer: Optional[aiokafka.AIOKafkaConsumer] = None
        
        # Redis client for event store
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Components
        self.event_store: Optional[EventStore] = None
        self.schema_registry = EventSchemaRegistry()
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Dead letter queue
        self.dlq_topic_suffix = ".dead_letter"
        
        # Metrics
        self.events_published = 0
        self.events_consumed = 0
        self.events_failed = 0

    async def initialize(self):
        """Initialize event bus components"""
        
        logger.info("Initializing event bus")
        
        try:
            # Initialize Kafka producer
            self.producer = aiokafka.AIOKafkaProducer(
                bootstrap_servers=self.kafka_bootstrap_servers,
                value_serializer=self._serialize_event,
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                retry_backoff_ms=100,
                request_timeout_ms=30000,
                enable_idempotence=True,
                acks='all'
            )
            await self.producer.start()
            
            # Initialize Redis client
            self.redis_client = aioredis.from_url(self.redis_url)
            
            # Initialize event store
            self.event_store = EventStore(self.redis_client)
            
            logger.info("Event bus initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize event bus", error=str(e))
            raise

    def _serialize_event(self, event_envelope: EventEnvelope) -> bytes:
        """Serialize event envelope for Kafka"""
        return json.dumps(asdict(event_envelope), default=str).encode('utf-8')

    def _deserialize_event(self, data: bytes) -> EventEnvelope:
        """Deserialize event envelope from Kafka"""
        
        event_data = json.loads(data.decode('utf-8'))
        
        # Reconstruct metadata
        metadata_data = event_data['metadata']
        metadata_data['priority'] = EventPriority(metadata_data['priority'])
        
        return EventEnvelope(
            event_type=event_data['event_type'],
            data=event_data['data'],
            metadata=EventMetadata(**metadata_data),
            schema_version=event_data['schema_version']
        )

    async def publish_event(self, topic: str, event_type: str, event_data: Dict[str, Any],
                          correlation_id: Optional[str] = None, 
                          source_service: str = "", 
                          priority: EventPriority = EventPriority.NORMAL,
                          key: Optional[str] = None) -> bool:
        """Publish event to topic"""
        
        try:
            # Validate event data
            if not self.schema_registry.validate_event(event_type, event_data):
                logger.error("Event validation failed", event_type=event_type)
                return False
            
            # Create event metadata
            metadata = EventMetadata(
                correlation_id=correlation_id,
                source_service=source_service,
                priority=priority
            )
            
            # Create event envelope
            event_envelope = EventEnvelope(
                event_type=event_type,
                data=event_data,
                metadata=metadata
            )
            
            # Store in event store for audit trail
            if self.event_store:
                stream_id = f"{source_service}_{event_type}"
                await self.event_store.append_event(stream_id, event_envelope)
            
            # Publish to Kafka
            await self.producer.send(topic, event_envelope, key=key)
            
            self.events_published += 1
            
            logger.info(
                "Event published",
                topic=topic,
                event_type=event_type,
                event_id=metadata.event_id,
                correlation_id=correlation_id
            )
            
            return True
            
        except Exception as e:
            self.events_failed += 1
            logger.error(
                "Failed to publish event",
                topic=topic,
                event_type=event_type,
                error=str(e)
            )
            return False

    def subscribe(self, event_type: str, handler: Callable[[EventEnvelope], None]):
        """Subscribe to event type"""
        
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        logger.info("Event handler subscribed", event_type=event_type)

    async def start_consumer(self, topics: List[str], group_id: str):
        """Start event consumer"""
        
        logger.info("Starting event consumer", topics=topics, group_id=group_id)
        
        self.consumer = aiokafka.AIOKafkaConsumer(
            *topics,
            bootstrap_servers=self.kafka_bootstrap_servers,
            group_id=group_id,
            value_deserializer=self._deserialize_event,
            auto_offset_reset='latest',
            enable_auto_commit=False  # Manual commit for better control
        )
        
        await self.consumer.start()
        
        try:
            async for message in self.consumer:
                await self._process_message(message)
                
        except Exception as e:
            logger.error("Consumer loop failed", error=str(e))
        finally:
            await self.consumer.stop()

    async def _process_message(self, message):
        """Process incoming message"""
        
        event_envelope = message.value
        correlation_id = event_envelope.metadata.correlation_id
        
        try:
            # Check if we have handlers for this event type
            handlers = self.event_handlers.get(event_envelope.event_type, [])
            
            if not handlers:
                logger.warning(
                    "No handlers for event type",
                    event_type=event_envelope.event_type,
                    correlation_id=correlation_id
                )
                await self.consumer.commit()
                return
            
            # Process with all handlers
            for handler in handlers:
                try:
                    await handler(event_envelope)
                except Exception as e:
                    logger.error(
                        "Event handler failed",
                        event_type=event_envelope.event_type,
                        handler=handler.__name__,
                        error=str(e),
                        correlation_id=correlation_id
                    )
                    
                    # Increment retry count
                    event_envelope.metadata.retry_count += 1
                    
                    # Send to dead letter queue if max retries exceeded
                    if event_envelope.metadata.retry_count >= event_envelope.metadata.max_retries:
                        await self._send_to_dead_letter_queue(message.topic, event_envelope, str(e))
                    else:
                        # Retry with exponential backoff
                        delay = 2 ** event_envelope.metadata.retry_count
                        await asyncio.sleep(min(delay, 60))  # Max 60 seconds delay
                        
                        # Republish for retry
                        await self.publish_event(
                            message.topic,
                            event_envelope.event_type,
                            event_envelope.data,
                            correlation_id=correlation_id,
                            source_service="event_bus_retry",
                            priority=event_envelope.metadata.priority
                        )
                    
                    continue  # Don't commit if handler failed
            
            # Commit message after successful processing
            await self.consumer.commit()
            self.events_consumed += 1
            
            logger.info(
                "Event processed successfully",
                event_type=event_envelope.event_type,
                correlation_id=correlation_id
            )
            
        except Exception as e:
            self.events_failed += 1
            logger.error(
                "Message processing failed",
                event_type=event_envelope.event_type,
                correlation_id=correlation_id,
                error=str(e)
            )
            
            # Send to dead letter queue
            await self._send_to_dead_letter_queue(message.topic, event_envelope, str(e))
            
            # Commit to avoid reprocessing
            await self.consumer.commit()

    async def _send_to_dead_letter_queue(self, original_topic: str, 
                                       event_envelope: EventEnvelope, error: str):
        """Send failed event to dead letter queue"""
        
        dlq_topic = f"{original_topic}{self.dlq_topic_suffix}"
        
        # Add error information to event data
        dlq_event_data = {
            **event_envelope.data,
            'dlq_metadata': {
                'original_topic': original_topic,
                'error': error,
                'failed_at': datetime.utcnow().isoformat(),
                'retry_count': event_envelope.metadata.retry_count
            }
        }
        
        # Update metadata
        event_envelope.metadata.retry_count = 0  # Reset for DLQ
        event_envelope.data = dlq_event_data
        
        try:
            await self.producer.send(dlq_topic, event_envelope)
            
            logger.warning(
                "Event sent to dead letter queue",
                original_topic=original_topic,
                dlq_topic=dlq_topic,
                event_type=event_envelope.event_type,
                error=error
            )
            
        except Exception as e:
            logger.error(
                "Failed to send to dead letter queue",
                dlq_topic=dlq_topic,
                error=str(e)
            )

    async def replay_events(self, stream_id: str, from_timestamp: Optional[datetime] = None) -> int:
        """Replay events from event store"""
        
        if not self.event_store:
            logger.error("Event store not available for replay")
            return 0
        
        try:
            # Calculate from_id based on timestamp
            from_id = "0"
            if from_timestamp:
                # Convert timestamp to Redis Stream ID format
                from_id = f"{int(from_timestamp.timestamp() * 1000)}-0"
            
            events = await self.event_store.read_stream(stream_id, from_id)
            
            replayed_count = 0
            for event in events:
                # Republish event
                success = await self.publish_event(
                    f"replay_{stream_id}",
                    event.event_type,
                    event.data,
                    correlation_id=f"replay_{event.metadata.event_id}",
                    source_service="event_replay",
                    priority=EventPriority.LOW
                )
                
                if success:
                    replayed_count += 1
            
            logger.info("Events replayed", stream_id=stream_id, count=replayed_count)
            return replayed_count
            
        except Exception as e:
            logger.error("Event replay failed", stream_id=stream_id, error=str(e))
            return 0

    async def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics"""
        
        return {
            'events_published': self.events_published,
            'events_consumed': self.events_consumed,
            'events_failed': self.events_failed,
            'registered_event_types': list(self.event_handlers.keys()),
            'schema_count': len(self.schema_registry.schemas),
            'kafka_connected': self.producer is not None,
            'redis_connected': self.redis_client is not None
        }

    async def cleanup(self):
        """Cleanup resources"""
        
        logger.info("Cleaning up event bus")
        
        if self.producer:
            await self.producer.stop()
        
        if self.consumer:
            await self.consumer.stop()
        
        if self.redis_client:
            await self.redis_client.close()

# Global event bus instance
event_bus: Optional[EventBus] = None

async def get_event_bus() -> EventBus:
    """Get global event bus instance"""
    
    global event_bus
    
    if event_bus is None:
        event_bus = EventBus()
        await event_bus.initialize()
    
    return event_bus

# Decorator for event handlers
def event_handler(event_type: str):
    """Decorator for registering event handlers"""
    
    def decorator(func):
        async def wrapper(event_envelope: EventEnvelope):
            return await func(event_envelope.data, event_envelope.metadata)
        
        # Register handler when event bus is available
        async def register_when_ready():
            bus = await get_event_bus()
            bus.subscribe(event_type, wrapper)
        
        asyncio.create_task(register_when_ready())
        return wrapper
    
    return decorator

# Example usage:
# @event_handler("news_article_discovered")
# async def handle_news_article(event_data: Dict[str, Any], metadata: EventMetadata):
#     logger.info("Processing news article", article_id=event_data.get('article_id'))
#     # Process the event...