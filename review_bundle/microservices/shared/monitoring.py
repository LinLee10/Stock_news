"""
Comprehensive Monitoring and Observability System
Prometheus metrics, Grafana dashboards, distributed tracing, and health checks
"""

import time
import asyncio
import json
import traceback
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import asynccontextmanager

import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
import redis

logger = structlog.get_logger(__name__)


class ServiceHealth(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Health check result"""
    name: str
    status: ServiceHealth
    message: str
    timestamp: datetime
    response_time_ms: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: datetime
    unit: str = "count"


@dataclass
class Alert:
    """System alert"""
    id: str
    service: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class PrometheusMetrics:
    """Prometheus metrics collection"""
    
    def __init__(self, service_name: str, registry: Optional[CollectorRegistry] = None):
        self.service_name = service_name
        self.registry = registry or CollectorRegistry()
        
        # Core metrics
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint', 'status_code', 'service'],
            registry=self.registry
        )
        
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code', 'service'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'active_connections',
            'Active connections',
            ['service'],
            registry=self.registry
        )
        
        self.error_count = Counter(
            'errors_total',
            'Total errors',
            ['service', 'error_type', 'severity'],
            registry=self.registry
        )
        
        # Business metrics
        self.business_events = Counter(
            'business_events_total',
            'Business events',
            ['service', 'event_type', 'status'],
            registry=self.registry
        )
        
        self.processing_duration = Histogram(
            'processing_duration_seconds',
            'Processing duration in seconds',
            ['service', 'operation'],
            registry=self.registry
        )
        
        self.queue_size = Gauge(
            'queue_size',
            'Queue size',
            ['service', 'queue_name'],
            registry=self.registry
        )
        
        # System metrics
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            ['service'],
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            ['service'],
            registry=self.registry
        )
        
        self.database_connections = Gauge(
            'database_connections_active',
            'Active database connections',
            ['service', 'database'],
            registry=self.registry
        )
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        labels = {
            'method': method,
            'endpoint': endpoint,
            'status_code': str(status_code),
            'service': self.service_name
        }
        
        self.request_duration.labels(**labels).observe(duration)
        self.request_count.labels(**labels).inc()
    
    def record_error(self, error_type: str, severity: str = "error"):
        """Record error metric"""
        self.error_count.labels(
            service=self.service_name,
            error_type=error_type,
            severity=severity
        ).inc()
    
    def record_business_event(self, event_type: str, status: str = "success"):
        """Record business event"""
        self.business_events.labels(
            service=self.service_name,
            event_type=event_type,
            status=status
        ).inc()
    
    def set_queue_size(self, queue_name: str, size: int):
        """Set queue size metric"""
        self.queue_size.labels(
            service=self.service_name,
            queue_name=queue_name
        ).set(size)
    
    def set_active_connections(self, count: int):
        """Set active connections metric"""
        self.active_connections.labels(service=self.service_name).set(count)
    
    @asynccontextmanager
    async def measure_operation(self, operation: str):
        """Context manager to measure operation duration"""
        start_time = time.time()
        try:
            yield
        except Exception as e:
            self.record_error(f"{operation}_failed", "error")
            raise
        finally:
            duration = time.time() - start_time
            self.processing_duration.labels(
                service=self.service_name,
                operation=operation
            ).observe(duration)
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest(self.registry).decode('utf-8')


class DistributedTracing:
    """Distributed tracing with Jaeger"""
    
    def __init__(self, service_name: str, jaeger_endpoint: str):
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint
        
        # Configure tracing
        trace.set_tracer_provider(TracerProvider())
        tracer_provider = trace.get_tracer_provider()
        
        # Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
            collector_endpoint=jaeger_endpoint,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        # Auto-instrumentation
        RequestsInstrumentor().instrument()
        RedisInstrumentor().instrument()
        
        self.tracer = trace.get_tracer(service_name)
    
    @asynccontextmanager
    async def trace_operation(self, operation_name: str, attributes: Dict[str, Any] = None):
        """Trace an async operation"""
        with self.tracer.start_as_current_span(operation_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            try:
                yield span
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                span.set_attribute("error.type", type(e).__name__)
                raise
    
    def add_span_attributes(self, span, attributes: Dict[str, Any]):
        """Add attributes to current span"""
        for key, value in attributes.items():
            span.set_attribute(key, value)


class HealthChecker:
    """Comprehensive health checking system"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.health_checks = {}
        self.last_results = {}
    
    def register_check(self, name: str, check_func: Callable[[], bool], 
                      interval: int = 30, timeout: int = 10):
        """Register a health check"""
        self.health_checks[name] = {
            'func': check_func,
            'interval': interval,
            'timeout': timeout,
            'last_run': None
        }
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check"""
        if name not in self.health_checks:
            return HealthCheckResult(
                name=name,
                status=ServiceHealth.UNKNOWN,
                message="Health check not found",
                timestamp=datetime.utcnow(),
                response_time_ms=0
            )
        
        check = self.health_checks[name]
        start_time = time.time()
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                asyncio.create_task(self._run_check_func(check['func'])),
                timeout=check['timeout']
            )
            
            response_time = (time.time() - start_time) * 1000
            
            status = ServiceHealth.HEALTHY if result else ServiceHealth.UNHEALTHY
            message = "Health check passed" if result else "Health check failed"
            
            health_result = HealthCheckResult(
                name=name,
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time
            )
            
            self.last_results[name] = health_result
            check['last_run'] = datetime.utcnow()
            
            return health_result
            
        except asyncio.TimeoutError:
            return HealthCheckResult(
                name=name,
                status=ServiceHealth.UNHEALTHY,
                message=f"Health check timed out after {check['timeout']}s",
                timestamp=datetime.utcnow(),
                response_time_ms=check['timeout'] * 1000
            )
        except Exception as e:
            return HealthCheckResult(
                name=name,
                status=ServiceHealth.UNHEALTHY,
                message=f"Health check error: {str(e)}",
                timestamp=datetime.utcnow(),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def _run_check_func(self, func: Callable) -> bool:
        """Run health check function"""
        if asyncio.iscoroutinefunction(func):
            return await func()
        else:
            return func()
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks"""
        results = {}
        tasks = []
        
        for name in self.health_checks:
            tasks.append(self.run_check(name))
        
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(check_results):
            name = list(self.health_checks.keys())[i]
            if isinstance(result, Exception):
                results[name] = HealthCheckResult(
                    name=name,
                    status=ServiceHealth.UNHEALTHY,
                    message=f"Exception: {str(result)}",
                    timestamp=datetime.utcnow(),
                    response_time_ms=0
                )
            else:
                results[name] = result
        
        return results
    
    def get_overall_health(self) -> ServiceHealth:
        """Get overall service health"""
        if not self.last_results:
            return ServiceHealth.UNKNOWN
        
        statuses = [result.status for result in self.last_results.values()]
        
        if all(status == ServiceHealth.HEALTHY for status in statuses):
            return ServiceHealth.HEALTHY
        elif any(status == ServiceHealth.UNHEALTHY for status in statuses):
            return ServiceHealth.UNHEALTHY
        else:
            return ServiceHealth.DEGRADED


class AlertManager:
    """Alert management system"""
    
    def __init__(self, service_name: str, redis_client: redis.Redis):
        self.service_name = service_name
        self.redis = redis_client
        self.alert_rules = []
        self.active_alerts = {}
    
    def add_alert_rule(self, name: str, condition: Callable[[Dict[str, Any]], bool], 
                      severity: AlertSeverity, title: str, description: str):
        """Add alert rule"""
        self.alert_rules.append({
            'name': name,
            'condition': condition,
            'severity': severity,
            'title': title,
            'description': description
        })
    
    async def evaluate_alerts(self, metrics: Dict[str, Any], health_results: Dict[str, HealthCheckResult]):
        """Evaluate alert rules against current metrics and health"""
        context = {
            'metrics': metrics,
            'health': health_results,
            'service': self.service_name,
            'timestamp': datetime.utcnow()
        }
        
        for rule in self.alert_rules:
            try:
                should_alert = rule['condition'](context)
                alert_id = f"{self.service_name}_{rule['name']}"
                
                if should_alert and alert_id not in self.active_alerts:
                    # Create new alert
                    alert = Alert(
                        id=alert_id,
                        service=self.service_name,
                        severity=rule['severity'],
                        title=rule['title'],
                        description=rule['description'],
                        timestamp=datetime.utcnow(),
                        metadata=context
                    )
                    
                    self.active_alerts[alert_id] = alert
                    await self._send_alert(alert)
                    
                elif not should_alert and alert_id in self.active_alerts:
                    # Resolve alert
                    alert = self.active_alerts[alert_id]
                    alert.resolved = True
                    alert.resolved_at = datetime.utcnow()
                    
                    await self._resolve_alert(alert)
                    del self.active_alerts[alert_id]
                    
            except Exception as e:
                logger.error("Error evaluating alert rule", rule=rule['name'], error=str(e))
    
    async def _send_alert(self, alert: Alert):
        """Send alert notification"""
        # Store alert in Redis
        alert_key = f"alerts:{alert.service}:{alert.id}"
        alert_data = asdict(alert)
        alert_data['timestamp'] = alert.timestamp.isoformat()
        
        await self.redis.hset(alert_key, mapping={
            'data': json.dumps(alert_data, default=str)
        })
        await self.redis.expire(alert_key, 86400 * 30)  # 30 days retention
        
        # Publish to alert channel
        await self.redis.publish('alerts', json.dumps(alert_data, default=str))
        
        logger.warning("Alert triggered", 
                      alert_id=alert.id,
                      severity=alert.severity.value,
                      title=alert.title)
    
    async def _resolve_alert(self, alert: Alert):
        """Resolve alert"""
        alert_key = f"alerts:{alert.service}:{alert.id}"
        alert_data = asdict(alert)
        alert_data['timestamp'] = alert.timestamp.isoformat()
        alert_data['resolved_at'] = alert.resolved_at.isoformat()
        
        await self.redis.hset(alert_key, mapping={
            'data': json.dumps(alert_data, default=str)
        })
        
        logger.info("Alert resolved", alert_id=alert.id)


class MonitoringSystem:
    """Comprehensive monitoring system"""
    
    def __init__(self, service_name: str, redis_client: redis.Redis, 
                 jaeger_endpoint: str = "http://localhost:14268/api/traces"):
        self.service_name = service_name
        self.redis = redis_client
        
        # Initialize components
        self.metrics = PrometheusMetrics(service_name)
        self.tracing = DistributedTracing(service_name, jaeger_endpoint)
        self.health_checker = HealthChecker(service_name)
        self.alert_manager = AlertManager(service_name, redis_client)
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        # Monitoring loop
        self.monitoring_task = None
    
    def _setup_default_health_checks(self):
        """Setup default health checks"""
        
        async def redis_health():
            """Check Redis connectivity"""
            try:
                await self.redis.ping()
                return True
            except:
                return False
        
        def memory_health():
            """Check memory usage"""
            import psutil
            memory_percent = psutil.virtual_memory().percent
            return memory_percent < 90
        
        def disk_health():
            """Check disk usage"""
            import psutil
            disk_percent = psutil.disk_usage('/').percent
            return disk_percent < 90
        
        self.health_checker.register_check("redis", redis_health, interval=30)
        self.health_checker.register_check("memory", memory_health, interval=60)
        self.health_checker.register_check("disk", disk_health, interval=300)
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        
        def high_error_rate(context):
            """Alert on high error rate"""
            # This would check actual error rate from metrics
            return False  # Placeholder
        
        def service_unhealthy(context):
            """Alert when service is unhealthy"""
            health_results = context['health']
            unhealthy_checks = [
                name for name, result in health_results.items() 
                if result.status == ServiceHealth.UNHEALTHY
            ]
            return len(unhealthy_checks) > 0
        
        def high_response_time(context):
            """Alert on high response times"""
            # This would check actual response time metrics
            return False  # Placeholder
        
        self.alert_manager.add_alert_rule(
            "high_error_rate",
            high_error_rate,
            AlertSeverity.WARNING,
            "High Error Rate",
            "Service is experiencing elevated error rates"
        )
        
        self.alert_manager.add_alert_rule(
            "service_unhealthy",
            service_unhealthy,
            AlertSeverity.CRITICAL,
            "Service Unhealthy",
            "One or more health checks are failing"
        )
        
        self.alert_manager.add_alert_rule(
            "high_response_time",
            high_response_time,
            AlertSeverity.WARNING,
            "High Response Time",
            "Service response times are elevated"
        )
    
    async def start_monitoring(self, interval: int = 60):
        """Start monitoring loop"""
        async def monitoring_loop():
            while True:
                try:
                    # Run health checks
                    health_results = await self.health_checker.run_all_checks()
                    
                    # Collect system metrics
                    system_metrics = await self._collect_system_metrics()
                    
                    # Evaluate alerts
                    await self.alert_manager.evaluate_alerts(system_metrics, health_results)
                    
                    # Update system metrics in Prometheus
                    self._update_system_metrics(system_metrics)
                    
                    await asyncio.sleep(interval)
                    
                except Exception as e:
                    logger.error("Error in monitoring loop", error=str(e))
                    await asyncio.sleep(interval)
        
        self.monitoring_task = asyncio.create_task(monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop monitoring loop"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics"""
        try:
            import psutil
            
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available': psutil.virtual_memory().available,
                'disk_percent': psutil.disk_usage('/').percent,
                'network_io': psutil.net_io_counters()._asdict(),
                'process_count': len(psutil.pids())
            }
        except ImportError:
            return {}
    
    def _update_system_metrics(self, metrics: Dict[str, Any]):
        """Update Prometheus with system metrics"""
        if 'cpu_percent' in metrics:
            self.metrics.cpu_usage.labels(service=self.service_name).set(metrics['cpu_percent'])
        
        if 'memory_available' in metrics:
            self.metrics.memory_usage.labels(service=self.service_name).set(metrics['memory_available'])
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        health_results = await self.health_checker.run_all_checks()
        overall_health = self.health_checker.get_overall_health()
        
        return {
            'service': self.service_name,
            'overall_health': overall_health.value,
            'health_checks': {
                name: {
                    'status': result.status.value,
                    'message': result.message,
                    'response_time_ms': result.response_time_ms,
                    'timestamp': result.timestamp.isoformat()
                }
                for name, result in health_results.items()
            },
            'active_alerts': len(self.alert_manager.active_alerts),
            'timestamp': datetime.utcnow().isoformat()
        }