"""
Monitoring and Alerting System for Financial News Processing Pipeline.

This module provides:
1. Real-time metrics collection and analysis
2. Anomaly detection for processing patterns
3. Performance monitoring and SLA tracking
4. Alert generation and notification system
5. Health checks and system status monitoring
6. Dashboard data aggregation
"""

import asyncio
import json
import time
import smtplib
import logging
from typing import Dict, List, Optional, Any, Callable, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
except ImportError:
    # Fallback for test environments
    MimeText = None
    MimeMultipart = None
from enum import Enum
import statistics

try:
    import structlog
except ImportError:
    import logging as structlog

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

try:
    import aiohttp
except ImportError:
    aiohttp = None
try:
    from jinja2 import Template
except ImportError:
    Template = None

try:
    logger = structlog.get_logger(__name__)
except AttributeError:
    import logging
    logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertType(Enum):
    """Types of alerts"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    HIGH_ERROR_RATE = "high_error_rate"
    QUEUE_BACKLOG = "queue_backlog"
    DUPLICATE_SPIKE = "duplicate_spike"
    LOW_QUALITY_ARTICLES = "low_quality_articles"
    PROCESSING_STOPPED = "processing_stopped"
    DEPENDENCY_FAILURE = "dependency_failure"
    ANOMALY_DETECTED = "anomaly_detected"
    SLA_BREACH = "sla_breach"
    SYSTEM_HEALTH = "system_health"


@dataclass
class Alert:
    """Alert notification"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    metrics: Dict[str, Any]
    suggested_actions: List[str]
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


@dataclass
class PerformanceThresholds:
    """Performance thresholds for alerting"""
    max_avg_processing_time_ms: int = 10000  # 10 seconds
    max_error_rate: float = 0.05  # 5%
    max_queue_length: int = 1000
    min_throughput_per_hour: int = 100
    max_duplicate_rate: float = 0.3  # 30%
    min_quality_score: float = 0.6
    max_p95_processing_time_ms: int = 30000  # 30 seconds


@dataclass
class SystemHealth:
    """System health status"""
    overall_status: str  # "healthy", "degraded", "critical"
    components: Dict[str, str]  # component -> status
    metrics_summary: Dict[str, Any]
    active_alerts: List[Alert]
    last_updated: datetime
    uptime_seconds: float


class MetricsCollector:
    """Collects and aggregates metrics from the processing pipeline"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.metrics_history = defaultdict(deque)  # Metric -> values (last 1000)
        self.time_series_data = defaultdict(list)  # For trend analysis
        
    async def collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current metrics from Redis"""
        try:
            # Get latest pipeline metrics
            metrics_keys = await self.redis_client.keys("metrics:*")
            if not metrics_keys:
                return {}
            
            # Get the most recent metrics
            latest_key = max(metrics_keys)
            metrics_data = await self.redis_client.get(latest_key)
            
            if metrics_data:
                return json.loads(metrics_data)
            
            return {}
            
        except Exception as e:
            logger.error("Failed to collect metrics", error=str(e))
            return {}
    
    async def collect_historical_metrics(self, hours: int = 24) -> Dict[str, List]:
        """Collect historical metrics for trend analysis"""
        try:
            cutoff_time = int(time.time()) - (hours * 3600)
            pattern = "metrics:*"
            
            historical_data = defaultdict(list)
            
            # Get all metrics keys
            metrics_keys = await self.redis_client.keys(pattern)
            
            for key in metrics_keys:
                # Extract timestamp from key
                timestamp = int(key.decode('utf-8').split(':')[1])
                
                if timestamp >= cutoff_time:
                    metrics_data = await self.redis_client.get(key)
                    if metrics_data:
                        data = json.loads(metrics_data)
                        for metric_name, value in data.items():
                            if isinstance(value, (int, float)):
                                historical_data[metric_name].append({
                                    'timestamp': timestamp,
                                    'value': value
                                })
            
            return dict(historical_data)
            
        except Exception as e:
            logger.error("Failed to collect historical metrics", error=str(e))
            return {}
    
    def update_metrics_history(self, metrics: Dict[str, Any]):
        """Update metrics history for trend analysis"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_history[key].append(value)
                if len(self.metrics_history[key]) > 1000:
                    self.metrics_history[key].popleft()
    
    def calculate_trends(self, metric_name: str, periods: int = 10) -> Dict[str, float]:
        """Calculate trends for a specific metric"""
        if metric_name not in self.metrics_history or len(self.metrics_history[metric_name]) < periods:
            return {}
        
        values = list(self.metrics_history[metric_name])[-periods:]
        
        if len(values) < 2:
            return {}
        
        # Calculate trend statistics
        mean_value = statistics.mean(values)
        median_value = statistics.median(values)
        
        # Simple linear trend (slope)
        n = len(values)
        x = list(range(n))
        x_mean = statistics.mean(x)
        xy_mean = statistics.mean([x[i] * values[i] for i in range(n)])
        x_sq_mean = statistics.mean([x[i] ** 2 for i in range(n)])
        
        if x_sq_mean - x_mean ** 2 != 0:
            slope = (xy_mean - x_mean * mean_value) / (x_sq_mean - x_mean ** 2)
        else:
            slope = 0.0
        
        return {
            'mean': mean_value,
            'median': median_value,
            'slope': slope,
            'trend': 'increasing' if slope > 0.1 else 'decreasing' if slope < -0.1 else 'stable'
        }


class AnomalyDetector:
    """Detects anomalies in processing metrics"""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Number of standard deviations for anomaly
        
    def detect_anomalies(self, metrics_collector: MetricsCollector) -> List[Dict[str, Any]]:
        """Detect anomalies in current metrics"""
        anomalies = []
        
        # Check key metrics for anomalies
        key_metrics = [
            'avg_processing_time_ms',
            'error_rate', 
            'throughput_jobs_per_hour',
            'duplicate_rate',
            'success_rate'
        ]
        
        for metric in key_metrics:
            if metric in metrics_collector.metrics_history:
                anomaly = self._check_metric_anomaly(metric, metrics_collector)
                if anomaly:
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _check_metric_anomaly(self, metric_name: str, collector: MetricsCollector) -> Optional[Dict[str, Any]]:
        """Check if a specific metric has an anomaly"""
        values = list(collector.metrics_history[metric_name])
        
        if len(values) < 10:  # Need enough data points
            return None
        
        # Use last 100 points for baseline
        baseline = values[-100:] if len(values) > 100 else values[:-1]
        current_value = values[-1]
        
        if len(baseline) < 5:
            return None
        
        mean = statistics.mean(baseline)
        stdev = statistics.stdev(baseline) if len(baseline) > 1 else 0
        
        if stdev == 0:
            return None
        
        z_score = abs(current_value - mean) / stdev
        
        if z_score > self.sensitivity:
            return {
                'metric': metric_name,
                'current_value': current_value,
                'baseline_mean': mean,
                'baseline_stdev': stdev,
                'z_score': z_score,
                'anomaly_type': 'spike' if current_value > mean else 'drop'
            }
        
        return None


class AlertManager:
    """Manages alert generation, tracking, and notifications"""
    
    def __init__(self, redis_client, thresholds: PerformanceThresholds):
        self.redis_client = redis_client
        self.thresholds = thresholds
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: List[Callable] = []
        
    async def check_thresholds(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Check metrics against thresholds and generate alerts"""
        new_alerts = []
        
        # Performance degradation
        if metrics.get('avg_processing_time_ms', 0) > self.thresholds.max_avg_processing_time_ms:
            alert = self._create_alert(
                AlertType.PERFORMANCE_DEGRADATION,
                AlertSeverity.HIGH,
                "High Average Processing Time",
                f"Average processing time is {metrics['avg_processing_time_ms']}ms "
                f"(threshold: {self.thresholds.max_avg_processing_time_ms}ms)",
                metrics,
                [
                    "Check system resources (CPU, memory)",
                    "Review recent code changes",
                    "Scale up worker count",
                    "Check database/Redis performance"
                ]
            )
            new_alerts.append(alert)
        
        # High error rate
        if metrics.get('error_rate', 0) > self.thresholds.max_error_rate:
            alert = self._create_alert(
                AlertType.HIGH_ERROR_RATE,
                AlertSeverity.CRITICAL,
                "High Error Rate Detected",
                f"Error rate is {metrics['error_rate']:.2%} "
                f"(threshold: {self.thresholds.max_error_rate:.2%})",
                metrics,
                [
                    "Check application logs for errors",
                    "Verify external service availability",
                    "Review recent deployments",
                    "Check data quality"
                ]
            )
            new_alerts.append(alert)
        
        # Queue backlog
        queue_lengths = metrics.get('queue_lengths', {})
        total_queue_length = sum(queue_lengths.values()) if queue_lengths else 0
        
        if total_queue_length > self.thresholds.max_queue_length:
            alert = self._create_alert(
                AlertType.QUEUE_BACKLOG,
                AlertSeverity.MEDIUM,
                "High Queue Backlog",
                f"Total queue length is {total_queue_length} "
                f"(threshold: {self.thresholds.max_queue_length})",
                metrics,
                [
                    "Scale up worker count",
                    "Check worker health",
                    "Review processing efficiency",
                    "Consider priority queue adjustments"
                ]
            )
            new_alerts.append(alert)
        
        # Low throughput
        if metrics.get('throughput_jobs_per_hour', 0) < self.thresholds.min_throughput_per_hour:
            alert = self._create_alert(
                AlertType.PERFORMANCE_DEGRADATION,
                AlertSeverity.MEDIUM,
                "Low Processing Throughput",
                f"Throughput is {metrics['throughput_jobs_per_hour']} jobs/hour "
                f"(threshold: {self.thresholds.min_throughput_per_hour})",
                metrics,
                [
                    "Check if workers are running",
                    "Review queue backlogs", 
                    "Check system resources",
                    "Verify Redis connectivity"
                ]
            )
            new_alerts.append(alert)
        
        # High duplicate rate
        if metrics.get('duplicate_rate', 0) > self.thresholds.max_duplicate_rate:
            alert = self._create_alert(
                AlertType.DUPLICATE_SPIKE,
                AlertSeverity.LOW,
                "High Duplicate Rate",
                f"Duplicate rate is {metrics['duplicate_rate']:.2%} "
                f"(threshold: {self.thresholds.max_duplicate_rate:.2%})",
                metrics,
                [
                    "Check news source quality",
                    "Review deduplication settings",
                    "Verify feed diversity",
                    "Check for feed issues"
                ]
            )
            new_alerts.append(alert)
        
        # Process new alerts
        for alert in new_alerts:
            await self._process_new_alert(alert)
        
        return new_alerts
    
    def _create_alert(self, alert_type: AlertType, severity: AlertSeverity,
                     title: str, description: str, metrics: Dict[str, Any],
                     suggested_actions: List[str]) -> Alert:
        """Create a new alert"""
        import uuid
        
        return Alert(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            severity=severity,
            title=title,
            description=description,
            timestamp=datetime.now(timezone.utc),
            metrics=metrics.copy(),
            suggested_actions=suggested_actions
        )
    
    async def _process_new_alert(self, alert: Alert):
        """Process a new alert (deduplication, storage, notification)"""
        # Check for similar active alerts (deduplication)
        similar_alert = self._find_similar_active_alert(alert)
        
        if similar_alert:
            # Update existing alert instead of creating new one
            similar_alert.timestamp = alert.timestamp
            similar_alert.metrics = alert.metrics
            logger.debug("Updated existing alert", alert_id=similar_alert.alert_id)
            return
        
        # Store new alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        # Store in Redis
        await self._store_alert(alert)
        
        # Send notifications
        await self._send_notifications(alert)
        
        logger.info("New alert created", 
                   alert_id=alert.alert_id, 
                   type=alert.alert_type.value,
                   severity=alert.severity.value)
    
    def _find_similar_active_alert(self, alert: Alert) -> Optional[Alert]:
        """Find similar active alert for deduplication"""
        for active_alert in self.active_alerts.values():
            if (active_alert.alert_type == alert.alert_type and 
                active_alert.severity == alert.severity and
                not active_alert.resolved):
                return active_alert
        return None
    
    async def _store_alert(self, alert: Alert):
        """Store alert in Redis"""
        try:
            alert_key = f"alerts:{alert.alert_id}"
            alert_data = json.dumps(asdict(alert), default=str)
            await self.redis_client.setex(alert_key, 86400 * 7, alert_data)  # 7 days TTL
        except Exception as e:
            logger.error("Failed to store alert", alert_id=alert.alert_id, error=str(e))
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications"""
        for handler in self.notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error("Notification handler failed", error=str(e))
    
    def add_notification_handler(self, handler: Callable):
        """Add a notification handler"""
        self.notification_handlers.append(handler)
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now(timezone.utc)
            
            await self._store_alert(alert)
            logger.info("Alert acknowledged", alert_id=alert_id, by=acknowledged_by)
    
    async def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now(timezone.utc)
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            await self._store_alert(alert)
            logger.info("Alert resolved", alert_id=alert_id)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())


class NotificationSystem:
    """Handles various notification channels"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.email_template = Template("""
        <html>
        <body>
        <h2 style="color: {{ color }};">{{ alert.severity.value.upper() }}: {{ alert.title }}</h2>
        <p><strong>Type:</strong> {{ alert.alert_type.value }}</p>
        <p><strong>Time:</strong> {{ alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC') }}</p>
        <p><strong>Description:</strong> {{ alert.description }}</p>
        
        <h3>Suggested Actions:</h3>
        <ul>
        {% for action in alert.suggested_actions %}
        <li>{{ action }}</li>
        {% endfor %}
        </ul>
        
        <h3>Metrics:</h3>
        <ul>
        {% for key, value in alert.metrics.items() %}
        <li><strong>{{ key }}:</strong> {{ value }}</li>
        {% endfor %}
        </ul>
        </body>
        </html>
        """)
    
    async def send_email_alert(self, alert: Alert):
        """Send email alert"""
        if not self.config.get('email', {}).get('enabled'):
            return
        
        try:
            smtp_config = self.config['email']
            
            # Create message
            msg = MimeMultipart('alternative')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            msg['From'] = smtp_config['from_email']
            msg['To'] = ', '.join(smtp_config['to_emails'])
            
            # Determine color based on severity
            color_map = {
                AlertSeverity.CRITICAL: '#FF0000',
                AlertSeverity.HIGH: '#FF6600', 
                AlertSeverity.MEDIUM: '#FFAA00',
                AlertSeverity.LOW: '#FFDD00',
                AlertSeverity.INFO: '#0088FF'
            }
            color = color_map.get(alert.severity, '#000000')
            
            # Render HTML content
            html_content = self.email_template.render(alert=alert, color=color)
            html_part = MimeText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port']) as server:
                if smtp_config.get('use_tls'):
                    server.starttls()
                if smtp_config.get('username'):
                    server.login(smtp_config['username'], smtp_config['password'])
                server.send_message(msg)
            
            logger.info("Email alert sent", alert_id=alert.alert_id)
            
        except Exception as e:
            logger.error("Failed to send email alert", alert_id=alert.alert_id, error=str(e))
    
    async def send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        webhook_config = self.config.get('webhook', {})
        if not webhook_config.get('enabled'):
            return
        
        try:
            payload = {
                'alert_id': alert.alert_id,
                'type': alert.alert_type.value,
                'severity': alert.severity.value,
                'title': alert.title,
                'description': alert.description,
                'timestamp': alert.timestamp.isoformat(),
                'metrics': alert.metrics,
                'suggested_actions': alert.suggested_actions
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_config['url'],
                    json=payload,
                    headers=webhook_config.get('headers', {}),
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info("Webhook alert sent", alert_id=alert.alert_id)
                    else:
                        logger.warning("Webhook alert failed", 
                                     alert_id=alert.alert_id, 
                                     status=response.status)
        
        except Exception as e:
            logger.error("Failed to send webhook alert", alert_id=alert.alert_id, error=str(e))


class HealthChecker:
    """Performs system health checks"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
    
    async def check_system_health(self) -> SystemHealth:
        """Perform comprehensive system health check"""
        components = {}
        
        # Check Redis
        components['redis'] = await self._check_redis()
        
        # Check processing pipeline
        components['pipeline'] = await self._check_pipeline()
        
        # Check queues
        components['queues'] = await self._check_queues()
        
        # Determine overall status
        overall_status = self._determine_overall_status(components)
        
        # Get metrics summary
        collector = MetricsCollector(self.redis_client)
        metrics_summary = await collector.collect_current_metrics()
        
        return SystemHealth(
            overall_status=overall_status,
            components=components,
            metrics_summary=metrics_summary,
            active_alerts=[],  # Will be populated by AlertManager
            last_updated=datetime.now(timezone.utc),
            uptime_seconds=time.time() - metrics_summary.get('start_time', time.time())
        )
    
    async def _check_redis(self) -> str:
        """Check Redis health"""
        try:
            await self.redis_client.ping()
            return "healthy"
        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
            return "critical"
    
    async def _check_pipeline(self) -> str:
        """Check processing pipeline health"""
        try:
            # Check if metrics are being updated (last update < 5 minutes)
            metrics_keys = await self.redis_client.keys("metrics:*")
            if not metrics_keys:
                return "critical"
            
            latest_key = max(metrics_keys)
            timestamp = int(latest_key.decode('utf-8').split(':')[1])
            
            if time.time() - timestamp > 300:  # 5 minutes
                return "degraded"
            
            return "healthy"
            
        except Exception as e:
            logger.error("Pipeline health check failed", error=str(e))
            return "critical"
    
    async def _check_queues(self) -> str:
        """Check queue health"""
        try:
            queue_lengths = {}
            queues = [
                "jobs:breaking_news", "jobs:high_priority", 
                "jobs:normal_priority", "jobs:low_priority", "jobs:historical"
            ]
            
            total_length = 0
            for queue in queues:
                length = await self.redis_client.llen(queue)
                queue_lengths[queue] = length
                total_length += length
            
            if total_length > 10000:  # Very high backlog
                return "critical"
            elif total_length > 1000:  # High backlog
                return "degraded"
            else:
                return "healthy"
                
        except Exception as e:
            logger.error("Queue health check failed", error=str(e))
            return "critical"
    
    def _determine_overall_status(self, components: Dict[str, str]) -> str:
        """Determine overall system status from components"""
        if any(status == "critical" for status in components.values()):
            return "critical"
        elif any(status == "degraded" for status in components.values()):
            return "degraded"
        else:
            return "healthy"


class MonitoringSystem:
    """Main monitoring and alerting system"""
    
    def __init__(self, redis_client, config: Dict[str, Any] = None):
        self.redis_client = redis_client
        self.config = config or {}
        
        # Initialize components
        self.thresholds = PerformanceThresholds()
        self.metrics_collector = MetricsCollector(redis_client)
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager(redis_client, self.thresholds)
        self.notification_system = NotificationSystem(self.config)
        self.health_checker = HealthChecker(redis_client)
        
        # Setup notification handlers
        self._setup_notification_handlers()
        
        # Monitoring control
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
    
    def _setup_notification_handlers(self):
        """Setup notification handlers"""
        self.alert_manager.add_notification_handler(
            self.notification_system.send_email_alert
        )
        self.alert_manager.add_notification_handler(
            self.notification_system.send_webhook_alert
        )
    
    async def start_monitoring(self, check_interval: int = 60):
        """Start the monitoring system"""
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(check_interval)
        )
        
        logger.info("Monitoring system started", check_interval=check_interval)
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Monitoring system stopped")
    
    async def _monitoring_loop(self, check_interval: int):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect current metrics
                metrics = await self.metrics_collector.collect_current_metrics()
                
                if metrics:
                    # Update metrics history
                    self.metrics_collector.update_metrics_history(metrics)
                    
                    # Check thresholds
                    await self.alert_manager.check_thresholds(metrics)
                    
                    # Detect anomalies
                    anomalies = self.anomaly_detector.detect_anomalies(self.metrics_collector)
                    for anomaly in anomalies:
                        await self._handle_anomaly(anomaly)
                
                # Health check
                await self._perform_health_check()
                
                # Log monitoring status
                logger.debug("Monitoring check completed", 
                           active_alerts=len(self.alert_manager.active_alerts))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Monitoring loop error", error=str(e))
            
            # Wait for next check
            await asyncio.sleep(check_interval)
    
    async def _handle_anomaly(self, anomaly: Dict[str, Any]):
        """Handle detected anomaly"""
        alert = self.alert_manager._create_alert(
            AlertType.ANOMALY_DETECTED,
            AlertSeverity.MEDIUM,
            f"Anomaly Detected in {anomaly['metric']}",
            f"Unusual pattern detected: {anomaly['anomaly_type']} in {anomaly['metric']}. "
            f"Current value: {anomaly['current_value']:.2f}, "
            f"Baseline mean: {anomaly['baseline_mean']:.2f} "
            f"(Z-score: {anomaly['z_score']:.2f})",
            anomaly,
            [
                "Investigate recent changes",
                "Check system resources",
                "Review application logs",
                "Monitor trend continuation"
            ]
        )
        
        await self.alert_manager._process_new_alert(alert)
    
    async def _perform_health_check(self):
        """Perform system health check"""
        try:
            health = await self.health_checker.check_system_health()
            
            # Update active alerts in health status
            health.active_alerts = self.alert_manager.get_active_alerts()
            
            # Store health status
            health_key = "system:health"
            health_data = json.dumps(asdict(health), default=str)
            await self.redis_client.setex(health_key, 300, health_data)  # 5 minutes TTL
            
            # Check for critical system status
            if health.overall_status == "critical":
                alert = self.alert_manager._create_alert(
                    AlertType.SYSTEM_HEALTH,
                    AlertSeverity.CRITICAL,
                    "System Health Critical",
                    f"System status is critical. Components: {health.components}",
                    {'health_status': health.overall_status, 'components': health.components},
                    [
                        "Check all system components",
                        "Review error logs",
                        "Verify external dependencies",
                        "Consider emergency procedures"
                    ]
                )
                await self.alert_manager._process_new_alert(alert)
        
        except Exception as e:
            logger.error("Health check failed", error=str(e))
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        try:
            # Current metrics
            current_metrics = await self.metrics_collector.collect_current_metrics()
            
            # Historical data (last 24 hours)
            historical_data = await self.metrics_collector.collect_historical_metrics(24)
            
            # System health
            health = await self.health_checker.check_system_health()
            health.active_alerts = self.alert_manager.get_active_alerts()
            
            # Alert summary
            alert_summary = {
                'active_count': len(self.alert_manager.active_alerts),
                'by_severity': defaultdict(int),
                'by_type': defaultdict(int),
                'recent_alerts': self.alert_manager.alert_history[-10:]  # Last 10 alerts
            }
            
            for alert in self.alert_manager.get_active_alerts():
                alert_summary['by_severity'][alert.severity.value] += 1
                alert_summary['by_type'][alert.alert_type.value] += 1
            
            return {
                'current_metrics': current_metrics,
                'historical_data': historical_data,
                'system_health': asdict(health),
                'alert_summary': dict(alert_summary),
                'thresholds': asdict(self.thresholds),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get dashboard data", error=str(e))
            return {}


# BEGIN F06 - Smart Alerts Implementation
import yaml
import pandas as pd
from pathlib import Path


@dataclass
class SmartAlert:
    """Smart alert for price moves, sentiment changes, and earnings"""
    alert_id: str
    symbol: str
    alert_type: str  # 'price_move', 'sentiment_swing', 'earnings_proximity'
    severity: str    # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    title: str
    description: str
    timestamp: datetime
    current_value: float
    previous_value: Optional[float]
    change_percent: Optional[float]
    guidance: str
    metadata: Dict[str, Any]


class SmartAlertsConfig:
    """Configuration loader for smart alerts"""
    
    def __init__(self, config_path: str = "config/alerts.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load alerts configuration from YAML file"""
        if not self.config_path.exists():
            logger.warning(f"Alerts config not found at {self.config_path}, using defaults")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded alerts config from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load alerts config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file not found"""
        return {
            'defaults': {
                'price_move_threshold_percent': 5.0,
                'price_move_timeframe_days': 1,
                'sentiment_threshold_change': 0.3,
                'sentiment_comparison_days': 7,
                'earnings_alert_days': [7, 3, 1],
                'batch_alerts': True,
                'immediate_email_severity': ['HIGH', 'CRITICAL'],
                'max_alerts_per_symbol_per_day': 3,
                'severity_thresholds': {
                    'price_move': {'LOW': 3.0, 'MEDIUM': 5.0, 'HIGH': 10.0, 'CRITICAL': 15.0},
                    'sentiment': {'LOW': 0.2, 'MEDIUM': 0.3, 'HIGH': 0.5, 'CRITICAL': 0.7},
                    'earnings': {'LOW': 7, 'MEDIUM': 3, 'HIGH': 1, 'CRITICAL': 0}
                }
            },
            'symbol_overrides': {},
            'templates': {
                'price_move': {
                    'title': 'Price Alert: {symbol} {direction} {percent}%',
                    'description': '{symbol} moved {direction} {percent}% to ${current_price} (was ${previous_price} {timeframe} ago)',
                    'guidance': 'Monitor for continued momentum. Check news for catalysts. Consider position sizing.'
                }
            }
        }
    
    def get_symbol_config(self, symbol: str) -> Dict[str, Any]:
        """Get configuration for a specific symbol (with overrides)"""
        defaults = self.config.get('defaults', {})
        overrides = self.config.get('symbol_overrides', {}).get(symbol, {})
        
        # Merge defaults with symbol-specific overrides
        symbol_config = defaults.copy()
        symbol_config.update(overrides)
        
        return symbol_config


class SmartAlertsEngine:
    """Engine for evaluating smart alerts on price moves, sentiment, and earnings"""
    
    def __init__(self, config: SmartAlertsConfig):
        self.config = config
        self.alert_history = []
        self.alert_cooldowns = {}  # symbol -> last_alert_time for cooldown tracking
    
    def evaluate_alerts(self, symbols: List[str], price_data: Dict[str, pd.DataFrame], 
                       sentiment_data: Dict[str, Dict], earnings_data: Dict[str, Any] = None) -> List[SmartAlert]:
        """
        Evaluate all smart alerts for given symbols and data.
        
        Args:
            symbols: List of stock symbols to evaluate
            price_data: Dict mapping symbol -> price DataFrame with Date, Close columns
            sentiment_data: Dict mapping symbol -> sentiment info with daily_sentiment
            earnings_data: Optional earnings calendar data
            
        Returns:
            List of triggered SmartAlert objects
        """
        triggered_alerts = []
        
        logger.info(f"Evaluating smart alerts for {len(symbols)} symbols")
        
        for symbol in symbols:
            try:
                # Check rate limiting / cooldown
                if self._is_symbol_on_cooldown(symbol):
                    continue
                
                symbol_config = self.config.get_symbol_config(symbol)
                
                # Evaluate price movement alerts
                price_alerts = self._evaluate_price_alerts(symbol, price_data.get(symbol), symbol_config)
                triggered_alerts.extend(price_alerts)
                
                # Evaluate sentiment swing alerts
                sentiment_alerts = self._evaluate_sentiment_alerts(symbol, sentiment_data.get(symbol), symbol_config)
                triggered_alerts.extend(sentiment_alerts)
                
                # Evaluate earnings proximity alerts
                if earnings_data:
                    earnings_alerts = self._evaluate_earnings_alerts(symbol, earnings_data.get(symbol), symbol_config)
                    triggered_alerts.extend(earnings_alerts)
                
                # Update cooldown if alerts were triggered
                if any(alert.symbol == symbol for alert in triggered_alerts):
                    self._update_cooldown(symbol)
                
            except Exception as e:
                logger.error(f"Failed to evaluate alerts for {symbol}: {e}")
                continue
        
        # Store alerts in history
        self.alert_history.extend(triggered_alerts)
        
        # Limit history size
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        logger.info(f"Triggered {len(triggered_alerts)} smart alerts")
        return triggered_alerts
    
    def _evaluate_price_alerts(self, symbol: str, price_df: pd.DataFrame, config: Dict[str, Any]) -> List[SmartAlert]:
        """Evaluate price movement alerts for a symbol"""
        if price_df is None or price_df.empty:
            return []
        
        alerts = []
        threshold = config.get('price_move_threshold_percent', 5.0)
        timeframe_days = config.get('price_move_timeframe_days', 1)
        
        # Ensure we have enough data
        if len(price_df) < timeframe_days + 1:
            return []
        
        # Get current and previous prices
        current_price = price_df.iloc[-1]['Close']
        previous_price = price_df.iloc[-(timeframe_days + 1)]['Close']
        
        # Calculate percentage change
        percent_change = ((current_price - previous_price) / previous_price) * 100
        
        # Check if change exceeds threshold
        if abs(percent_change) >= threshold:
            severity = self._determine_price_severity(abs(percent_change), config)
            direction = "up" if percent_change > 0 else "down"
            
            alert = self._create_price_alert(
                symbol, current_price, previous_price, percent_change, 
                direction, timeframe_days, severity, config
            )
            alerts.append(alert)
        
        return alerts
    
    def _evaluate_sentiment_alerts(self, symbol: str, sentiment_info: Dict, config: Dict[str, Any]) -> List[SmartAlert]:
        """Evaluate sentiment swing alerts for a symbol"""
        if not sentiment_info or 'daily_sentiment' not in sentiment_info:
            return []
        
        alerts = []
        threshold = config.get('sentiment_threshold_change', 0.3)
        comparison_days = config.get('sentiment_comparison_days', 7)
        
        daily_sentiment = sentiment_info['daily_sentiment']
        if len(daily_sentiment) < comparison_days:
            return []
        
        # Calculate current vs historical average sentiment
        sentiment_values = list(daily_sentiment.values())
        current_sentiment = sentiment_values[-1] if sentiment_values else 0
        
        # Get historical average (excluding current day)
        if len(sentiment_values) > comparison_days:
            historical_values = sentiment_values[-(comparison_days + 1):-1]
        else:
            historical_values = sentiment_values[:-1]
        
        if not historical_values:
            return []
        
        avg_historical_sentiment = sum(historical_values) / len(historical_values)
        sentiment_change = current_sentiment - avg_historical_sentiment
        
        # Check if change exceeds threshold
        if abs(sentiment_change) >= threshold:
            severity = self._determine_sentiment_severity(abs(sentiment_change), config)
            direction = "improved" if sentiment_change > 0 else "deteriorated"
            
            alert = self._create_sentiment_alert(
                symbol, current_sentiment, avg_historical_sentiment, 
                sentiment_change, direction, comparison_days, severity, config
            )
            alerts.append(alert)
        
        return alerts
    
    def _evaluate_earnings_alerts(self, symbol: str, earnings_info: Dict, config: Dict[str, Any]) -> List[SmartAlert]:
        """Evaluate earnings proximity alerts for a symbol"""
        if not earnings_info or 'date' not in earnings_info:
            return []
        
        alerts = []
        alert_days = config.get('earnings_alert_days', [7, 3, 1])
        
        try:
            earnings_date = pd.to_datetime(earnings_info['date']).date()
            today = datetime.now().date()
            days_until_earnings = (earnings_date - today).days
            
            # Check if we should alert for this time frame
            if days_until_earnings in alert_days and days_until_earnings >= 0:
                severity = self._determine_earnings_severity(days_until_earnings, config)
                
                alert = self._create_earnings_alert(
                    symbol, earnings_date, days_until_earnings, 
                    earnings_info, severity, config
                )
                alerts.append(alert)
        
        except Exception as e:
            logger.error(f"Failed to parse earnings date for {symbol}: {e}")
        
        return alerts
    
    def _determine_price_severity(self, percent_change: float, config: Dict[str, Any]) -> str:
        """Determine severity level for price movement"""
        thresholds = config.get('severity_thresholds', {}).get('price_move', {})
        
        if percent_change >= thresholds.get('CRITICAL', 15.0):
            return 'CRITICAL'
        elif percent_change >= thresholds.get('HIGH', 10.0):
            return 'HIGH'
        elif percent_change >= thresholds.get('MEDIUM', 5.0):
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _determine_sentiment_severity(self, sentiment_change: float, config: Dict[str, Any]) -> str:
        """Determine severity level for sentiment change"""
        thresholds = config.get('severity_thresholds', {}).get('sentiment', {})
        
        if sentiment_change >= thresholds.get('CRITICAL', 0.7):
            return 'CRITICAL'
        elif sentiment_change >= thresholds.get('HIGH', 0.5):
            return 'HIGH'
        elif sentiment_change >= thresholds.get('MEDIUM', 0.3):
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _determine_earnings_severity(self, days_until: int, config: Dict[str, Any]) -> str:
        """Determine severity level for earnings proximity"""
        thresholds = config.get('severity_thresholds', {}).get('earnings', {})
        
        if days_until <= thresholds.get('CRITICAL', 0):
            return 'CRITICAL'
        elif days_until <= thresholds.get('HIGH', 1):
            return 'HIGH'
        elif days_until <= thresholds.get('MEDIUM', 3):
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _create_price_alert(self, symbol: str, current_price: float, previous_price: float,
                           percent_change: float, direction: str, timeframe_days: int,
                           severity: str, config: Dict[str, Any]) -> SmartAlert:
        """Create price movement alert"""
        import uuid
        
        template = config.get('templates', {}).get('price_move', {})
        title = template.get('title', 'Price Alert: {symbol} {direction} {percent}%').format(
            symbol=symbol, direction=direction, percent=abs(percent_change)
        )
        
        description = template.get('description', 
            '{symbol} moved {direction} {percent}% to ${current_price} (was ${previous_price} {timeframe} ago)'
        ).format(
            symbol=symbol, direction=direction, percent=abs(percent_change),
            current_price=current_price, previous_price=previous_price,
            timeframe=f"{timeframe_days} day{'s' if timeframe_days > 1 else ''}"
        )
        
        guidance = template.get('guidance', 'Monitor for continued momentum. Check news for catalysts.')
        
        return SmartAlert(
            alert_id=str(uuid.uuid4()),
            symbol=symbol,
            alert_type='price_move',
            severity=severity,
            title=title,
            description=description,
            timestamp=datetime.now(timezone.utc),
            current_value=current_price,
            previous_value=previous_price,
            change_percent=percent_change,
            guidance=guidance,
            metadata={
                'timeframe_days': timeframe_days,
                'direction': direction,
                'threshold_exceeded': abs(percent_change)
            }
        )
    
    def _create_sentiment_alert(self, symbol: str, current_sentiment: float, previous_sentiment: float,
                               sentiment_change: float, direction: str, comparison_days: int,
                               severity: str, config: Dict[str, Any]) -> SmartAlert:
        """Create sentiment swing alert"""
        import uuid
        
        template = config.get('templates', {}).get('sentiment', {})
        title = template.get('title', 'Sentiment Alert: {symbol} sentiment {direction}').format(
            symbol=symbol, direction=direction
        )
        
        description = template.get('description',
            '{symbol} sentiment changed from {previous_sentiment:.2f} to {current_sentiment:.2f} ({change:+.2f})'
        ).format(
            symbol=symbol, current_sentiment=current_sentiment,
            previous_sentiment=previous_sentiment, change=sentiment_change
        )
        
        guidance = template.get('guidance', 'Review recent news coverage. Validate sentiment with multiple sources.')
        
        return SmartAlert(
            alert_id=str(uuid.uuid4()),
            symbol=symbol,
            alert_type='sentiment_swing',
            severity=severity,
            title=title,
            description=description,
            timestamp=datetime.now(timezone.utc),
            current_value=current_sentiment,
            previous_value=previous_sentiment,
            change_percent=None,
            guidance=guidance,
            metadata={
                'sentiment_change': sentiment_change,
                'comparison_days': comparison_days,
                'direction': direction
            }
        )
    
    def _create_earnings_alert(self, symbol: str, earnings_date: datetime.date, days_until: int,
                              earnings_info: Dict, severity: str, config: Dict[str, Any]) -> SmartAlert:
        """Create earnings proximity alert"""
        import uuid
        
        template = config.get('templates', {}).get('earnings', {})
        title = template.get('title', 'Earnings Alert: {symbol} reports in {days} days').format(
            symbol=symbol, days=days_until
        )
        
        description = template.get('description',
            '{symbol} earnings expected on {date}. Days until earnings: {days}'
        ).format(
            symbol=symbol, date=earnings_date.strftime('%Y-%m-%d'), days=days_until
        )
        
        guidance = template.get('guidance', 'Review analyst estimates. Consider volatility plays.')
        
        return SmartAlert(
            alert_id=str(uuid.uuid4()),
            symbol=symbol,
            alert_type='earnings_proximity',
            severity=severity,
            title=title,
            description=description,
            timestamp=datetime.now(timezone.utc),
            current_value=days_until,
            previous_value=None,
            change_percent=None,
            guidance=guidance,
            metadata={
                'earnings_date': earnings_date.isoformat(),
                'days_until': days_until,
                'earnings_info': earnings_info
            }
        )
    
    def _is_symbol_on_cooldown(self, symbol: str) -> bool:
        """Check if symbol is on cooldown to prevent spam"""
        if symbol not in self.alert_cooldowns:
            return False
        
        cooldown_minutes = self.config.config.get('security', {}).get('alert_cooldown_minutes', 60)
        last_alert_time = self.alert_cooldowns[symbol]
        time_since_alert = (datetime.now(timezone.utc) - last_alert_time).total_seconds() / 60
        
        return time_since_alert < cooldown_minutes
    
    def _update_cooldown(self, symbol: str):
        """Update cooldown timestamp for symbol"""
        self.alert_cooldowns[symbol] = datetime.now(timezone.utc)
    
    def get_daily_alerts_summary(self, date: datetime.date = None) -> Dict[str, Any]:
        """Get summary of alerts for a given date"""
        if date is None:
            date = datetime.now().date()
        
        daily_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp.date() == date
        ]
        
        summary = {
            'date': date.isoformat(),
            'total_alerts': len(daily_alerts),
            'by_severity': defaultdict(int),
            'by_type': defaultdict(int),
            'by_symbol': defaultdict(int),
            'alerts': daily_alerts
        }
        
        for alert in daily_alerts:
            summary['by_severity'][alert.severity] += 1
            summary['by_type'][alert.alert_type] += 1
            summary['by_symbol'][alert.symbol] += 1
        
        return dict(summary)


def create_smart_alerts_engine(config_path: str = "config/alerts.yaml") -> SmartAlertsEngine:
    """Factory function to create SmartAlertsEngine"""
    config = SmartAlertsConfig(config_path)
    return SmartAlertsEngine(config)

# END F06 - Smart Alerts Implementation


# Utility functions for easy integration
async def create_monitoring_system(redis_url: str = "redis://localhost:6379/0",
                                 config: Dict[str, Any] = None):
    """Create and initialize monitoring system"""
    if redis is None:
        return None
    redis_client = redis.from_url(redis_url)
    return MonitoringSystem(redis_client, config)


if __name__ == "__main__":
    # Test the monitoring system
    async def test_monitoring():
        # Create monitoring system
        monitoring = await create_monitoring_system()
        
        # Start monitoring
        await monitoring.start_monitoring(check_interval=30)
        
        # Run for a few minutes
        await asyncio.sleep(180)
        
        # Get dashboard data
        dashboard_data = await monitoring.get_dashboard_data()
        print("Dashboard Data:", json.dumps(dashboard_data, indent=2, default=str))
        
        # Stop monitoring
        await monitoring.stop_monitoring()
    
    # Run test
    asyncio.run(test_monitoring())