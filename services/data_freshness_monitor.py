"""
Data Freshness Monitoring and Alerting System

Monitors data staleness across all sources and components,
triggering alerts when data becomes outdated during critical periods.
"""

import asyncio
import logging
import json
import smtplib
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import sqlite3
from pathlib import Path
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class DataSource(Enum):
    ALPHA_VANTAGE = "alpha_vantage"
    YFINANCE = "yfinance" 
    NEWS_SCRAPER = "news_scraper"
    SENTIMENT_ANALYZER = "sentiment_analyzer"
    CACHE_SYSTEM = "cache_system"
    DATABASE = "database"

class AlertChannel(Enum):
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"
    SLACK = "slack"
    LOG = "log"

@dataclass
class FreshnessThreshold:
    """Data freshness threshold configuration"""
    source: DataSource
    data_type: str
    max_age_market_hours: timedelta
    max_age_after_hours: timedelta
    max_age_weekend: timedelta
    alert_severity: AlertSeverity
    grace_period: timedelta = timedelta(minutes=5)

@dataclass
class DataStalenessAlert:
    """Alert for stale data"""
    alert_id: str
    timestamp: datetime
    source: DataSource
    data_type: str
    symbol: Optional[str]
    last_update: datetime
    age: timedelta
    threshold_exceeded: timedelta
    severity: AlertSeverity
    message: str
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class MonitoringMetrics:
    """System monitoring metrics"""
    total_data_sources: int = 0
    healthy_sources: int = 0
    warning_sources: int = 0
    critical_sources: int = 0
    last_check_time: datetime = field(default_factory=datetime.now)
    overall_health_score: float = 100.0
    alerts_last_24h: int = 0
    avg_data_age_minutes: float = 0.0

class DataFreshnessMonitor:
    """Comprehensive data freshness monitoring system"""
    
    def __init__(self):
        self.db_path = "data/freshness_monitor.db"
        
        # Market hours (Eastern Time)
        self.market_open = time(9, 30)
        self.market_close = time(16, 0)
        
        # Monitoring configuration
        self.check_interval = timedelta(minutes=2)
        self.alert_cooldown = timedelta(minutes=15)  # Prevent spam
        
        # Portfolio and watchlist tickers for priority monitoring
        self.portfolio_tickers = ['RTX', 'PFE', 'MRVL', 'ADI', 'LLY', 'RIVN', 'TSLA', 'PLTR']
        self.watchlist_tickers = ['NVDA', 'GOOGL', 'AMD', 'MSFT']
        
        # Freshness thresholds
        self.thresholds: List[FreshnessThreshold] = []
        self._initialize_thresholds()
        
        # Alert channels
        self.alert_channels: Dict[AlertChannel, Dict[str, Any]] = {}
        self._initialize_alert_channels()
        
        # Active alerts tracking
        self.active_alerts: Dict[str, DataStalenessAlert] = {}
        self.alert_history: List[DataStalenessAlert] = []
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.metrics = MonitoringMetrics()
        
        self._initialize_database()

    def _initialize_database(self):
        """Initialize SQLite database for monitoring data"""
        
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Data freshness tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_freshness (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                data_type TEXT NOT NULL,
                symbol TEXT,
                last_update INTEGER NOT NULL,
                check_time INTEGER NOT NULL,
                age_minutes REAL NOT NULL,
                is_stale BOOLEAN NOT NULL,
                threshold_minutes REAL NOT NULL
            )
        """)
        
        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS freshness_alerts (
                alert_id TEXT PRIMARY KEY,
                timestamp INTEGER NOT NULL,
                source TEXT NOT NULL,
                data_type TEXT NOT NULL,
                symbol TEXT,
                last_update INTEGER NOT NULL,
                age_minutes REAL NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                resolved BOOLEAN DEFAULT 0,
                resolution_time INTEGER
            )
        """)
        
        # System metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                timestamp INTEGER PRIMARY KEY,
                total_sources INTEGER NOT NULL,
                healthy_sources INTEGER NOT NULL,
                warning_sources INTEGER NOT NULL,
                critical_sources INTEGER NOT NULL,
                overall_health_score REAL NOT NULL,
                alerts_24h INTEGER NOT NULL,
                avg_data_age_minutes REAL NOT NULL
            )
        """)
        
        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_freshness_source ON data_freshness(source, data_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_freshness_time ON data_freshness(check_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_time ON freshness_alerts(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON freshness_alerts(severity)")
        
        conn.commit()
        conn.close()

    def _initialize_thresholds(self):
        """Initialize data freshness thresholds"""
        
        # Critical portfolio data - strict thresholds
        for ticker in self.portfolio_tickers:
            self.thresholds.extend([
                FreshnessThreshold(
                    source=DataSource.ALPHA_VANTAGE,
                    data_type=f"daily_price_{ticker}",
                    max_age_market_hours=timedelta(minutes=30),
                    max_age_after_hours=timedelta(hours=4),
                    max_age_weekend=timedelta(hours=24),
                    alert_severity=AlertSeverity.CRITICAL
                ),
                FreshnessThreshold(
                    source=DataSource.NEWS_SCRAPER,
                    data_type=f"news_{ticker}",
                    max_age_market_hours=timedelta(minutes=10),
                    max_age_after_hours=timedelta(hours=1),
                    max_age_weekend=timedelta(hours=6),
                    alert_severity=AlertSeverity.WARNING
                )
            ])
        
        # Watchlist data - moderate thresholds
        for ticker in self.watchlist_tickers:
            self.thresholds.extend([
                FreshnessThreshold(
                    source=DataSource.ALPHA_VANTAGE,
                    data_type=f"daily_price_{ticker}",
                    max_age_market_hours=timedelta(hours=1),
                    max_age_after_hours=timedelta(hours=6),
                    max_age_weekend=timedelta(days=1),
                    alert_severity=AlertSeverity.WARNING
                ),
                FreshnessThreshold(
                    source=DataSource.NEWS_SCRAPER,
                    data_type=f"news_{ticker}",
                    max_age_market_hours=timedelta(minutes=30),
                    max_age_after_hours=timedelta(hours=2),
                    max_age_weekend=timedelta(hours=12),
                    alert_severity=AlertSeverity.INFO
                )
            ])
        
        # System-wide components
        self.thresholds.extend([
            FreshnessThreshold(
                source=DataSource.SENTIMENT_ANALYZER,
                data_type="sentiment_analysis",
                max_age_market_hours=timedelta(minutes=15),
                max_age_after_hours=timedelta(hours=2),
                max_age_weekend=timedelta(hours=8),
                alert_severity=AlertSeverity.WARNING
            ),
            FreshnessThreshold(
                source=DataSource.CACHE_SYSTEM,
                data_type="cache_health",
                max_age_market_hours=timedelta(minutes=5),
                max_age_after_hours=timedelta(minutes=10),
                max_age_weekend=timedelta(minutes=15),
                alert_severity=AlertSeverity.CRITICAL
            ),
            FreshnessThreshold(
                source=DataSource.DATABASE,
                data_type="database_health",
                max_age_market_hours=timedelta(minutes=5),
                max_age_after_hours=timedelta(minutes=10),
                max_age_weekend=timedelta(minutes=15),
                alert_severity=AlertSeverity.EMERGENCY
            )
        ])

    def _initialize_alert_channels(self):
        """Initialize alert notification channels"""
        
        # Email configuration (would be loaded from config)
        self.alert_channels[AlertChannel.EMAIL] = {
            'enabled': True,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'alerts@yourdomain.com',
            'password': 'your_app_password',  # Use app password
            'recipients': ['admin@yourdomain.com', 'trader@yourdomain.com'],
            'severity_filter': [AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        }
        
        # Webhook configuration
        self.alert_channels[AlertChannel.WEBHOOK] = {
            'enabled': True,
            'url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
            'headers': {'Content-Type': 'application/json'},
            'severity_filter': [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        }
        
        # Log channel (always enabled)
        self.alert_channels[AlertChannel.LOG] = {
            'enabled': True,
            'severity_filter': [AlertSeverity.INFO, AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
        }

    async def start_monitoring(self):
        """Start the data freshness monitoring system"""
        
        logger.info("Starting data freshness monitoring system")
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Data freshness monitoring started successfully")

    async def stop_monitoring(self):
        """Stop the monitoring system"""
        
        logger.info("Stopping data freshness monitoring")
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while True:
            try:
                await self._check_data_freshness()
                await self._update_metrics()
                await asyncio.sleep(self.check_interval.total_seconds())
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _check_data_freshness(self):
        """Check freshness of all monitored data sources"""
        
        check_time = datetime.now()
        stale_count = 0
        total_count = 0
        
        for threshold in self.thresholds:
            total_count += 1
            
            try:
                # Get last update time for this data source
                last_update = await self._get_last_update_time(threshold.source, threshold.data_type)
                
                if last_update is None:
                    # No data found - treat as stale
                    age = timedelta(hours=24)  # Assume very stale
                    is_stale = True
                else:
                    age = check_time - last_update
                    is_stale = self._is_data_stale(age, threshold)
                
                # Record freshness check
                await self._record_freshness_check(threshold, last_update, check_time, age, is_stale)
                
                if is_stale:
                    stale_count += 1
                    await self._handle_stale_data(threshold, last_update or check_time, age)
                else:
                    # Data is fresh - resolve any existing alerts
                    await self._resolve_alerts(threshold.source, threshold.data_type)
                
            except Exception as e:
                logger.error(f"Error checking freshness for {threshold.source.value}/{threshold.data_type}: {e}")
                stale_count += 1
        
        # Update overall metrics
        self.metrics.total_data_sources = total_count
        self.metrics.critical_sources = stale_count
        self.metrics.healthy_sources = total_count - stale_count
        self.metrics.last_check_time = check_time

    async def _get_last_update_time(self, source: DataSource, data_type: str) -> Optional[datetime]:
        """Get the last update time for a specific data source/type"""
        
        try:
            # This would integrate with your actual data sources
            # For now, return a placeholder based on current time with some variation
            
            if source == DataSource.ALPHA_VANTAGE:
                # Simulate Alpha Vantage data age
                minutes_ago = np.random.randint(5, 120)  # 5 minutes to 2 hours old
                return datetime.now() - timedelta(minutes=minutes_ago)
            
            elif source == DataSource.NEWS_SCRAPER:
                # Simulate news scraper data age
                minutes_ago = np.random.randint(2, 30)  # 2 to 30 minutes old
                return datetime.now() - timedelta(minutes=minutes_ago)
            
            elif source == DataSource.SENTIMENT_ANALYZER:
                # Simulate sentiment analysis data age
                minutes_ago = np.random.randint(10, 60)  # 10 to 60 minutes old
                return datetime.now() - timedelta(minutes=minutes_ago)
            
            elif source == DataSource.CACHE_SYSTEM:
                # Cache should be very fresh
                minutes_ago = np.random.randint(1, 10)  # 1 to 10 minutes old
                return datetime.now() - timedelta(minutes=minutes_ago)
            
            elif source == DataSource.DATABASE:
                # Database should be very fresh
                minutes_ago = np.random.randint(1, 5)  # 1 to 5 minutes old
                return datetime.now() - timedelta(minutes=minutes_ago)
            
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get last update time for {source.value}: {e}")
            return None

    def _is_data_stale(self, age: timedelta, threshold: FreshnessThreshold) -> bool:
        """Check if data is considered stale based on current market conditions"""
        
        now = datetime.now()
        current_time = now.time()
        weekday = now.weekday()
        
        # Determine appropriate threshold based on market hours
        if weekday >= 5:  # Weekend
            max_age = threshold.max_age_weekend
        elif self.market_open <= current_time <= self.market_close:
            max_age = threshold.max_age_market_hours
        else:
            max_age = threshold.max_age_after_hours
        
        return age > max_age + threshold.grace_period

    async def _record_freshness_check(self, threshold: FreshnessThreshold, 
                                    last_update: Optional[datetime],
                                    check_time: datetime, age: timedelta, is_stale: bool):
        """Record freshness check in database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Determine current threshold
            max_age = self._get_current_threshold(threshold)
            
            cursor.execute("""
                INSERT INTO data_freshness 
                (source, data_type, symbol, last_update, check_time, age_minutes, is_stale, threshold_minutes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                threshold.source.value,
                threshold.data_type,
                self._extract_symbol_from_data_type(threshold.data_type),
                int(last_update.timestamp()) if last_update else None,
                int(check_time.timestamp()),
                age.total_seconds() / 60,
                is_stale,
                max_age.total_seconds() / 60
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to record freshness check: {e}")

    def _get_current_threshold(self, threshold: FreshnessThreshold) -> timedelta:
        """Get current threshold based on market conditions"""
        
        now = datetime.now()
        current_time = now.time()
        weekday = now.weekday()
        
        if weekday >= 5:  # Weekend
            return threshold.max_age_weekend
        elif self.market_open <= current_time <= self.market_close:
            return threshold.max_age_market_hours
        else:
            return threshold.max_age_after_hours

    def _extract_symbol_from_data_type(self, data_type: str) -> Optional[str]:
        """Extract symbol from data type string"""
        
        parts = data_type.split('_')
        if len(parts) > 1:
            potential_symbol = parts[-1].upper()
            if potential_symbol in self.portfolio_tickers + self.watchlist_tickers:
                return potential_symbol
        return None

    async def _handle_stale_data(self, threshold: FreshnessThreshold, 
                               last_update: datetime, age: timedelta):
        """Handle stale data by generating alerts"""
        
        # Create alert ID
        alert_key = f"{threshold.source.value}_{threshold.data_type}"
        
        # Check if we already have an active alert (prevent spam)
        if alert_key in self.active_alerts:
            last_alert_time = self.active_alerts[alert_key].timestamp
            if datetime.now() - last_alert_time < self.alert_cooldown:
                return  # Skip alert due to cooldown
        
        # Create alert
        alert = DataStalenessAlert(
            alert_id=f"stale_{alert_key}_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            source=threshold.source,
            data_type=threshold.data_type,
            symbol=self._extract_symbol_from_data_type(threshold.data_type),
            last_update=last_update,
            age=age,
            threshold_exceeded=age - self._get_current_threshold(threshold),
            severity=threshold.alert_severity,
            message=self._generate_alert_message(threshold, age)
        )
        
        # Store alert
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        await self._send_alert_notifications(alert)
        
        # Store in database
        await self._store_alert(alert)
        
        logger.warning(f"Stale data alert: {alert.message}")

    def _generate_alert_message(self, threshold: FreshnessThreshold, age: timedelta) -> str:
        """Generate human-readable alert message"""
        
        symbol = self._extract_symbol_from_data_type(threshold.data_type)
        symbol_text = f" for {symbol}" if symbol else ""
        
        age_text = f"{int(age.total_seconds() / 60)} minutes"
        if age.total_seconds() > 3600:
            age_text = f"{age.total_seconds() / 3600:.1f} hours"
        
        return (f"Stale data detected: {threshold.source.value} {threshold.data_type}{symbol_text} "
               f"is {age_text} old, exceeding threshold during current market conditions")

    async def _send_alert_notifications(self, alert: DataStalenessAlert):
        """Send alert through configured notification channels"""
        
        for channel, config in self.alert_channels.items():
            if not config.get('enabled', False):
                continue
            
            if alert.severity not in config.get('severity_filter', []):
                continue
            
            try:
                if channel == AlertChannel.EMAIL:
                    await self._send_email_alert(alert, config)
                elif channel == AlertChannel.WEBHOOK:
                    await self._send_webhook_alert(alert, config)
                elif channel == AlertChannel.LOG:
                    await self._send_log_alert(alert)
                
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")

    async def _send_email_alert(self, alert: DataStalenessAlert, config: Dict[str, Any]):
        """Send email alert"""
        
        try:
            msg = MimeMultipart()
            msg['From'] = config['username']
            msg['To'] = ', '.join(config['recipients'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] Data Freshness Alert - {alert.source.value}"
            
            body = f"""
Data Freshness Alert

Severity: {alert.severity.value.upper()}
Source: {alert.source.value}
Data Type: {alert.data_type}
Symbol: {alert.symbol or 'N/A'}
Last Update: {alert.last_update.strftime('%Y-%m-%d %H:%M:%S')}
Age: {int(alert.age.total_seconds() / 60)} minutes
Threshold Exceeded By: {int(alert.threshold_exceeded.total_seconds() / 60)} minutes

Message: {alert.message}

Alert ID: {alert.alert_id}
Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Please investigate and resolve this issue promptly.
"""
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    async def _send_webhook_alert(self, alert: DataStalenessAlert, config: Dict[str, Any]):
        """Send webhook alert (Slack, Discord, etc.)"""
        
        try:
            payload = {
                "text": f"🚨 Data Freshness Alert - {alert.severity.value.upper()}",
                "attachments": [
                    {
                        "color": "danger" if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] else "warning",
                        "fields": [
                            {"title": "Source", "value": alert.source.value, "short": True},
                            {"title": "Data Type", "value": alert.data_type, "short": True},
                            {"title": "Symbol", "value": alert.symbol or "N/A", "short": True},
                            {"title": "Age", "value": f"{int(alert.age.total_seconds() / 60)} minutes", "short": True},
                            {"title": "Message", "value": alert.message, "short": False}
                        ],
                        "footer": f"Alert ID: {alert.alert_id}",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config['url'],
                    headers=config.get('headers', {}),
                    json=payload
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook alert sent for {alert.alert_id}")
                    else:
                        logger.error(f"Webhook alert failed with status {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    async def _send_log_alert(self, alert: DataStalenessAlert):
        """Send alert to logs"""
        
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.CRITICAL,
            AlertSeverity.EMERGENCY: logging.CRITICAL
        }.get(alert.severity, logging.WARNING)
        
        logger.log(log_level, f"FRESHNESS ALERT: {alert.message} [ID: {alert.alert_id}]")

    async def _store_alert(self, alert: DataStalenessAlert):
        """Store alert in database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO freshness_alerts 
                (alert_id, timestamp, source, data_type, symbol, last_update, 
                 age_minutes, severity, message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id,
                int(alert.timestamp.timestamp()),
                alert.source.value,
                alert.data_type,
                alert.symbol,
                int(alert.last_update.timestamp()),
                alert.age.total_seconds() / 60,
                alert.severity.value,
                alert.message
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")

    async def _resolve_alerts(self, source: DataSource, data_type: str):
        """Resolve active alerts for fresh data"""
        
        alert_key = f"{source.value}_{data_type}"
        
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.resolved = True
            alert.resolution_time = datetime.now()
            
            # Update database
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE freshness_alerts 
                    SET resolved = 1, resolution_time = ?
                    WHERE alert_id = ?
                """, (int(alert.resolution_time.timestamp()), alert.alert_id))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                logger.error(f"Failed to update resolved alert: {e}")
            
            # Remove from active alerts
            del self.active_alerts[alert_key]
            
            logger.info(f"Resolved alert for {source.value}/{data_type}")

    async def _update_metrics(self):
        """Update system monitoring metrics"""
        
        try:
            # Count alerts in last 24 hours
            day_ago = datetime.now() - timedelta(hours=24)
            alerts_24h = sum(1 for alert in self.alert_history if alert.timestamp > day_ago)
            
            # Calculate average data age
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT AVG(age_minutes) 
                FROM data_freshness 
                WHERE check_time > ?
            """, (int((datetime.now() - timedelta(hours=1)).timestamp()),))
            
            avg_age = cursor.fetchone()[0] or 0.0
            
            # Update metrics
            self.metrics.alerts_last_24h = alerts_24h
            self.metrics.avg_data_age_minutes = avg_age
            
            # Calculate health score
            if self.metrics.total_data_sources > 0:
                health_score = (self.metrics.healthy_sources / self.metrics.total_data_sources) * 100
                self.metrics.overall_health_score = health_score
            
            # Store metrics in database
            cursor.execute("""
                INSERT INTO system_metrics 
                (timestamp, total_sources, healthy_sources, warning_sources, 
                 critical_sources, overall_health_score, alerts_24h, avg_data_age_minutes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(datetime.now().timestamp()),
                self.metrics.total_data_sources,
                self.metrics.healthy_sources,
                self.metrics.warning_sources,
                self.metrics.critical_sources,
                self.metrics.overall_health_score,
                self.metrics.alerts_last_24h,
                self.metrics.avg_data_age_minutes
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")

    async def _cleanup_loop(self):
        """Background cleanup of old data"""
        
        while True:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Run every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)

    async def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clean data older than 7 days
            week_ago = int((datetime.now() - timedelta(days=7)).timestamp())
            
            cursor.execute("DELETE FROM data_freshness WHERE check_time < ?", (week_ago,))
            freshness_deleted = cursor.rowcount
            
            cursor.execute("DELETE FROM freshness_alerts WHERE timestamp < ? AND resolved = 1", (week_ago,))
            alerts_deleted = cursor.rowcount
            
            cursor.execute("DELETE FROM system_metrics WHERE timestamp < ?", (week_ago,))
            metrics_deleted = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            if freshness_deleted + alerts_deleted + metrics_deleted > 0:
                logger.info(f"Cleaned up {freshness_deleted} freshness records, "
                           f"{alerts_deleted} alerts, {metrics_deleted} metrics")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

    async def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        
        try:
            # Current metrics
            dashboard = {
                'current_metrics': asdict(self.metrics),
                'active_alerts_count': len(self.active_alerts),
                'active_alerts': [
                    {
                        'alert_id': alert.alert_id,
                        'source': alert.source.value,
                        'data_type': alert.data_type,
                        'symbol': alert.symbol,
                        'severity': alert.severity.value,
                        'age_minutes': int(alert.age.total_seconds() / 60),
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in self.active_alerts.values()
                ]
            }
            
            # Recent alerts (last 24 hours)
            day_ago = datetime.now() - timedelta(hours=24)
            recent_alerts = [
                alert for alert in self.alert_history 
                if alert.timestamp > day_ago
            ]
            
            dashboard['recent_alerts'] = [
                {
                    'alert_id': alert.alert_id,
                    'source': alert.source.value,
                    'data_type': alert.data_type,
                    'symbol': alert.symbol,
                    'severity': alert.severity.value,
                    'resolved': alert.resolved,
                    'timestamp': alert.timestamp.isoformat(),
                    'resolution_time': alert.resolution_time.isoformat() if alert.resolution_time else None
                }
                for alert in recent_alerts
            ]
            
            # Source health breakdown
            source_health = {}
            for threshold in self.thresholds:
                source = threshold.source.value
                if source not in source_health:
                    source_health[source] = {'healthy': 0, 'stale': 0, 'total': 0}
                
                source_health[source]['total'] += 1
                
                # Check if this source has active alerts
                alert_key = f"{threshold.source.value}_{threshold.data_type}"
                if alert_key in self.active_alerts:
                    source_health[source]['stale'] += 1
                else:
                    source_health[source]['healthy'] += 1
            
            dashboard['source_health'] = source_health
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Failed to get monitoring dashboard: {e}")
            return {'error': str(e)}

    async def force_freshness_check(self, source: DataSource = None, data_type: str = None):
        """Force an immediate freshness check for specific source/type or all"""
        
        logger.info(f"Forcing freshness check for {source.value if source else 'all sources'}")
        
        if source and data_type:
            # Check specific threshold
            for threshold in self.thresholds:
                if threshold.source == source and threshold.data_type == data_type:
                    last_update = await self._get_last_update_time(threshold.source, threshold.data_type)
                    check_time = datetime.now()
                    age = check_time - last_update if last_update else timedelta(hours=24)
                    is_stale = self._is_data_stale(age, threshold)
                    
                    await self._record_freshness_check(threshold, last_update, check_time, age, is_stale)
                    
                    if is_stale:
                        await self._handle_stale_data(threshold, last_update or check_time, age)
                    else:
                        await self._resolve_alerts(threshold.source, threshold.data_type)
                    break
        else:
            # Check all thresholds
            await self._check_data_freshness()

    def add_custom_threshold(self, threshold: FreshnessThreshold):
        """Add custom freshness threshold"""
        self.thresholds.append(threshold)
        logger.info(f"Added custom threshold for {threshold.source.value}/{threshold.data_type}")

    def configure_alert_channel(self, channel: AlertChannel, config: Dict[str, Any]):
        """Configure alert notification channel"""
        self.alert_channels[channel] = config
        logger.info(f"Configured alert channel: {channel.value}")

# Factory function
async def create_freshness_monitor() -> DataFreshnessMonitor:
    """Create and start data freshness monitor"""
    
    monitor = DataFreshnessMonitor()
    await monitor.start_monitoring()
    
    logger.info("Data freshness monitor created and started")
    return monitor

# Add missing import
import numpy as np