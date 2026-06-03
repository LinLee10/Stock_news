#!/usr/bin/env python3
"""
Structured logging and audit trail service for vNext features
Provides consistent logging for symbols, steps, durations, counts, and source links
"""
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class AuditLogEntry:
    """Structured audit log entry"""
    timestamp: str
    log_id: str
    feature: str  # symbol_intake, news_clustering, recommendations, earnings
    operation: str  # intake_job, clustering_job, generate_recs, analyze_earnings
    symbol: Optional[str]
    step: str  # validate, fetch_data, analyze, persist, etc.
    duration_ms: Optional[int]
    count_in: Optional[int]  # Input records/items
    count_out: Optional[int]  # Output records/items
    source_links: List[str]  # URLs or file paths accessed
    status: str  # started, completed, failed, warning
    metadata: Dict[str, Any]  # Additional context
    error_message: Optional[str] = None


class AuditLogger:
    """Centralized audit logging service for vNext features"""
    
    def __init__(self):
        self.data_path = Path("data")
        self.logs_path = self.data_path / "audit_logs"
        self.logs_path.mkdir(exist_ok=True)
        
        # Configure structured logging
        self._setup_structured_logging()
    
    def _setup_structured_logging(self):
        """Configure JSON structured logging for audit trail"""
        log_file = self.logs_path / f"vnext_audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        # BEGIN F15_REDACTION
        # Apply centralized redaction to all audit log messages
        from utils.redact import redact
        
        class RedactingFormatter(logging.Formatter):
            """Formatter that applies redaction to log messages"""
            def format(self, record):
                # Redact the main message
                record.msg = redact(str(record.msg))
                
                # Redact any additional arguments
                if record.args:
                    record.args = tuple(redact(str(arg)) for arg in record.args)
                
                return super().format(record)
        
        # Install redacting formatter for all audit logging
        handler = logging.FileHandler(log_file)
        handler.setFormatter(RedactingFormatter())
        audit_logger = logging.getLogger('audit')
        audit_logger.addHandler(handler)
        audit_logger.setLevel(logging.INFO)
        # END F15_REDACTION
        
        # Create custom formatter for JSON logs
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                if hasattr(record, 'audit_data'):
                    return json.dumps(record.audit_data)
                return super().format(record)
        
        # Set up file handler for audit logs
        audit_handler = logging.FileHandler(log_file)
        audit_handler.setFormatter(JSONFormatter())
        audit_handler.setLevel(logging.INFO)
        
        # Create dedicated audit logger
        self.audit_logger = logging.getLogger('vnext_audit')
        self.audit_logger.setLevel(logging.INFO)
        self.audit_logger.addHandler(audit_handler)
        self.audit_logger.propagate = False
    
    def log_entry(self, entry: AuditLogEntry):
        """Log a structured audit entry"""
        try:
            log_data = asdict(entry)
            self.audit_logger.info("audit_entry", extra={'audit_data': log_data})
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def start_operation(self, feature: str, operation: str, symbol: Optional[str] = None, 
                       metadata: Dict[str, Any] = None) -> str:
        """Start logging an operation and return log_id for tracking"""
        log_id = str(uuid.uuid4())
        
        entry = AuditLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            log_id=log_id,
            feature=feature,
            operation=operation,
            symbol=symbol,
            step="started",
            duration_ms=None,
            count_in=None,
            count_out=None,
            source_links=[],
            status="started",
            metadata=metadata or {},
            error_message=None
        )
        
        self.log_entry(entry)
        return log_id
    
    def log_step(self, log_id: str, step: str, count_in: Optional[int] = None,
                count_out: Optional[int] = None, source_links: List[str] = None,
                metadata: Dict[str, Any] = None):
        """Log a step within an operation"""
        entry = AuditLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            log_id=log_id,
            feature="",  # Will be filled from context
            operation="",  # Will be filled from context
            symbol=None,  # Will be filled from context
            step=step,
            duration_ms=None,
            count_in=count_in,
            count_out=count_out,
            source_links=source_links or [],
            status="in_progress",
            metadata=metadata or {}
        )
        
        self.log_entry(entry)
    
    def complete_operation(self, log_id: str, duration_ms: int, count_out: Optional[int] = None,
                          source_links: List[str] = None, metadata: Dict[str, Any] = None):
        """Complete an operation with final metrics"""
        entry = AuditLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            log_id=log_id,
            feature="",  # Will be filled from context
            operation="",  # Will be filled from context
            symbol=None,  # Will be filled from context
            step="completed",
            duration_ms=duration_ms,
            count_in=None,
            count_out=count_out,
            source_links=source_links or [],
            status="completed",
            metadata=metadata or {}
        )
        
        self.log_entry(entry)
    
    def fail_operation(self, log_id: str, error_message: str, duration_ms: int,
                      metadata: Dict[str, Any] = None):
        """Mark an operation as failed"""
        entry = AuditLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            log_id=log_id,
            feature="",  # Will be filled from context
            operation="",  # Will be filled from context
            symbol=None,  # Will be filled from context
            step="failed",
            duration_ms=duration_ms,
            count_in=None,
            count_out=None,
            source_links=[],
            status="failed",
            metadata=metadata or {},
            error_message=error_message
        )
        
        self.log_entry(entry)
    
    @contextmanager
    def operation_context(self, feature: str, operation: str, symbol: Optional[str] = None,
                         metadata: Dict[str, Any] = None):
        """Context manager for tracking operation duration and auto-logging"""
        log_id = self.start_operation(feature, operation, symbol, metadata)
        start_time = time.time()
        
        try:
            yield OperationLogger(self, log_id, feature, operation, symbol)
            
            duration_ms = int((time.time() - start_time) * 1000)
            self.complete_operation(log_id, duration_ms)
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.fail_operation(log_id, str(e), duration_ms)
            raise
    
    def get_recent_logs(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Retrieve audit logs from the last N hours"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            logs = []
            
            # Check current day's log file
            current_log = self.logs_path / f"vnext_audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
            if current_log.exists():
                logs.extend(self._read_log_file(current_log, cutoff_time))
            
            # Check previous day's log file if needed
            yesterday = datetime.now() - timedelta(days=1)
            prev_log = self.logs_path / f"vnext_audit_{yesterday.strftime('%Y%m%d')}.jsonl"
            if prev_log.exists():
                logs.extend(self._read_log_file(prev_log, cutoff_time))
            
            # Sort by timestamp
            logs.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return logs
            
        except Exception as e:
            logger.error(f"Failed to retrieve recent logs: {e}")
            return []
    
    def _read_log_file(self, log_file: Path, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Read and filter log entries from a file"""
        logs = []
        
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                        
                        if entry_time >= cutoff_time:
                            logs.append(entry)
                            
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading log file {log_file}: {e}")
        
        return logs
    
    def get_operation_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Generate summary statistics for operations in the last N hours"""
        logs = self.get_recent_logs(hours)
        
        if not logs:
            return {
                'total_operations': 0,
                'by_feature': {},
                'by_status': {},
                'avg_duration_ms': 0,
                'total_symbols_processed': 0
            }
        
        # Aggregate statistics
        by_feature = {}
        by_status = {}
        durations = []
        symbols = set()
        
        for entry in logs:
            feature = entry.get('feature', 'unknown')
            status = entry.get('status', 'unknown')
            
            by_feature[feature] = by_feature.get(feature, 0) + 1
            by_status[status] = by_status.get(status, 0) + 1
            
            if entry.get('duration_ms'):
                durations.append(entry['duration_ms'])
            
            if entry.get('symbol'):
                symbols.add(entry['symbol'])
        
        return {
            'total_operations': len(logs),
            'by_feature': by_feature,
            'by_status': by_status,
            'avg_duration_ms': int(sum(durations) / len(durations)) if durations else 0,
            'total_symbols_processed': len(symbols),
            'time_range_hours': hours
        }


class OperationLogger:
    """Helper class for logging within an operation context"""
    
    def __init__(self, audit_logger: AuditLogger, log_id: str, feature: str, 
                 operation: str, symbol: Optional[str]):
        self.audit_logger = audit_logger
        self.log_id = log_id
        self.feature = feature
        self.operation = operation
        self.symbol = symbol
    
    def step(self, step: str, count_in: Optional[int] = None, count_out: Optional[int] = None,
             source_links: List[str] = None, metadata: Dict[str, Any] = None):
        """Log a step with enriched context"""
        entry = AuditLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            log_id=self.log_id,
            feature=self.feature,
            operation=self.operation,
            symbol=self.symbol,
            step=step,
            duration_ms=None,
            count_in=count_in,
            count_out=count_out,
            source_links=source_links or [],
            status="in_progress",
            metadata=metadata or {}
        )
        
        self.audit_logger.log_entry(entry)
    
    def warning(self, step: str, message: str, metadata: Dict[str, Any] = None):
        """Log a warning within the operation"""
        entry = AuditLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            log_id=self.log_id,
            feature=self.feature,
            operation=self.operation,
            symbol=self.symbol,
            step=step,
            duration_ms=None,
            count_in=None,
            count_out=None,
            source_links=[],
            status="warning",
            metadata=metadata or {},
            error_message=message
        )
        
        self.audit_logger.log_entry(entry)


# Global audit logger instance
audit_logger = AuditLogger()


# Convenience functions for common operations
def log_symbol_intake(symbol: str, metadata: Dict[str, Any] = None):
    """Start logging a symbol intake operation"""
    return audit_logger.operation_context("symbol_intake", "intake_job", symbol, metadata)


def log_news_processing(operation: str, symbol: Optional[str] = None, metadata: Dict[str, Any] = None):
    """Start logging a news processing operation"""
    return audit_logger.operation_context("news_processing", operation, symbol, metadata)


def log_recommendation_generation(symbol: Optional[str] = None, metadata: Dict[str, Any] = None):
    """Start logging a recommendation generation operation"""
    return audit_logger.operation_context("recommendations", "generate_recs", symbol, metadata)


def log_earnings_analysis(symbol: str, metadata: Dict[str, Any] = None):
    """Start logging an earnings analysis operation"""
    return audit_logger.operation_context("earnings", "analyze_earnings", symbol, metadata)


def log_api_request(endpoint: str, symbol: Optional[str] = None, metadata: Dict[str, Any] = None):
    """Start logging an API request"""
    return audit_logger.operation_context("api", f"request_{endpoint}", symbol, metadata)