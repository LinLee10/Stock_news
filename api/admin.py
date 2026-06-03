#!/usr/bin/env python3
"""
Admin endpoints for monitoring and audit logs
Provides access to recent logs and system status for vNext features
"""
from flask import Blueprint, request, jsonify
from config.feature_flags import is_api_endpoints_enabled
from services.audit_logger import audit_logger
from .auth import require_api_key
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
admin_blueprint = Blueprint('admin', __name__)

@admin_blueprint.route('/logs', methods=['GET'])
@require_api_key
def get_recent_logs():
    """
    GET /logs?hours=24&feature=all&operation=all
    Retrieve recent audit logs for vNext features
    """
    if not is_api_endpoints_enabled():
        return jsonify({
            'error': 'API endpoints are disabled',
            'code': 'FEATURE_DISABLED'
        }), 503
    
    try:
        # Parse query parameters
        hours = int(request.args.get('hours', '24'))
        feature_filter = request.args.get('feature', 'all')
        operation_filter = request.args.get('operation', 'all')
        status_filter = request.args.get('status', 'all')
        
        # Validate hours parameter
        if hours < 1 or hours > 168:  # Max 1 week
            return jsonify({
                'error': 'Invalid hours parameter',
                'details': 'Hours must be between 1 and 168 (1 week)'
            }), 400
        
        # Get recent logs
        logs = audit_logger.get_recent_logs(hours=hours)
        
        # Apply filters
        filtered_logs = []
        for log_entry in logs:
            # Feature filter
            if feature_filter != 'all' and log_entry.get('feature') != feature_filter:
                continue
            
            # Operation filter
            if operation_filter != 'all' and log_entry.get('operation') != operation_filter:
                continue
            
            # Status filter
            if status_filter != 'all' and log_entry.get('status') != status_filter:
                continue
            
            filtered_logs.append(log_entry)
        
        # Get summary statistics
        summary = audit_logger.get_operation_summary(hours=hours)
        
        response = {
            'query': {
                'hours': hours,
                'feature_filter': feature_filter,
                'operation_filter': operation_filter,
                'status_filter': status_filter,
                'generated_at': datetime.utcnow().isoformat()
            },
            'logs': filtered_logs,
            'summary': summary,
            'total_entries': len(filtered_logs),
            'available_filters': {
                'features': list(set(log.get('feature', 'unknown') for log in logs)),
                'operations': list(set(log.get('operation', 'unknown') for log in logs)),
                'statuses': list(set(log.get('status', 'unknown') for log in logs))
            }
        }
        
        return jsonify(response), 200
        
    except ValueError as e:
        return jsonify({
            'error': 'Invalid parameter',
            'details': str(e)
        }), 400
    except Exception as e:
        logger.exception("Error retrieving audit logs")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e) if logger.level <= logging.DEBUG else 'See logs for details'
        }), 500

@admin_blueprint.route('/logs/summary', methods=['GET'])
@require_api_key
def get_logs_summary():
    """
    GET /logs/summary?hours=24
    Get aggregated summary of recent operations
    """
    if not is_api_endpoints_enabled():
        return jsonify({
            'error': 'API endpoints are disabled',
            'code': 'FEATURE_DISABLED'
        }), 503
    
    try:
        hours = int(request.args.get('hours', '24'))
        
        if hours < 1 or hours > 168:
            return jsonify({
                'error': 'Invalid hours parameter',
                'details': 'Hours must be between 1 and 168 (1 week)'
            }), 400
        
        summary = audit_logger.get_operation_summary(hours=hours)
        
        response = {
            'query': {
                'hours': hours,
                'generated_at': datetime.utcnow().isoformat()
            },
            'summary': summary
        }
        
        return jsonify(response), 200
        
    except ValueError as e:
        return jsonify({
            'error': 'Invalid parameter',
            'details': str(e)
        }), 400
    except Exception as e:
        logger.exception("Error retrieving logs summary")
        return jsonify({
            'error': 'Internal server error'
        }), 500

@admin_blueprint.route('/logs/operation/<operation_id>', methods=['GET'])
@require_api_key
def get_operation_logs(operation_id: str):
    """
    GET /logs/operation/{operation_id}
    Get all log entries for a specific operation by log_id
    """
    if not is_api_endpoints_enabled():
        return jsonify({
            'error': 'API endpoints are disabled',
            'code': 'FEATURE_DISABLED'
        }), 503
    
    try:
        # Get logs for the last 7 days to ensure we capture the operation
        all_logs = audit_logger.get_recent_logs(hours=168)
        
        # Filter by log_id (operation_id)
        operation_logs = [
            log for log in all_logs 
            if log.get('log_id') == operation_id
        ]
        
        if not operation_logs:
            return jsonify({
                'error': 'Operation not found',
                'details': f'No logs found for operation {operation_id}'
            }), 404
        
        # Sort by timestamp
        operation_logs.sort(key=lambda x: x.get('timestamp', ''))
        
        # Calculate operation duration if completed
        duration_ms = None
        start_time = None
        end_time = None
        
        for log in operation_logs:
            if log.get('step') == 'started':
                start_time = log.get('timestamp')
            elif log.get('step') in ['completed', 'failed'] and log.get('duration_ms'):
                duration_ms = log.get('duration_ms')
                end_time = log.get('timestamp')
        
        response = {
            'operation_id': operation_id,
            'logs': operation_logs,
            'total_steps': len(operation_logs),
            'timeline': {
                'start_time': start_time,
                'end_time': end_time,
                'duration_ms': duration_ms
            },
            'operation_info': {
                'feature': operation_logs[0].get('feature'),
                'operation': operation_logs[0].get('operation'),
                'symbol': operation_logs[0].get('symbol'),
                'final_status': operation_logs[-1].get('status') if operation_logs else None
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.exception(f"Error retrieving operation logs for {operation_id}")
        return jsonify({
            'error': 'Internal server error'
        }), 500

@admin_blueprint.route('/logs/export', methods=['GET'])
@require_api_key
def export_logs():
    """
    GET /logs/export?hours=24&format=jsonl
    Export logs in various formats (currently JSONL only)
    """
    if not is_api_endpoints_enabled():
        return jsonify({
            'error': 'API endpoints are disabled',
            'code': 'FEATURE_DISABLED'
        }), 503
    
    try:
        hours = int(request.args.get('hours', '24'))
        format_type = request.args.get('format', 'jsonl')
        
        if hours < 1 or hours > 168:
            return jsonify({
                'error': 'Invalid hours parameter',
                'details': 'Hours must be between 1 and 168 (1 week)'
            }), 400
        
        if format_type not in ['jsonl']:
            return jsonify({
                'error': 'Unsupported format',
                'details': 'Only "jsonl" format is currently supported'
            }), 400
        
        logs = audit_logger.get_recent_logs(hours=hours)
        
        if format_type == 'jsonl':
            import json
            # Return as newline-delimited JSON
            jsonl_content = '\n'.join(json.dumps(log) for log in logs)
            
            response = {
                'format': 'jsonl',
                'total_entries': len(logs),
                'time_range_hours': hours,
                'generated_at': datetime.utcnow().isoformat(),
                'content': jsonl_content
            }
            
            return jsonify(response), 200
        
    except ValueError as e:
        return jsonify({
            'error': 'Invalid parameter',
            'details': str(e)
        }), 400
    except Exception as e:
        logger.exception("Error exporting logs")
        return jsonify({
            'error': 'Internal server error'
        }), 500