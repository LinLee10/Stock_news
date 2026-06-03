#!/usr/bin/env python3
"""
Symbol intake endpoint implementation
Handles POST /symbols/intake for on-demand symbol analysis
"""
from flask import Blueprint, request, jsonify
from marshmallow import Schema, fields, ValidationError
from config.feature_flags import is_symbol_intake_enabled
from config.config import SYMBOL_INTAKE_CONFIG
from services.symbol_intake import SymbolIntakeService
from services.job_queue import JobQueue
from services.audit_logger import log_api_request
from .auth import require_api_key
import logging
import uuid
import re
import time

logger = logging.getLogger(__name__)
symbols_blueprint = Blueprint('symbols', __name__)

class SymbolIntakeRequest(Schema):
    """Request schema for symbol intake"""
    ticker = fields.Str(required=True, validate=lambda x: len(x) <= 10)
    company_name = fields.Str(required=True, validate=lambda x: len(x) <= 255)
    priority = fields.Str(missing='normal', validate=lambda x: x in ['low', 'normal', 'high'])
    force_refresh = fields.Bool(missing=False)

class SymbolIntakeResponse(Schema):
    """Response schema for symbol intake"""
    job_id = fields.Str(required=True)
    symbol_id = fields.Str(required=True)
    status = fields.Str(required=True)
    estimated_completion_seconds = fields.Int()
    message = fields.Str()

def validate_ticker_format(ticker: str) -> tuple[bool, str]:
    """Validate ticker symbol format"""
    if not ticker:
        return False, "Ticker cannot be empty"
    
    # Basic ticker validation: 1-10 alphanumeric characters
    if not re.match(r'^[A-Z0-9^.-]{1,10}$', ticker.upper()):
        return False, "Ticker must be 1-10 alphanumeric characters, dots, dashes, or carets"
    
    # Check for common invalid patterns
    invalid_patterns = ['TEST', 'NULL', 'UNDEFINED']
    if ticker.upper() in invalid_patterns:
        return False, f"'{ticker}' is not a valid ticker symbol"
    
    return True, ""

@symbols_blueprint.route('/symbols/intake', methods=['POST'])
@require_api_key
def symbol_intake():
    """
    POST /symbols/intake
    Accept new symbol for analysis and return job tracking info
    """
    with log_api_request("symbols_intake", metadata={'endpoint': '/symbols/intake', 'method': 'POST'}) as op:
        if not is_symbol_intake_enabled():
            op.warning("feature_disabled", "Symbol intake feature is disabled")
            return jsonify({
                'error': 'Symbol intake feature is disabled',
                'code': 'FEATURE_DISABLED'
            }), 503
        
        # Parse and validate request
        schema = SymbolIntakeRequest()
        try:
            data = schema.load(request.json or {})
            op.step("request_validated", count_in=1, metadata={'ticker': data.get('ticker')})
        except ValidationError as e:
            op.warning("validation_failed", f"Request validation failed: {e.messages}")
            return jsonify({
                'error': 'Validation failed',
                'details': e.messages
            }), 400
        
        ticker = data['ticker'].upper()
        company_name = data['company_name']
        priority = data['priority']
        force_refresh = data['force_refresh']
        
        # Update operation context with symbol
        op.symbol = ticker
        
        # Validate ticker format
        is_valid, error_msg = validate_ticker_format(ticker)
        if not is_valid:
            op.warning("invalid_ticker_format", error_msg)
            return jsonify({
                'error': 'Invalid ticker format',
                'details': error_msg
            }), 400
        
        op.step("ticker_validated", metadata={'ticker': ticker, 'company_name': company_name})
        
            # Initialize services
            intake_service = SymbolIntakeService()
            job_queue = JobQueue()
            op.step("services_initialized")
            
            # Check for duplicate/existing symbol
            existing_symbol = intake_service.get_symbol_by_ticker(ticker)
            op.step("duplicate_check", count_in=1 if existing_symbol else 0, 
                   metadata={'exists': bool(existing_symbol), 'force_refresh': force_refresh})
                   
            if existing_symbol and not force_refresh:
                if existing_symbol.get('intake_status') == 'completed':
                    op.step("return_existing", count_out=1, metadata={'status': 'already_exists'})
                    return jsonify({
                        'job_id': existing_symbol.get('last_job_id', ''),
                        'symbol_id': existing_symbol.get('symbol_id', ticker),
                        'status': 'already_exists',
                        'message': f'Symbol {ticker} already analyzed. Use force_refresh=true to re-analyze.'
                    }), 200
                elif existing_symbol.get('intake_status') in ['queued', 'processing']:
                    op.step("return_queued", count_out=1, metadata={'status': 'already_queued'})
                    return jsonify({
                        'job_id': existing_symbol.get('last_job_id', ''),
                        'symbol_id': existing_symbol.get('symbol_id', ticker),
                        'status': 'already_queued',
                        'message': f'Symbol {ticker} is already being processed.'
                    }), 202
            
            # Generate unique IDs
            job_id = str(uuid.uuid4())
            symbol_id = f"sym_{ticker}_{int(time.time())}"
            op.step("ids_generated", metadata={'job_id': job_id, 'symbol_id': symbol_id})
            
            # Upsert symbol record
            symbol_record = intake_service.upsert_symbol(
                ticker=ticker,
                company_name=company_name,
                symbol_id=symbol_id,
                intake_status='queued',
                job_id=job_id
            )
            op.step("symbol_upserted", count_out=1)
            
            # Estimate completion time based on API limits and queue size
            queue_size = job_queue.get_queue_size()
            # Alpha Vantage: 12 seconds between calls + processing time
            estimated_seconds = max(60, queue_size * 15)  # Minimum 1 minute
            op.step("queue_checked", count_in=queue_size, metadata={'estimated_seconds': estimated_seconds})
            
            # Check queue capacity
            max_queue_size = SYMBOL_INTAKE_CONFIG['max_queue_size']
            if queue_size >= max_queue_size:
                op.warning("queue_capacity_exceeded", f"Queue size {queue_size} >= max {max_queue_size}")
                return jsonify({
                    'error': 'Queue capacity exceeded',
                    'details': f'Maximum queue size ({max_queue_size}) reached. Try again later.',
                    'queue_size': queue_size
                }), 429
            
            # Enqueue intake job
            job_data = {
                'ticker': ticker,
                'company_name': company_name,
                'symbol_id': symbol_id,
                'priority': priority,
                'force_refresh': force_refresh,
                'created_at': time.time()
            }
            
            job_queue.enqueue_job(
                job_id=job_id,
                job_type='symbol_intake',
                job_data=job_data,
                priority=priority
            )
            op.step("job_enqueued", count_out=1, metadata={'queue_position': queue_size + 1})
            
            logger.info(f"Enqueued symbol intake job {job_id} for {ticker}")
            
            # Return 202 Accepted with job tracking info
            response_data = SymbolIntakeResponse().dump({
                'job_id': job_id,
                'symbol_id': symbol_id,
                'status': 'queued',
                'estimated_completion_seconds': estimated_seconds,
                'message': f'Symbol {ticker} queued for analysis'
            })
            
            op.step("response_prepared", count_out=1)
            return jsonify(response_data), 202
            
        except Exception as e:
            logger.exception(f"Error processing symbol intake for {ticker}")
            return jsonify({
                'error': 'Internal server error',
                'details': str(e) if logger.level <= logging.DEBUG else 'See logs for details'
            }), 500

@symbols_blueprint.route('/symbols/jobs/<job_id>', methods=['GET'])
@require_api_key
def get_job_status(job_id: str):
    """Get status of a symbol intake job"""
    if not is_symbol_intake_enabled():
        return jsonify({'error': 'Symbol intake feature is disabled'}), 503
    
    try:
        job_queue = JobQueue()
        job_status = job_queue.get_job_status(job_id)
        
        if not job_status:
            return jsonify({'error': 'Job not found'}), 404
        
        return jsonify(job_status), 200
        
    except Exception as e:
        logger.exception(f"Error getting job status for {job_id}")
        return jsonify({'error': 'Internal server error'}), 500

@symbols_blueprint.route('/symbols/<symbol>/intake_status', methods=['GET'])
@require_api_key
def get_symbol_intake_status(symbol: str):
    """
    GET /symbols/{symbol}/intake_status?job_id=...
    Returns detailed intake progress and status
    """
    if not is_symbol_intake_enabled():
        return jsonify({'error': 'Symbol intake feature is disabled'}), 503
    
    try:
        intake_service = SymbolIntakeService()
        job_queue = JobQueue()
        
        # Get job_id from query params or symbol record
        job_id = request.args.get('job_id')
        if not job_id:
            symbol_record = intake_service.get_symbol_by_ticker(symbol)
            if symbol_record:
                job_id = symbol_record.get('last_job_id')
        
        if not job_id:
            return jsonify({'error': 'No job found for symbol'}), 404
        
        # Get job status from queue
        job_status = job_queue.get_job_status(job_id)
        if not job_status:
            return jsonify({'error': 'Job not found'}), 404
        
        # Calculate progress percentage based on status
        progress_map = {
            'queued': 0,
            'processing': 50,
            'completed': 100,
            'failed': 0
        }
        
        # Determine current step
        step_map = {
            'queued': 'Waiting in queue',
            'processing': 'Fetching price data and news',
            'completed': 'Analysis complete',
            'failed': 'Processing failed'
        }
        
        status = job_status.get('status', 'unknown')
        percent = progress_map.get(status, 0)
        last_step = step_map.get(status, 'Unknown status')
        
        # Include any errors
        errors = None
        if status == 'failed' and 'error' in job_status:
            errors = [job_status['error']]
        
        response = {
            'symbol': symbol.upper(),
            'job_id': job_id,
            'state': status,
            'percent': percent,
            'last_step': last_step,
            'created_at': job_status.get('created_at'),
            'updated_at': job_status.get('updated_at')
        }
        
        if errors:
            response['errors'] = errors
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.exception(f"Error getting intake status for {symbol}")
        return jsonify({'error': 'Internal server error'}), 500

@symbols_blueprint.route('/symbols/<symbol>/snapshot', methods=['GET'])
@require_api_key
def get_symbol_snapshot(symbol: str):
    """
    GET /symbols/{symbol}/snapshot
    Returns comprehensive symbol overview: price card, chart data, headlines, earnings
    """
    if not is_symbol_intake_enabled():
        return jsonify({'error': 'Symbol intake feature is disabled'}), 503
    
    try:
        from services.symbol_snapshot import SymbolSnapshotService
        snapshot_service = SymbolSnapshotService()
        
        # Generate snapshot data
        snapshot = snapshot_service.generate_snapshot(symbol)
        
        if not snapshot:
            return jsonify({'error': 'Symbol not found or no data available'}), 404
        
        return jsonify(snapshot), 200
        
    except Exception as e:
        logger.exception(f"Error generating snapshot for {symbol}")
        return jsonify({'error': 'Internal server error'}), 500