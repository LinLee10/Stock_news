#!/usr/bin/env python3
"""
Earnings API endpoints for calendar and analysis
Provides upcoming earnings and detailed explanations
"""
from flask import Blueprint, request, jsonify
from config.feature_flags import is_earnings_reads_enabled
from services.earnings_service import EarningsAnalysisService
from .auth import require_api_key
import logging

logger = logging.getLogger(__name__)
earnings_blueprint = Blueprint('earnings', __name__)

@earnings_blueprint.route('/earnings/upcoming', methods=['GET'])
@require_api_key
def get_upcoming_earnings():
    """
    GET /earnings/upcoming?days=21
    Get upcoming earnings within specified days with analysis
    """
    if not is_earnings_reads_enabled():
        return jsonify({
            'error': 'Earnings analysis feature is disabled',
            'code': 'FEATURE_DISABLED'
        }), 503
    
    try:
        # Parse query parameters
        days = int(request.args.get('days', '21'))
        
        # Validate days parameter
        if days < 1 or days > 90:
            return jsonify({
                'error': 'Invalid days parameter',
                'details': 'Days must be between 1 and 90'
            }), 400
        
        earnings_service = EarningsAnalysisService()
        upcoming_earnings = earnings_service.get_upcoming_earnings(days=days)
        
        # Add metadata
        response = {
            'query': {
                'days_ahead': days,
                'generated_at': earnings_service._get_current_timestamp()
            },
            'upcoming_earnings': upcoming_earnings,
            'summary': {
                'total_earnings': len(upcoming_earnings),
                'confirmed_count': sum(1 for e in upcoming_earnings if e.get('confirmed', False)),
                'high_risk_count': sum(1 for e in upcoming_earnings 
                                     if e.get('analysis', {}).get('risk_level') == 'high'),
                'avg_implied_move': _calculate_avg_implied_move(upcoming_earnings)
            },
            'meta': {
                'feature_enabled': True,
                'data_sources': ['internal_calendar', 'options_estimation', 'historical_analysis'],
                'disclaimer': (
                    "Earnings predictions and implied moves are estimates based on historical data "
                    "and current market conditions. Actual results may vary significantly. "
                    "This information is for educational purposes only and should not be used "
                    "as the sole basis for investment decisions."
                )
            }
        }
        
        return jsonify(response), 200
        
    except ValueError as e:
        return jsonify({
            'error': 'Invalid parameter',
            'details': str(e)
        }), 400
    except Exception as e:
        logger.exception("Error fetching upcoming earnings")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e) if logger.level <= logging.DEBUG else 'See logs for details'
        }), 500

@earnings_blueprint.route('/earnings/<symbol>/explain', methods=['GET'])
@require_api_key
def explain_earnings_analysis(symbol: str):
    """
    GET /earnings/{symbol}/explain
    Get detailed explanation of earnings analysis for a specific symbol
    """
    if not is_earnings_reads_enabled():
        return jsonify({
            'error': 'Earnings analysis feature is disabled',
            'code': 'FEATURE_DISABLED'
        }), 503
    
    try:
        # Validate symbol format
        symbol = symbol.upper().strip()
        if not symbol or len(symbol) > 10:
            return jsonify({
                'error': 'Invalid symbol',
                'details': 'Symbol must be 1-10 characters'
            }), 400
        
        earnings_service = EarningsAnalysisService()
        explanation = earnings_service.explain_earnings_analysis(symbol)
        
        # Check if analysis was successful
        if 'error' in explanation:
            return jsonify({
                'error': 'Analysis failed',
                'details': explanation['error'],
                'symbol': symbol
            }), 404
        
        return jsonify(explanation), 200
        
    except Exception as e:
        logger.exception(f"Error explaining earnings analysis for {symbol}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e) if logger.level <= logging.DEBUG else 'See logs for details'
        }), 500

def _calculate_avg_implied_move(earnings_list: list) -> float:
    """Calculate average implied move from earnings list"""
    implied_moves = [
        e.get('analysis', {}).get('implied_move_pct', 0) 
        for e in earnings_list 
        if e.get('analysis', {}).get('implied_move_pct') is not None
    ]
    
    if not implied_moves:
        return 0.0
    
    return round(sum(implied_moves) / len(implied_moves), 1)

# Add helper method to EarningsAnalysisService
def _get_current_timestamp():
    """Get current timestamp in ISO format"""
    from datetime import datetime
    return datetime.utcnow().isoformat()

# Monkey patch the helper method
EarningsAnalysisService._get_current_timestamp = _get_current_timestamp