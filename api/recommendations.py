#!/usr/bin/env python3
"""
Recommendations endpoint implementation
Provides buy/hold/reduce/exit recommendations for watchlist and portfolio
"""
from flask import Blueprint, request, jsonify
from marshmallow import Schema, fields, ValidationError
from config.feature_flags import is_recos_enabled
from services.recommendations_service import RecommendationsService
from services.audit_logger import log_api_request
from .auth import require_api_key
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
recommendations_blueprint = Blueprint('recommendations', __name__)

class RecommendationsRequest(Schema):
    """Request schema for recommendations"""
    scope = fields.Str(required=True, validate=lambda x: x in ['watchlist', 'portfolio'])
    include_details = fields.Bool(missing=False)
    max_age_hours = fields.Int(missing=24)

@recommendations_blueprint.route('/recs', methods=['GET'])
@require_api_key
def get_recommendations():
    """
    GET /recs?scope=watchlist|portfolio
    Generate recommendations for tracked symbols with financial context
    """
    with log_api_request("recommendations", metadata={'endpoint': '/recs', 'method': 'GET'}) as op:
        if not is_recos_enabled():
            op.warning("feature_disabled", "Recommendations feature is disabled")
            return jsonify({
                'error': 'Recommendations feature is disabled',
                'code': 'FEATURE_DISABLED'
            }), 503
        
        # Parse and validate query parameters
        scope = request.args.get('scope', 'watchlist')
        include_details = request.args.get('include_details', 'false').lower() == 'true'
        max_age_hours = int(request.args.get('max_age_hours', '24'))
        
        op.step("params_parsed", metadata={'scope': scope, 'include_details': include_details, 'max_age_hours': max_age_hours})
        
        if scope not in ['watchlist', 'portfolio']:
            op.warning("invalid_scope", f"Invalid scope: {scope}")
            return jsonify({
                'error': 'Invalid scope',
                'details': 'Scope must be "watchlist" or "portfolio"'
            }), 400
        
        try:
            recs_service = RecommendationsService()
            
            if scope == 'watchlist':
                recommendations = recs_service.generate_watchlist_recommendations(
                    include_details=include_details,
                    max_age_hours=max_age_hours
                )
            else:  # portfolio
                recommendations = recs_service.generate_portfolio_recommendations(
                    include_details=include_details,
                    max_age_hours=max_age_hours
                )
            
            # Add metadata and disclaimer
            response = {
                'scope': scope,
                'generated_at': datetime.utcnow().isoformat(),
                'recommendations': recommendations,
                'summary': _generate_summary(recommendations),
                'meta': {
                    'disclaimer': _get_financial_disclaimer(),
                    'data_freshness_hours': max_age_hours,
                    'recommendation_count': len(recommendations),
                    'feature_flags': {
                        'news_corroboration': True,  # Would check actual flags
                        'earnings_analysis': True
                    }
                }
            }
            
            return jsonify(response), 200
            
        except Exception as e:
            logger.exception(f"Error generating {scope} recommendations")
            return jsonify({
                'error': 'Internal server error',
                'details': str(e) if logger.level <= logging.DEBUG else 'See logs for details'
            }), 500

def _generate_summary(recommendations: list) -> dict:
    """Generate summary statistics for recommendations"""
    if not recommendations:
        return {'total': 0}
    
    actions = [rec['action'] for rec in recommendations]
    action_counts = {}
    for action in ['buy', 'hold', 'reduce', 'exit']:
        action_counts[action] = actions.count(action)
    
    # Calculate average confidence
    confidences = [rec['confidence'] for rec in recommendations]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    # Find highest conviction recommendations
    high_conviction = [rec for rec in recommendations if rec['confidence'] >= 0.8]
    
    summary = {
        'total': len(recommendations),
        'action_breakdown': action_counts,
        'average_confidence': round(avg_confidence, 2),
        'high_conviction_count': len(high_conviction),
        'top_recommendations': sorted(recommendations, key=lambda x: x['confidence'], reverse=True)[:3]
    }
    
    return summary

def _get_financial_disclaimer() -> str:
    """Return financial advice disclaimer"""
    return (
        "IMPORTANT DISCLAIMER: These recommendations are for informational purposes only and "
        "do not constitute financial advice, investment advice, or trading advice. "
        "All investment decisions should be made based on your own research and risk tolerance. "
        "Past performance does not guarantee future results. Please consult with a qualified "
        "financial advisor before making any investment decisions. The algorithms and models "
        "used may contain errors or biases, and market conditions can change rapidly."
    )
