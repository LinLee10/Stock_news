#!/usr/bin/env python3
"""
Feature flags configuration for Stonk News system
Controls rollout of new features without breaking existing functionality
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv("config/secrets.env")

class FeatureFlags:
    """Centralized feature flag management"""
    
    def __init__(self):
        self._flags = {
            'enable_symbol_intake': self._get_bool_env('ENABLE_SYMBOL_INTAKE', False),
            'enable_news_corroboration': self._get_bool_env('ENABLE_NEWS_CORROBORATION', False), 
            'enable_earnings_reads': self._get_bool_env('ENABLE_EARNINGS_READS', False),
            'enable_recos': self._get_bool_env('ENABLE_RECOS', False),
            'enable_90_day_sentiment': self._get_bool_env('ENABLE_90_DAY_SENTIMENT', False),
            'enable_multisource_prices': self._get_bool_env('ENABLE_MULTISOURCE_PRICES', False),
            'enable_paid_sources': self._get_bool_env('ENABLE_PAID_SOURCES', False),
            'enable_alpha_vantage_batching': self._get_bool_env('ENABLE_ALPHA_VANTAGE_BATCHING', False),
            'enable_newsapi_ingestion': self._get_bool_env('ENABLE_NEWSAPI_INGESTION', False),
            'enable_finbert_pipeline': self._get_bool_env('ENABLE_FINBERT_PIPELINE', False),
            'enable_finbert_backtest': self._get_bool_env('ENABLE_FINBERT_BACKTEST', False),
            'enable_portfolio_analytics': self._get_bool_env('ENABLE_PORTFOLIO_ANALYTICS', False),
            'enable_smart_alerts': self._get_bool_env('ENABLE_SMART_ALERTS', False),
            # BEGIN F-YF-RATE-FLAGS
            'enable_yf_prices': self._get_bool_env('ENABLE_YF_PRICES', False),
            'enable_yf_daily_refresh': self._get_bool_env('ENABLE_YF_DAILY_REFRESH', False),
            'enable_yf_profiles': self._get_bool_env('ENABLE_YF_PROFILES', False),
            'enable_yf_backoff_debug': self._get_bool_env('ENABLE_YF_BACKOFF_DEBUG', False),
            # END F-YF-RATE-FLAGS
            # Additional operational flags
            'enable_debug_mode': self._get_bool_env('ENABLE_DEBUG_MODE', False),
            'enable_api_endpoints': self._get_bool_env('ENABLE_API_ENDPOINTS', False),
            'enable_async_io': self._get_bool_env('ENABLE_ASYNC_IO', False),
            'enable_timescale_persistence': self._get_bool_env('ENABLE_TIMESCALE_PERSISTENCE', False),
            'enable_vector_search': self._get_bool_env('ENABLE_VECTOR_SEARCH', False),
            'enable_alt_forecasts': self._get_bool_env('ENABLE_ALT_FORECASTS', False),
            'enable_timegpt_stub': self._get_bool_env('ENABLE_TIMEGPT_STUB', False),
            'enable_correlation': self._get_bool_env('ENABLE_CORRELATION', False),
            'enable_gnn_scaffold': self._get_bool_env('ENABLE_GNN_SCAFFOLD', False),
            'enable_microservices_mode': self._get_bool_env('ENABLE_MICROSERVICES_MODE', False)
        }
    
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Parse boolean environment variable with safe defaults"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on', 'enabled')
    
    def is_enabled(self, flag_name: str) -> bool:
        """Check if a feature flag is enabled"""
        return self._flags.get(flag_name, False)
    
    def get_all_flags(self) -> Dict[str, bool]:
        """Get all feature flags for debugging"""
        return self._flags.copy()
    
    def set_flag(self, flag_name: str, enabled: bool) -> None:
        """Runtime flag override (for testing)"""
        if flag_name in self._flags:
            self._flags[flag_name] = enabled

# Global instance
feature_flags = FeatureFlags()

# Convenience functions for common usage
def is_symbol_intake_enabled() -> bool:
    return feature_flags.is_enabled('enable_symbol_intake')

def is_news_corroboration_enabled() -> bool:
    return feature_flags.is_enabled('enable_news_corroboration')

def is_earnings_reads_enabled() -> bool:
    return feature_flags.is_enabled('enable_earnings_reads')

def is_recos_enabled() -> bool:
    return feature_flags.is_enabled('enable_recos')

def is_90_day_sentiment_enabled() -> bool:
    return feature_flags.is_enabled('enable_90_day_sentiment')

def is_multisource_prices_enabled() -> bool:
    return feature_flags.is_enabled('enable_multisource_prices')

def is_paid_sources_enabled() -> bool:
    return feature_flags.is_enabled('enable_paid_sources')

def is_alpha_vantage_batching_enabled() -> bool:
    return feature_flags.is_enabled('enable_alpha_vantage_batching')

def is_newsapi_ingestion_enabled() -> bool:
    return feature_flags.is_enabled('enable_newsapi_ingestion')

def is_finbert_pipeline_enabled() -> bool:
    return feature_flags.is_enabled('enable_finbert_pipeline')

def is_finbert_backtest_enabled() -> bool:
    return feature_flags.is_enabled('enable_finbert_backtest')

def is_portfolio_analytics_enabled() -> bool:
    return feature_flags.is_enabled('enable_portfolio_analytics')

def is_yf_prices_enabled() -> bool:
    return feature_flags.is_enabled('enable_yf_prices')

def is_yf_daily_refresh_enabled() -> bool:
    return feature_flags.is_enabled('enable_yf_daily_refresh')

def is_yf_profiles_enabled() -> bool:
    return feature_flags.is_enabled('enable_yf_profiles')

def is_yf_backoff_debug_enabled() -> bool:
    return feature_flags.is_enabled('enable_yf_backoff_debug')

def is_smart_alerts_enabled() -> bool:
    return feature_flags.is_enabled('enable_smart_alerts')

def is_api_endpoints_enabled() -> bool:
    return feature_flags.is_enabled('enable_api_endpoints')

def is_async_io_enabled() -> bool:
    return feature_flags.is_enabled('enable_async_io')

def is_timescale_persistence_enabled() -> bool:
    return feature_flags.is_enabled('enable_timescale_persistence')

def is_vector_search_enabled() -> bool:
    return feature_flags.is_enabled('enable_vector_search')

def is_alt_forecasts_enabled() -> bool:
    return feature_flags.is_enabled('enable_alt_forecasts')

def is_timegpt_stub_enabled() -> bool:
    return feature_flags.is_enabled('enable_timegpt_stub')

def is_correlation_enabled() -> bool:
    return feature_flags.is_enabled('enable_correlation')

def is_gnn_scaffold_enabled() -> bool:
    return feature_flags.is_enabled('enable_gnn_scaffold')

def is_microservices_mode_enabled() -> bool:
    return feature_flags.is_enabled('enable_microservices_mode')