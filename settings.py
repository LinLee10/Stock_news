#!/usr/bin/env python3
"""Centralized settings for DRY_RUN and testing modes"""
import os
from typing import Literal

# Core DRY_RUN settings
DRY_RUN = os.getenv('DRY_RUN', '0') == '1'
DISABLE_EMAIL_SEND = os.getenv('DISABLE_EMAIL_SEND', '0') == '1'
FORCE_CACHE_ONLY = os.getenv('FORCE_CACHE_ONLY', '0') == '1'

# Data source configuration
NEWS_SOURCE: Literal["live", "fixture", "disabled"] = os.getenv('NEWS_SOURCE', 'live')
PRICE_SOURCE: Literal["live", "cache", "disabled"] = os.getenv('PRICE_SOURCE', 'live')

# Matplotlib backend
MATPLOTLIB_BACKEND = os.getenv('MATPLOTLIB_BACKEND', 'Agg' if DRY_RUN else 'default')

# Artifacts directory
ARTIFACTS_DIR = os.getenv('ARTIFACTS_DIR', 'artifacts')
EMAIL_ARTIFACTS_DIR = os.path.join(ARTIFACTS_DIR, 'emails')

# Ensure directories exist
os.makedirs(EMAIL_ARTIFACTS_DIR, exist_ok=True)

def get_client_factory():
    """Factory that returns real or fake clients based on DRY_RUN"""
    if DRY_RUN:
        from fakes.client_factory import FakeClientFactory
        return FakeClientFactory()
    else:
        from services.client_factory import RealClientFactory
        return RealClientFactory()