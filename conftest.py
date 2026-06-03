#!/usr/bin/env python3
"""pytest configuration for dry-run mode"""
import os
import pytest
import socket
from freezegun import freeze_time
from unittest.mock import patch, MagicMock
import random
import numpy as np

# Import our settings
from settings import DRY_RUN

# Enable pytest-socket when DRY_RUN=1
if DRY_RUN:
    try:
        import pytest_socket
        pytest_socket.disable_socket()
    except ImportError:
        pass  # Fallback to manual socket patching


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "network: marks tests requiring network")


def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers in DRY_RUN mode"""
    if DRY_RUN:
        skip_e2e = pytest.mark.skip(reason="Skipping e2e tests in DRY_RUN mode")
        skip_slow = pytest.mark.skip(reason="Skipping slow tests in DRY_RUN mode")
        skip_network = pytest.mark.skip(reason="Skipping network tests in DRY_RUN mode")
        
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)
            elif "slow" in item.keywords:
                item.add_marker(skip_slow)
            elif "network" in item.keywords:
                item.add_marker(skip_network)


@pytest.fixture(autouse=True)
def block_network_hard():
    """HARD network blocking - no whitelists"""
    if not DRY_RUN:
        yield
        return
    
    def blocked_socket(*args, **kwargs):
        raise ConnectionError("🚫 Network access blocked in DRY_RUN mode - all sockets disabled")
    
    def blocked_urlopen(*args, **kwargs):
        raise ConnectionError("🚫 urllib blocked in DRY_RUN mode")
    
    def blocked_request(*args, **kwargs):
        raise ConnectionError("🚫 HTTP requests blocked in DRY_RUN mode")
    
    patches = [
        patch('socket.socket', blocked_socket),
        patch('socket.create_connection', blocked_socket),
        patch('urllib.request.urlopen', blocked_urlopen),
        patch('urllib3.poolmanager.PoolManager.urlopen', blocked_request),
        patch('requests.Session.request', blocked_request),
        patch('requests.get', blocked_request),
        patch('requests.post', blocked_request),
        patch('httpx.get', blocked_request),
        patch('httpx.post', blocked_request),
        patch('aiohttp.ClientSession.request', blocked_request),
    ]
    
    started = []
    for p in patches:
        try:
            started.append(p.start())
        except Exception:
            pass
    
    yield
    
    for p in patches:
        try:
            p.stop()
        except Exception:
            pass


@pytest.fixture(autouse=True)
def freeze_test_time():
    """Freeze time for deterministic tests"""
    if DRY_RUN:
        with freeze_time("2024-01-15 10:00:00"):
            yield
    else:
        yield


@pytest.fixture(autouse=True)
def seed_all_randomness():
    """Seed all sources of randomness"""
    if DRY_RUN:
        random.seed(42)
        np.random.seed(42)
        os.environ['PYTHONHASHSEED'] = '42'
    yield


@pytest.fixture(autouse=True)
def patch_externals():
    """Auto-patch all external services with fakes"""
    if not DRY_RUN:
        yield
        return
    
    patches = []
    
    # Alpha Vantage
    try:
        from fakes.alpha_vantage import FakeAlphaVantageManager
        patches.append(patch('services.alpha_vantage_manager.AlphaVantageManager', FakeAlphaVantageManager))
    except ImportError:
        pass
    
    # yfinance
    try:
        from fakes.yfinance import fake_download, FakeTicker
        patches.append(patch('yfinance.download', fake_download))
        patches.append(patch('yfinance.Ticker', FakeTicker))
    except ImportError:
        pass
    
    # Email
    try:
        from fakes.email import FakeEmailSender
        patches.append(patch('services.email_sender.EmailSender', FakeEmailSender))
        patches.append(patch('email.send_email', FakeEmailSender().send_email))
    except ImportError:
        pass
    
    # RSS/News
    try:
        from fakes.rss import FakeRSSClient
        patches.append(patch('services.rss_client.RSSClient', FakeRSSClient))
        patches.append(patch('feedparser.parse', FakeRSSClient.parse_fixture))
    except ImportError:
        pass
    
    # LLM clients
    try:
        from fakes.llm import FakeLLMClient
        patches.append(patch('openai.ChatCompletion.create', FakeLLMClient().chat_completion))
        patches.append(patch('anthropic.Client.messages.create', FakeLLMClient().chat_completion))
    except ImportError:
        pass
    
    # structlog shim
    try:
        from fakes.structlog_shim import get_logger
        patches.append(patch('structlog.get_logger', get_logger))
    except ImportError:
        pass
    
    started = []
    for p in patches:
        try:
            started.append(p.start())
        except Exception:
            pass
    
    yield
    
    for p in patches:
        try:
            p.stop()
        except Exception:
            pass


@pytest.fixture
def disable_socket():
    """Explicitly disable socket for specific tests"""
    def blocked_socket(*args, **kwargs):
        raise ConnectionError("Socket access blocked by test")
    
    with patch('socket.socket', blocked_socket):
        yield