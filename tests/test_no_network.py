#!/usr/bin/env python3
"""Test that network access is properly blocked in DRY_RUN mode"""
import pytest
import socket
from settings import DRY_RUN


@pytest.mark.skipif(not DRY_RUN, reason="Only test network blocking in DRY_RUN mode")
def test_socket_blocked():
    """Verify socket connections fail in DRY_RUN mode"""
    with pytest.raises(ConnectionError, match="Network access blocked"):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("8.8.8.8", 80))


@pytest.mark.skipif(not DRY_RUN, reason="Only test network blocking in DRY_RUN mode")  
def test_requests_blocked():
    """Verify requests are blocked in DRY_RUN mode"""
    import requests
    
    with pytest.raises(ConnectionError, match="blocked in DRY_RUN mode"):
        requests.get("https://example.com")


@pytest.mark.skipif(not DRY_RUN, reason="Only test network blocking in DRY_RUN mode")
def test_urllib_blocked():
    """Verify urllib is blocked in DRY_RUN mode"""
    import urllib.request
    
    with pytest.raises(ConnectionError, match="blocked in DRY_RUN mode"):
        urllib.request.urlopen("https://example.com")


@pytest.mark.skipif(not DRY_RUN, reason="Only test network blocking in DRY_RUN mode")
def test_feedparser_blocked():
    """Verify feedparser doesn't hit network in DRY_RUN mode"""
    try:
        import feedparser
        
        # This should use our fake RSS client, not hit the network
        result = feedparser.parse("https://news.google.com/rss")
        
        # Should return fixture data, not real network data
        assert 'entries' in result
        assert len(result.entries) > 0
        
        # Check it's fixture data (deterministic)
        assert result.entries[0].title == "Fake News Article 1"
        
    except ImportError:
        pytest.skip("feedparser not available")