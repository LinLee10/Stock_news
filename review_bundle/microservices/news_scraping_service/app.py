#!/usr/bin/env python3
"""
F18 News Scraping Service - Minimal RSS-only news service
Handles basic RSS feed parsing for microservices mode
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
import xml.etree.ElementTree as ET

# Add shared to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from flask import Flask, jsonify, request
    import requests
    from urllib.parse import urljoin
    import feedparser
except ImportError:
    Flask = None
    requests = None
    feedparser = None

logger = logging.getLogger(__name__)

class NewsScrapingService:
    """Minimal RSS-based news scraping service for F18"""
    
    def __init__(self):
        self.rss_feeds = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://www.cnbc.com/id/100727362/device/rss/rss.html',  # CNBC Top News
            'https://www.marketwatch.com/rss/topstories'
        ]
        
    def fetch_rss_headlines(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """Fetch headlines from RSS feeds with simple retry logic"""
        headlines = []
        
        if feedparser is None:
            logger.warning("feedparser not available, returning mock data")
            return self._get_mock_headlines()
        
        for feed_url in self.rss_feeds:
            try:
                # Simple retry logic
                for attempt in range(2):
                    try:
                        logger.info(f"Fetching RSS feed: {feed_url}")
                        feed = feedparser.parse(feed_url)
                        
                        if feed.bozo:
                            logger.warning(f"RSS feed may be malformed: {feed_url}")
                        
                        for entry in feed.entries[:max_items]:
                            headline = {
                                'title': entry.get('title', 'No title'),
                                'url': entry.get('link', ''),
                                'published': entry.get('published', ''),
                                'summary': entry.get('summary', '')[:200] + '...' if entry.get('summary', '') else '',
                                'source': feed_url,
                                'timestamp': datetime.utcnow().isoformat() + 'Z'
                            }
                            headlines.append(headline)
                            
                            if len(headlines) >= max_items:
                                break
                        
                        break  # Success, don't retry
                        
                    except Exception as e:
                        logger.warning(f"Attempt {attempt + 1} failed for {feed_url}: {e}")
                        if attempt == 1:  # Last attempt
                            logger.error(f"Failed to fetch {feed_url} after retries")
                            
            except Exception as e:
                logger.error(f"Error processing feed {feed_url}: {e}")
        
        # Limit total results
        return headlines[:max_items]
    
    def _get_mock_headlines(self) -> List[Dict[str, Any]]:
        """Mock headlines when feedparser is not available"""
        return [
            {
                'title': 'Market Update: Tech Stocks Rise',
                'url': 'https://example.com/news1',
                'published': datetime.utcnow().isoformat() + 'Z',
                'summary': 'Technology stocks showed strong performance in early trading...',
                'source': 'mock_feed',
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            },
            {
                'title': 'Federal Reserve Policy Update',
                'url': 'https://example.com/news2', 
                'published': datetime.utcnow().isoformat() + 'Z',
                'summary': 'The Federal Reserve announced policy changes affecting interest rates...',
                'source': 'mock_feed',
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
        ]
        
    def create_app(self):
        """Create Flask application for News Scraping Service"""
        if Flask is None:
            raise ImportError("Flask is required for microservices mode")
            
        app = Flask(__name__)
        
        @app.route('/healthz')
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'service': 'news_scraping_service',
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'feeds_configured': len(self.rss_feeds),
                'version': '1.0.0'
            })
        
        @app.route('/news')
        def get_news():
            """Get news headlines from RSS feeds"""
            try:
                # Get max_items from query params
                max_items = request.args.get('limit', default=10, type=int)
                max_items = min(max_items, 50)  # Cap at 50
                
                headlines = self.fetch_rss_headlines(max_items=max_items)
                
                return jsonify({
                    'headlines': headlines,
                    'count': len(headlines),
                    'max_items': max_items,
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'service': 'news_scraping_service'
                })
                
            except Exception as e:
                logger.error(f"Error fetching news: {e}")
                return jsonify({
                    'error': 'Failed to fetch news',
                    'message': str(e),
                    'timestamp': datetime.utcnow().isoformat() + 'Z'
                }), 500
        
        return app

def create_app():
    """Factory function for creating the app"""
    service = NewsScrapingService()
    return service.create_app()

# For running directly
if __name__ == '__main__':
    app = create_app()
    port = int(os.getenv('PORT', 8001))
    app.run(host='0.0.0.0', port=port, debug=False)