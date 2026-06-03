#!/usr/bin/env python3
"""Fake RSS client for DRY_RUN mode"""
import os
from pathlib import Path
from typing import Dict, List, Any
import xml.etree.ElementTree as ET
from datetime import datetime


class FakeRSSClient:
    """Fake RSS client that uses local fixtures instead of network"""
    
    FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "rss"
    
    # Map URLs to fixture files
    URL_TO_FIXTURE = {
        "https://news.google.com/rss": "tech_news_sample.xml",
        "https://feeds.finance.yahoo.com/rss/2.0/headline": "financial_news_sample.xml",
        "https://rss.cnn.com/rss/money_news_international.rss": "financial_news_sample.xml",
        # Add more as needed
    }
    
    def __init__(self):
        self.call_count = 0
    
    @classmethod
    def parse_fixture(cls, url_or_file: str) -> Dict[str, Any]:
        """Parse RSS fixture file and return feedparser-like structure"""
        fixture_file = None
        
        # Check if it's a URL we have a fixture for
        if url_or_file.startswith("http"):
            fixture_file = cls.URL_TO_FIXTURE.get(url_or_file, "tech_news_sample.xml")
        else:
            # Assume it's a file path
            fixture_file = url_or_file
        
        fixture_path = cls.FIXTURE_DIR / fixture_file
        
        if not fixture_path.exists():
            # Fall back to first available fixture
            available_fixtures = list(cls.FIXTURE_DIR.glob("*.xml"))
            if available_fixtures:
                fixture_path = available_fixtures[0]
            else:
                # Return empty feed if no fixtures
                return {
                    'feed': {
                        'title': 'No RSS fixtures available',
                        'link': url_or_file,
                        'description': 'No RSS fixtures found'
                    },
                    'entries': []
                }
        
        try:
            tree = ET.parse(fixture_path)
            root = tree.getroot()
            
            # Parse RSS structure
            channel = root.find('channel')
            if channel is None:
                return {'feed': {}, 'entries': []}
            
            # Extract feed metadata
            feed_info = {
                'title': cls._get_text(channel, 'title', 'Unknown Feed'),
                'link': cls._get_text(channel, 'link', url_or_file),
                'description': cls._get_text(channel, 'description', 'No description'),
            }
            
            # Extract entries
            entries = []
            for item in channel.findall('item'):
                entry = {
                    'title': cls._get_text(item, 'title', 'No title'),
                    'link': cls._get_text(item, 'link', ''),
                    'description': cls._get_text(item, 'description', ''),
                    'published': cls._get_text(item, 'pubDate', ''),
                    'guid': cls._get_text(item, 'guid', ''),
                    'summary': cls._get_text(item, 'description', ''),  # feedparser compatibility
                }
                
                # Parse date if available
                pub_date = entry['published']
                if pub_date:
                    try:
                        # Convert to feedparser-like time structure
                        entry['published_parsed'] = datetime.strptime(
                            pub_date, "%a, %d %b %Y %H:%M:%S GMT"
                        ).timetuple()
                    except ValueError:
                        entry['published_parsed'] = None
                
                entries.append(entry)
            
            return {
                'feed': feed_info,
                'entries': entries,
                'status': 200,
                'version': 'rss20'
            }
            
        except Exception as e:
            # Return error structure
            return {
                'feed': {
                    'title': f'Error parsing fixture: {e}',
                    'link': url_or_file,
                    'description': 'RSS parsing error'
                },
                'entries': [],
                'status': 500
            }
    
    @staticmethod
    def _get_text(element: ET.Element, tag: str, default: str = '') -> str:
        """Get text content of XML element"""
        child = element.find(tag)
        return child.text if child is not None and child.text else default
    
    async def fetch_feed(self, url: str) -> Dict[str, Any]:
        """Fetch RSS feed (fake - uses fixtures)"""
        self.call_count += 1
        return self.parse_fixture(url)
    
    def fetch_feed_sync(self, url: str) -> Dict[str, Any]:
        """Synchronous version"""
        self.call_count += 1
        return self.parse_fixture(url)


# For direct feedparser imports
def parse(url_or_file: str) -> Dict[str, Any]:
    """Fake feedparser.parse() function"""
    return FakeRSSClient.parse_fixture(url_or_file)