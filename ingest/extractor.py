"""
FinancialContentExtractor - Extract article content and metadata from HTML.

This module provides content extraction capabilities that detect paywalls,
extract article text and metadata, and handle various website structures.
"""

import re
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse, urljoin

import structlog
from selectolax.parser import HTMLParser

from .models import ExtractedArticle, PaywallStatus, RenderResult

logger = structlog.get_logger(__name__)


class PaywallDetector:
    """Detects paywalls and subscription barriers"""
    
    # Common paywall indicators
    PAYWALL_INDICATORS = [
        # Text-based indicators
        "subscribe to continue",
        "subscription required",
        "unlock this story",
        "become a member",
        "premium content",
        "subscriber only",
        "login to read",
        "create free account",
        "start your free trial",
        "this article is for subscribers",
        "upgrade to premium",
        "paywall",
        
        # CSS classes/IDs commonly used for paywalls
        "paywall",
        "subscription-required",
        "premium-content",
        "subscriber-only",
        "login-wall",
        "subscription-wall",
        "content-gate",
        "article-gate",
        "meter-content",
        
        # JavaScript-based paywall indicators
        "piano-paywall",
        "tinypass",
        "subscriptions-section",
        "registration-required",
    ]
    
    BLUR_INDICATORS = [
        "blur",
        "fade-out",
        "content-fade",
        "paywall-blur",
        "subscription-blur"
    ]
    
    @classmethod
    def detect_paywall(cls, html: str, final_url: str) -> PaywallStatus:
        """Detect paywall status from HTML content"""
        html_lower = html.lower()
        
        # Check for explicit paywall indicators
        paywall_count = sum(1 for indicator in cls.PAYWALL_INDICATORS if indicator in html_lower)
        
        if paywall_count >= 2:
            return PaywallStatus.PAYWALLED
        elif paywall_count == 1:
            # Single indicator might be a preview/teaser
            return PaywallStatus.PREVIEW
        
        # Check for blur/fade effects (visual paywall indicators)
        blur_count = sum(1 for indicator in cls.BLUR_INDICATORS if indicator in html_lower)
        if blur_count >= 1:
            return PaywallStatus.PREVIEW
        
        # Check content length - very short articles might be truncated
        parser = HTMLParser(html)
        text_content = parser.text(deep=True, separator=" ", strip=True)
        word_count = len(text_content.split())
        
        # If article is suspiciously short and has subscription mentions
        if word_count < 100 and any(term in html_lower for term in ["subscribe", "premium", "member"]):
            return PaywallStatus.PREVIEW
        
        return PaywallStatus.FREE


class MetadataExtractor:
    """Extracts metadata from HTML"""
    
    @staticmethod
    def extract_title(parser: HTMLParser) -> str:
        """Extract article title"""
        # Try multiple selectors in order of preference
        selectors = [
            'h1[class*="headline"]',
            'h1[class*="title"]',
            'h1[class*="article"]',
            '.article-title',
            '.headline',
            '.entry-title',
            '[property="og:title"]',
            '[name="twitter:title"]',
            'h1',
            'title'
        ]
        
        for selector in selectors:
            elements = parser.css(selector)
            if elements:
                title = elements[0].text(deep=True, strip=True)
                if title and len(title) > 5:
                    return title
        
        return "Untitled Article"
    
    @staticmethod
    def extract_authors(parser: HTMLParser) -> List[str]:
        """Extract article authors"""
        authors = []
        
        # Try various author selectors
        author_selectors = [
            '[rel="author"]',
            '.author',
            '.byline',
            '.article-author',
            '.writer',
            '[class*="author"]',
            '[property="article:author"]',
            '[name="author"]'
        ]
        
        for selector in author_selectors:
            elements = parser.css(selector)
            for element in elements:
                author_text = element.text(deep=True, strip=True)
                if author_text and author_text not in authors:
                    # Clean up author text
                    author_text = re.sub(r'^(by\s+|author:\s*)', '', author_text, flags=re.IGNORECASE)
                    if len(author_text) < 100:  # Reasonable author name length
                        authors.append(author_text)
        
        return authors[:5]  # Limit to 5 authors max
    
    @staticmethod
    def extract_dates(parser: HTMLParser) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Extract published and updated dates"""
        published_at = None
        updated_at = None
        
        # Date selectors
        date_selectors = [
            '[property="article:published_time"]',
            '[name="publish_date"]',
            '[name="publication_date"]',
            '[class*="publish"]',
            '[class*="date"]',
            'time[datetime]',
            'time[pubdate]'
        ]
        
        for selector in date_selectors:
            elements = parser.css(selector)
            for element in elements:
                # Try datetime attribute first
                date_str = element.attributes.get('datetime') or element.attributes.get('content')
                if not date_str:
                    date_str = element.text(deep=True, strip=True)
                
                if date_str:
                    parsed_date = MetadataExtractor._parse_date(date_str)
                    if parsed_date:
                        if not published_at:
                            published_at = parsed_date
                        else:
                            updated_at = parsed_date
                        break
        
        # Try updated date specific selectors
        update_selectors = [
            '[property="article:modified_time"]',
            '[name="last_modified"]',
            '[class*="updated"]',
            '[class*="modified"]'
        ]
        
        for selector in update_selectors:
            elements = parser.css(selector)
            for element in elements:
                date_str = element.attributes.get('datetime') or element.attributes.get('content')
                if not date_str:
                    date_str = element.text(deep=True, strip=True)
                
                if date_str:
                    parsed_date = MetadataExtractor._parse_date(date_str)
                    if parsed_date:
                        updated_at = parsed_date
                        break
        
        return published_at, updated_at
    
    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """Parse date string into datetime object"""
        if not date_str:
            return None
        
        # Common date formats
        formats = [
            '%Y-%m-%dT%H:%M:%S%z',      # ISO 8601 with timezone
            '%Y-%m-%dT%H:%M:%SZ',       # ISO 8601 UTC
            '%Y-%m-%dT%H:%M:%S',        # ISO 8601 without timezone
            '%Y-%m-%d %H:%M:%S',        # SQL datetime
            '%Y-%m-%d',                 # Just date
            '%B %d, %Y',                # Month Day, Year
            '%b %d, %Y',                # Mon Day, Year
            '%d %B %Y',                 # Day Month Year
            '%d %b %Y',                 # Day Mon Year
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                # If no timezone info, assume UTC
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        
        return None
    
    @staticmethod
    def extract_source(parser: HTMLParser, url: str) -> str:
        """Extract source/publication name"""
        # Try meta tags first
        source_selectors = [
            '[property="og:site_name"]',
            '[name="application-name"]',
            '[name="site_name"]',
            '.site-name',
            '.publication-name',
            '.brand-name'
        ]
        
        for selector in source_selectors:
            elements = parser.css(selector)
            if elements:
                source = elements[0].attributes.get('content') or elements[0].text(deep=True, strip=True)
                if source:
                    return source
        
        # Fallback to domain name
        domain = urlparse(url).netloc
        return domain.replace('www.', '')


class ContentExtractor:
    """Extracts main article content from HTML"""
    
    # Content selectors in order of preference
    CONTENT_SELECTORS = [
        'article',
        '[role="main"]',
        '.article-content',
        '.entry-content',
        '.post-content',
        '.main-content',
        '.story-body',
        '.article-body',
        '.content-body',
        '[class*="content"]',
        '[class*="article"]',
        '[class*="story"]',
        'main'
    ]
    
    # Elements to remove (boilerplate)
    NOISE_SELECTORS = [
        'nav', 'header', 'footer', 'aside',
        '.navigation', '.sidebar', '.comments',
        '.social-share', '.advertisement', '.ad',
        '.related-articles', '.more-stories',
        '.newsletter-signup', '.subscription',
        '.tags', '.breadcrumb', '.pagination',
        'script', 'style', 'noscript'
    ]
    
    @classmethod
    def extract_content(cls, parser: HTMLParser) -> Tuple[str, List[str]]:
        """Extract main article text content"""
        selectors_used = []
        
        # Remove noise elements
        for selector in cls.NOISE_SELECTORS:
            for element in parser.css(selector):
                element.decompose()
        
        # Try content selectors
        content_text = ""
        for selector in cls.CONTENT_SELECTORS:
            elements = parser.css(selector)
            if elements:
                # Use the first matching element
                content_element = elements[0]
                content_text = content_element.text(deep=True, separator=" ", strip=True)
                
                if content_text and len(content_text.split()) >= 50:  # Minimum content threshold
                    selectors_used.append(selector)
                    break
        
        # If no good content found, try the largest text block
        if not content_text or len(content_text.split()) < 50:
            content_text = cls._extract_largest_text_block(parser)
            if content_text:
                selectors_used.append("largest_text_block")
        
        # Clean up the content
        content_text = cls._clean_content(content_text)
        
        return content_text, selectors_used
    
    @classmethod
    def _extract_largest_text_block(cls, parser: HTMLParser) -> str:
        """Extract the largest text block as fallback"""
        candidates = []
        
        # Get all paragraph-like elements
        for tag in ['p', 'div', 'section']:
            elements = parser.css(tag)
            for element in elements:
                text = element.text(deep=True, separator=" ", strip=True)
                if text and len(text.split()) >= 20:
                    candidates.append(text)
        
        # Return the longest text block
        if candidates:
            return max(candidates, key=lambda x: len(x.split()))
        
        return ""
    
    @classmethod
    def _clean_content(cls, content: str) -> str:
        """Clean up extracted content"""
        if not content:
            return ""
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common boilerplate patterns
        patterns_to_remove = [
            r'share on facebook',
            r'share on twitter',
            r'subscribe to.*newsletter',
            r'follow us on',
            r'©\s*\d{4}.*',  # Copyright notices
            r'all rights reserved',
            r'terms of service',
            r'privacy policy',
        ]
        
        for pattern in patterns_to_remove:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        return content.strip()


class FinancialContentExtractor:
    """
    Main content extraction class that handles financial article extraction
    with paywall detection and metadata extraction.
    """
    
    def __init__(self):
        self.paywall_detector = PaywallDetector()
        self.metadata_extractor = MetadataExtractor()
        self.content_extractor = ContentExtractor()
    
    def extract_article(self, render_result: RenderResult) -> ExtractedArticle:
        """
        Extract article content and metadata from rendered HTML.
        
        Args:
            render_result: Result from browser rendering
            
        Returns:
            ExtractedArticle with content and metadata
        """
        start_time = time.time()
        
        if not render_result.success or not render_result.html:
            return self._create_failed_article(
                render_result.url, 
                render_result.final_url,
                "No HTML content available"
            )
        
        try:
            # Parse HTML
            parser = HTMLParser(render_result.html)
            
            # Detect paywall
            paywall_status = self.paywall_detector.detect_paywall(
                render_result.html, 
                render_result.final_url
            )
            
            # If paywalled, return minimal metadata only
            if paywall_status == PaywallStatus.PAYWALLED:
                return self._create_paywalled_article(parser, render_result)
            
            # Extract metadata
            title = self.metadata_extractor.extract_title(parser)
            authors = self.metadata_extractor.extract_authors(parser)
            published_at, updated_at = self.metadata_extractor.extract_dates(parser)
            source = self.metadata_extractor.extract_source(parser, render_result.final_url)
            
            # Extract content
            content_text, selectors_used = self.content_extractor.extract_content(parser)
            word_count = len(content_text.split()) if content_text else 0
            
            # Get canonical URL
            canonical_url = self._extract_canonical_url(parser) or render_result.final_url
            
            extraction_time = int((time.time() - start_time) * 1000)
            
            return ExtractedArticle(
                url=render_result.url,
                canonical_url=canonical_url,
                title=title,
                authors=authors,
                published_at=published_at,
                updated_at=updated_at,
                source=source,
                text=content_text,
                word_count=word_count,
                paywall_status=paywall_status,
                selectors_used=selectors_used,
                extraction_time_ms=extraction_time,
                metadata={
                    'original_status_code': render_result.status_code,
                    'fetch_strategy': render_result.strategy,
                    'render_time_ms': render_result.render_time_ms
                }
            )
            
        except Exception as e:
            logger.error("Content extraction failed", url=render_result.url, error=str(e))
            return self._create_failed_article(
                render_result.url,
                render_result.final_url,
                f"Extraction failed: {str(e)}"
            )
    
    def _create_failed_article(self, url: str, final_url: str, error: str) -> ExtractedArticle:
        """Create a failed extraction result"""
        return ExtractedArticle(
            url=url,
            canonical_url=final_url,
            title="Extraction Failed",
            text="",
            word_count=0,
            source=urlparse(url).netloc,
            paywall_status=PaywallStatus.UNKNOWN,
            metadata={'error': error}
        )
    
    def _create_paywalled_article(self, parser: HTMLParser, render_result: RenderResult) -> ExtractedArticle:
        """Create article result for paywalled content"""
        title = self.metadata_extractor.extract_title(parser)
        authors = self.metadata_extractor.extract_authors(parser)
        published_at, updated_at = self.metadata_extractor.extract_dates(parser)
        source = self.metadata_extractor.extract_source(parser, render_result.final_url)
        
        return ExtractedArticle(
            url=render_result.url,
            canonical_url=render_result.final_url,
            title=title,
            authors=authors,
            published_at=published_at,
            updated_at=updated_at,
            source=source,
            text="[PAYWALLED CONTENT]",
            word_count=0,
            paywall_status=PaywallStatus.PAYWALLED,
            metadata={'paywall_detected': True}
        )
    
    def _extract_canonical_url(self, parser: HTMLParser) -> Optional[str]:
        """Extract canonical URL from meta tags"""
        canonical_selectors = [
            '[rel="canonical"]',
            '[property="og:url"]'
        ]
        
        for selector in canonical_selectors:
            elements = parser.css(selector)
            if elements:
                canonical = elements[0].attributes.get('href') or elements[0].attributes.get('content')
                if canonical:
                    return canonical
        
        return None