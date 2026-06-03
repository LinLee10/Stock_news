"""
Tests for content extraction functionality.
"""

import pytest
from datetime import datetime, timezone

from ..models import RenderResult, FetchStrategy, PaywallStatus
from ..extractor import (
    FinancialContentExtractor, PaywallDetector, MetadataExtractor, ContentExtractor
)


class TestPaywallDetector:
    """Test paywall detection"""
    
    def test_detect_free_content(self):
        """Test detection of free content"""
        html = """
        <html>
        <body>
        <article>
        <h1>Market Update</h1>
        <p>The stock market closed higher today as investors welcomed news...</p>
        <p>This is a full article with substantial content that should be considered free.</p>
        </article>
        </body>
        </html>
        """
        
        result = PaywallDetector.detect_paywall(html, "https://example.com/article")
        assert result == PaywallStatus.FREE
    
    def test_detect_paywall(self):
        """Test detection of paywalled content"""
        html = """
        <html>
        <body>
        <article>
        <h1>Premium Market Analysis</h1>
        <p>The market showed strong performance...</p>
        <div class="paywall">Subscribe to continue reading this exclusive analysis</div>
        <div class="subscription-required">Become a member to unlock this story</div>
        </article>
        </body>
        </html>
        """
        
        result = PaywallDetector.detect_paywall(html, "https://example.com/premium")
        assert result == PaywallStatus.PAYWALLED
    
    def test_detect_preview(self):
        """Test detection of preview content"""
        html = """
        <html>
        <body>
        <article>
        <h1>Market Analysis</h1>
        <p>The market showed strong performance...</p>
        <div class="content-fade">Subscribe to read more</div>
        </article>
        </body>
        </html>
        """
        
        result = PaywallDetector.detect_paywall(html, "https://example.com/preview")
        assert result == PaywallStatus.PREVIEW


class TestMetadataExtractor:
    """Test metadata extraction"""
    
    def test_extract_title(self):
        """Test title extraction"""
        from selectolax.parser import HTMLParser
        
        html = """
        <html>
        <head><title>Page Title</title></head>
        <body>
        <h1 class="headline">Stock Market Rises on Positive News</h1>
        <article>Content here</article>
        </body>
        </html>
        """
        
        parser = HTMLParser(html)
        title = MetadataExtractor.extract_title(parser)
        
        assert title == "Stock Market Rises on Positive News"
    
    def test_extract_authors(self):
        """Test author extraction"""
        from selectolax.parser import HTMLParser
        
        html = """
        <html>
        <body>
        <div class="author">By John Smith</div>
        <div class="byline">Jane Doe</div>
        <article>Content here</article>
        </body>
        </html>
        """
        
        parser = HTMLParser(html)
        authors = MetadataExtractor.extract_authors(parser)
        
        assert "John Smith" in authors
        assert "Jane Doe" in authors
    
    def test_parse_date(self):
        """Test date parsing"""
        # ISO format
        date1 = MetadataExtractor._parse_date("2023-12-01T15:30:00Z")
        assert date1 is not None
        assert date1.year == 2023
        assert date1.month == 12
        assert date1.day == 1
        
        # Human readable format
        date2 = MetadataExtractor._parse_date("December 1, 2023")
        assert date2 is not None
        assert date2.year == 2023
        assert date2.month == 12
        assert date2.day == 1


class TestContentExtractor:
    """Test content extraction"""
    
    def test_extract_content(self):
        """Test main content extraction"""
        from selectolax.parser import HTMLParser
        
        html = """
        <html>
        <body>
        <nav>Navigation</nav>
        <article>
        <h1>Market News</h1>
        <p>The stock market experienced significant volatility today as investors 
        reacted to the latest economic data. Trading volume was higher than average 
        across all major indices.</p>
        <p>Analysts suggest that the current market conditions reflect broader 
        economic uncertainties. The Federal Reserve's recent policy statements 
        have added to market speculation about future interest rate changes.</p>
        </article>
        <aside>Related articles</aside>
        <footer>Footer content</footer>
        </body>
        </html>
        """
        
        parser = HTMLParser(html)
        content, selectors = ContentExtractor.extract_content(parser)
        
        assert "stock market experienced significant volatility" in content
        assert "analysts suggest that the current market conditions" in content
        assert "article" in selectors
        assert len(content.split()) >= 50  # Should meet minimum threshold


class TestFinancialContentExtractor:
    """Test the main extraction class"""
    
    def test_extract_free_article(self):
        """Test extraction of free article"""
        html = """
        <html>
        <head>
        <title>Stock Market Update</title>
        <meta property="og:url" content="https://example.com/canonical">
        <meta name="author" content="Financial Reporter">
        <meta property="article:published_time" content="2023-12-01T10:00:00Z">
        </head>
        <body>
        <article>
        <h1>Market Closes Higher on Earnings Beat</h1>
        <div class="author">By Financial Reporter</div>
        <time datetime="2023-12-01T10:00:00Z">December 1, 2023</time>
        <div class="article-content">
        <p>The stock market closed higher today as several major companies reported 
        earnings that beat analyst expectations. Technology stocks led the gains, 
        with the NASDAQ index rising 2.5% on strong trading volume.</p>
        <p>Investors are showing renewed confidence in the tech sector after months 
        of uncertainty. The Federal Reserve's recent policy decisions have also 
        contributed to the positive market sentiment, with interest rates remaining 
        stable for the near term.</p>
        <p>Market analysts expect this upward trend to continue into the next quarter, 
        citing strong corporate earnings and improving economic indicators. However, 
        geopolitical tensions continue to pose risks to market stability.</p>
        </div>
        </article>
        </body>
        </html>
        """
        
        render_result = RenderResult(
            url="https://example.com/article",
            final_url="https://example.com/article", 
            html=html,
            status_code=200,
            strategy=FetchStrategy.PLAYWRIGHT,
            render_time_ms=1500,
            success=True
        )
        
        extractor = FinancialContentExtractor()
        article = extractor.extract_article(render_result)
        
        assert article.title == "Market Closes Higher on Earnings Beat"
        assert article.canonical_url == "https://example.com/canonical"
        assert "Financial Reporter" in article.authors
        assert article.published_at is not None
        assert article.paywall_status == PaywallStatus.FREE
        assert article.word_count > 50
        assert "stock market closed higher" in article.text
        assert "NASDAQ index rising" in article.text
    
    def test_extract_paywalled_article(self):
        """Test extraction of paywalled article"""
        html = """
        <html>
        <body>
        <article>
        <h1>Premium Analysis: Market Outlook</h1>
        <p>Brief preview of the content...</p>
        <div class="paywall">Subscribe to continue reading</div>
        <div class="subscription-required">Premium content requires subscription</div>
        </article>
        </body>
        </html>
        """
        
        render_result = RenderResult(
            url="https://example.com/premium",
            final_url="https://example.com/premium",
            html=html,
            status_code=200,
            strategy=FetchStrategy.PLAYWRIGHT,
            render_time_ms=1200,
            success=True
        )
        
        extractor = FinancialContentExtractor()
        article = extractor.extract_article(render_result)
        
        assert article.title == "Premium Analysis: Market Outlook"
        assert article.paywall_status == PaywallStatus.PAYWALLED
        assert article.text == "[PAYWALLED CONTENT]"
        assert article.word_count == 0
    
    def test_extract_failed_render(self):
        """Test extraction when render fails"""
        render_result = RenderResult(
            url="https://example.com/failed",
            final_url="https://example.com/failed",
            html="",
            status_code=404,
            strategy=FetchStrategy.PLAYWRIGHT,
            render_time_ms=5000,
            success=False,
            error_message="Page not found"
        )
        
        extractor = FinancialContentExtractor()
        article = extractor.extract_article(render_result)
        
        assert article.title == "Extraction Failed"
        assert article.text == ""
        assert article.word_count == 0
        assert "error" in article.metadata


if __name__ == '__main__':
    pytest.main([__file__])