#!/usr/bin/env python3
"""
Unit tests for news corroboration service
Tests primary source matching and speculation detection
"""
import unittest
from datetime import datetime
from services.news_corroboration import NewsCorroborationService

class TestNewsCorroborationService(unittest.TestCase):
    """Test cases for NewsCorroborationService"""
    
    def setUp(self):
        """Set up test environment"""
        self.service = NewsCorroborationService()
    
    def test_high_confidence_with_corroboration(self):
        """Test that primary source headlines get high confidence with corroboration"""
        # Tesla earnings from official Tesla IR page
        headlines_data = {
            'TSLA': {
                'headlines': [
                    (
                        "Tesla Reports Record Q3 2023 Financial Results",
                        "https://ir.tesla.com/press-release/tesla-q3-2023-earnings",
                        datetime.now()
                    ),
                    (
                        "Tesla Q3 Earnings Beat Street Estimates",
                        "https://reuters.com/business/tesla-earnings-beat",
                        datetime.now()
                    ),
                    (
                        "TSLA Form 8-K Filed with SEC",
                        "https://sec.gov/edgar/data/1318605/000095017023044672/tsla-8k.htm",
                        datetime.now()
                    )
                ]
            }
        }
        
        enhanced_data = self.service.corroborate_headlines(headlines_data)
        headlines = enhanced_data['TSLA']['headlines']
        
        # First headline should have high confidence due to official IR source
        official_headline = headlines[0]
        self.assertIn('official_company', official_headline['flags'])
        self.assertTrue(official_headline['is_corroborated'])
        self.assertGreater(official_headline['confidence_boost'], 0.1)
        self.assertIn('https://ir.tesla.com/press-release/tesla-q3-2023-earnings', 
                     official_headline['corroboration_urls'])
        
        # SEC filing should also be corroborated
        sec_headline = headlines[2]
        self.assertIn('sec_filing', sec_headline['flags'])
        self.assertTrue(sec_headline['is_corroborated'])
        
        # Check aggregate stats
        stats = enhanced_data['TSLA']['corroboration_stats']
        self.assertGreater(stats['corroborated_count'], 1)
        self.assertGreater(stats['overall_confidence'], 0.6)
    
    def test_low_confidence_speculation_no_corroboration(self):
        """Test that speculative headlines without corroboration get low confidence"""
        # Speculative Tesla rumors without corroboration
        headlines_data = {
            'TSLA': {
                'headlines': [
                    (
                        "Tesla Rumored to Be Considering Bitcoin Purchase",
                        "https://crypto-blog.com/tesla-bitcoin-rumor",
                        datetime.now()
                    ),
                    (
                        "Sources Say Tesla May Announce New Model",
                        "https://unknown-news.com/tesla-sources-model",
                        datetime.now()
                    ),
                    (
                        "Tesla Reportedly Planning Factory Expansion",
                        "https://random-site.com/tesla-expansion-leak",
                        datetime.now()
                    )
                ]
            }
        }
        
        enhanced_data = self.service.corroborate_headlines(headlines_data)
        headlines = enhanced_data['TSLA']['headlines']
        
        # All headlines should be marked as speculative
        for headline in headlines:
            self.assertIn('speculative', headline['flags'])
            self.assertFalse(headline['is_corroborated'])
            self.assertLess(headline['confidence_boost'], 0.0)
            self.assertEqual(len(headline['corroboration_urls']), 0)
        
        # Check aggregate stats show low confidence
        stats = enhanced_data['TSLA']['corroboration_stats']
        self.assertEqual(stats['corroborated_count'], 0)
        self.assertEqual(stats['speculative_count'], 3)
        self.assertLess(stats['overall_confidence'], 0.5)
    
    def test_primary_source_detection(self):
        """Test detection of primary sources (SEC, company IR)"""
        # Test SEC domain
        self.assertTrue(self.service._is_primary_source("https://sec.gov/edgar/data/1234567/file.htm", "TSLA"))
        self.assertTrue(self.service._is_primary_source("https://edgar.sec.gov/filing.htm", "AAPL"))
        
        # Test company domains
        self.assertTrue(self.service._is_primary_source("https://tesla.com/news/earnings", "TSLA"))
        self.assertTrue(self.service._is_primary_source("https://apple.com/newsroom/", "AAPL"))
        
        # Test non-primary sources
        self.assertFalse(self.service._is_primary_source("https://cnbc.com/tesla-news", "TSLA"))
        self.assertFalse(self.service._is_primary_source("https://unknown-blog.com", "TSLA"))
    
    def test_company_official_source_detection(self):
        """Test detection of company IR/newsroom pages"""
        # Test IR patterns
        self.assertTrue(self.service._is_company_official_source("https://ir.tesla.com/news", "TSLA"))
        self.assertTrue(self.service._is_company_official_source("https://investor.apple.com/news", "AAPL"))
        self.assertTrue(self.service._is_company_official_source("https://newsroom.tesla.com/releases", "TSLA"))
        
        # Test company domain with IR path
        self.assertTrue(self.service._is_company_official_source("https://tesla.com/investor-relations", "TSLA"))
        self.assertTrue(self.service._is_company_official_source("https://apple.com/newsroom/2023/earnings", "AAPL"))
        
        # Test non-official sources
        self.assertFalse(self.service._is_company_official_source("https://cnbc.com/investing", "TSLA"))
        self.assertFalse(self.service._is_company_official_source("https://tesla.com/models", "TSLA"))
    
    def test_sec_filing_detection(self):
        """Test detection of SEC filing references"""
        # Test SEC URLs
        self.assertTrue(self.service._is_sec_related("https://sec.gov/edgar/data/123/8k.htm", "Any title"))
        self.assertTrue(self.service._is_sec_related("https://edgar.sec.gov/filing", "Any title"))
        
        # Test SEC keywords in titles
        self.assertTrue(self.service._is_sec_related("https://reuters.com/news", "Tesla Files Form 8-K with SEC"))
        self.assertTrue(self.service._is_sec_related("https://bloomberg.com/news", "Apple 10-K Annual Report Filed"))
        self.assertTrue(self.service._is_sec_related("https://cnbc.com/news", "Company Submits SEC Filing"))
        
        # Test non-SEC content
        self.assertFalse(self.service._is_sec_related("https://cnbc.com/news", "Tesla Stock Price Rises"))
        self.assertFalse(self.service._is_sec_related("https://reuters.com/tech", "Apple Launches New iPhone"))
    
    def test_speculation_detection(self):
        """Test detection of speculative and rumor language"""
        # Test rumor keywords
        self.assertTrue(self.service._contains_speculation("Tesla rumored to be acquiring company"))
        self.assertTrue(self.service._contains_speculation("Sources say Apple may announce new product"))
        self.assertTrue(self.service._contains_speculation("Unconfirmed reports suggest merger talks"))
        self.assertTrue(self.service._contains_speculation("Tesla allegedly planning expansion"))
        
        # Test non-speculative language
        self.assertFalse(self.service._contains_speculation("Tesla reports quarterly earnings"))
        self.assertFalse(self.service._contains_speculation("Apple announces new iPhone"))
        self.assertFalse(self.service._contains_speculation("Company completes acquisition"))
    
    def test_mixed_scenario_corroboration(self):
        """Test mixed scenario with both corroborated and speculative headlines"""
        headlines_data = {
            'AAPL': {
                'headlines': [
                    # Official announcement - should have high confidence
                    (
                        "Apple Announces Q4 2023 Financial Results",
                        "https://apple.com/newsroom/2023/10/q4-results",
                        datetime.now()
                    ),
                    # Reuters coverage of same - medium confidence
                    (
                        "Apple Q4 Earnings Beat Analyst Expectations",
                        "https://reuters.com/technology/apple-q4-earnings",
                        datetime.now()
                    ),
                    # Speculative rumor - low confidence
                    (
                        "Apple Rumored to Be Developing AR Glasses",
                        "https://tech-blog.com/apple-ar-rumor",
                        datetime.now()
                    )
                ]
            }
        }
        
        enhanced_data = self.service.corroborate_headlines(headlines_data)
        headlines = enhanced_data['AAPL']['headlines']
        
        # Official announcement should be highly corroborated
        official = headlines[0]
        self.assertIn('official_company', official['flags'])
        self.assertTrue(official['is_corroborated'])
        self.assertGreater(official['confidence_boost'], 0.15)
        
        # Reuters coverage should not be corroborated but also not speculative
        reuters = headlines[1]
        self.assertNotIn('speculative', reuters['flags'])
        self.assertFalse(reuters['is_corroborated'])
        
        # Rumor should be speculative and not corroborated
        rumor = headlines[2]
        self.assertIn('speculative', rumor['flags'])
        self.assertFalse(rumor['is_corroborated'])
        self.assertLess(rumor['confidence_boost'], 0.0)
        
        # Aggregate stats should show mixed confidence
        stats = enhanced_data['AAPL']['corroboration_stats']
        self.assertEqual(stats['corroborated_count'], 1)
        self.assertEqual(stats['speculative_count'], 1)
        self.assertGreater(stats['overall_confidence'], 0.4)
        self.assertLess(stats['overall_confidence'], 0.8)
    
    def test_confidence_boost_calculation(self):
        """Test confidence boost calculations for different source types"""
        # Primary source should give highest boost
        headline_primary = self.service._corroborate_single_headline(
            ("Tesla Reports Earnings", "https://sec.gov/edgar/data/1318605/8k.htm", datetime.now()),
            "TSLA"
        )
        
        # Company official should give medium boost
        headline_official = self.service._corroborate_single_headline(
            ("Tesla Reports Earnings", "https://ir.tesla.com/earnings", datetime.now()),
            "TSLA"
        )
        
        # Speculative should give negative boost
        headline_spec = self.service._corroborate_single_headline(
            ("Tesla rumored to expand", "https://blog.com/tesla-rumor", datetime.now()),
            "TSLA"
        )
        
        # Check confidence boosts
        self.assertGreater(headline_primary['confidence_boost'], headline_official['confidence_boost'])
        self.assertGreater(headline_official['confidence_boost'], 0)
        self.assertLess(headline_spec['confidence_boost'], 0)
    
    def test_corroboration_url_storage(self):
        """Test that corroboration URLs are properly stored"""
        headlines_data = {
            'MSFT': {
                'headlines': [
                    (
                        "Microsoft Files Quarterly Report with SEC",
                        "https://sec.gov/edgar/data/789019/msft-10q.htm",
                        datetime.now()
                    )
                ]
            }
        }
        
        enhanced_data = self.service.corroborate_headlines(headlines_data)
        headline = enhanced_data['MSFT']['headlines'][0]
        
        # Should store the SEC URL as corroboration
        self.assertEqual(len(headline['corroboration_urls']), 1)
        self.assertIn('https://sec.gov/edgar/data/789019/msft-10q.htm', 
                     headline['corroboration_urls'])

if __name__ == '__main__':
    unittest.main()