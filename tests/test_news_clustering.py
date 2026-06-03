#!/usr/bin/env python3
"""
Unit tests for news clustering and consensus scoring
"""
import unittest
import tempfile
import shutil
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Set up test environment
import os
os.environ['ENABLE_NEWS_CORROBORATION'] = 'true'

from services.news_clustering import NewsClusteringService, NewsConsensusEnhancer

class TestNewsClusteringService(unittest.TestCase):
    """Test cases for NewsClusteringService"""
    
    def setUp(self):
        """Set up test environment with synthetic headlines"""
        self.temp_dir = tempfile.mkdtemp()
        self.service = NewsClusteringService()
        
        # Mock the data path
        self.service.data_path = Path(self.temp_dir)
        self.service.clusters_path = self.service.data_path / "news_clusters.csv"
        
        # Create synthetic headlines for testing
        base_time = datetime.now()
        self.test_headlines = [
            # Cluster 1: Tesla earnings (similar stories within 6 hours)
            ("Tesla Reports Record Q3 Earnings Beat", "https://reuters.com/tesla-earnings-1", base_time),
            ("TSLA Beats Q3 Earnings Expectations", "https://bloomberg.com/tesla-q3-beat", base_time + timedelta(hours=2)),
            ("Tesla Q3 Results Exceed Wall Street Forecasts", "https://wsj.com/tesla-results", base_time + timedelta(hours=4)),
            
            # Cluster 2: Apple iPhone news (similar stories)
            ("Apple Unveils New iPhone 15 Models", "https://cnbc.com/apple-iphone-15", base_time + timedelta(hours=1)),
            ("Apple Launches Latest iPhone Series", "https://marketwatch.com/apple-iphone-new", base_time + timedelta(hours=3)),
            
            # Standalone: Different story, outside time window
            ("Tesla Stock Price Rises After Analyst Upgrade", "https://yahoo.com/tesla-upgrade", base_time + timedelta(hours=8)),
            
            # Cluster 3: Single source (low confidence)
            ("Tesla Considering New Factory Location", "https://unknown-blog.com/tesla-factory", base_time + timedelta(hours=1)),
            
            # Similar to first cluster but outside time window (should be separate)
            ("Tesla Q3 Earnings Analysis and Market Impact", "https://reuters.com/tesla-analysis", base_time + timedelta(hours=7))
        ]
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_cluster_creation_within_time_window(self):
        """Test that similar headlines within 6 hours are clustered together"""
        headlines_data = {
            'TSLA': {
                'headlines': self.test_headlines,
                'count': len(self.test_headlines)
            }
        }
        
        enhanced_data = self.service.cluster_headlines(headlines_data)
        clusters = enhanced_data['TSLA']['clusters']
        
        # Should have multiple clusters
        self.assertGreater(len(clusters), 1)
        
        # Find the Tesla earnings cluster (should have 3 members)
        earnings_cluster = None
        for cluster in clusters:
            if cluster['member_count'] >= 3:
                earnings_cluster = cluster
                break
        
        self.assertIsNotNone(earnings_cluster)
        self.assertEqual(earnings_cluster['member_count'], 3)
        self.assertEqual(earnings_cluster['symbol'], 'TSLA')
    
    def test_similarity_threshold_enforcement(self):
        """Test that only sufficiently similar headlines are clustered"""
        dissimilar_headlines = [
            ("Tesla Reports Strong Earnings", "https://source1.com/1", datetime.now()),
            ("Apple Launches New Product", "https://source2.com/2", datetime.now() + timedelta(hours=1)),
            ("Market Shows Volatility Today", "https://source3.com/3", datetime.now() + timedelta(hours=2))
        ]
        
        headlines_data = {
            'MIXED': {
                'headlines': dissimilar_headlines,
                'count': len(dissimilar_headlines)
            }
        }
        
        enhanced_data = self.service.cluster_headlines(headlines_data)
        clusters = enhanced_data['MIXED']['clusters']
        
        # Each dissimilar headline should form its own cluster
        self.assertEqual(len(clusters), 3)
        for cluster in clusters:
            self.assertEqual(cluster['member_count'], 1)
    
    def test_time_window_enforcement(self):
        """Test that headlines outside 6-hour window don't cluster"""
        base_time = datetime.now()
        time_separated_headlines = [
            ("Tesla Earnings Beat Expected", "https://source1.com/1", base_time),
            ("Tesla Earnings Results Strong", "https://source2.com/2", base_time + timedelta(hours=7))  # Outside window
        ]
        
        headlines_data = {
            'TSLA': {
                'headlines': time_separated_headlines,
                'count': len(time_separated_headlines)
            }
        }
        
        enhanced_data = self.service.cluster_headlines(headlines_data)
        clusters = enhanced_data['TSLA']['clusters']
        
        # Should form separate clusters due to time separation
        self.assertEqual(len(clusters), 2)
        for cluster in clusters:
            self.assertEqual(cluster['member_count'], 1)
    
    def test_confidence_scoring(self):
        """Test confidence scoring based on source diversity and reputation"""
        # High confidence: 3+ sources with primary source
        high_conf_headlines = [
            ("Tesla Strong Results", "https://reuters.com/1", datetime.now()),
            ("Tesla Beat Estimates", "https://bloomberg.com/2", datetime.now() + timedelta(hours=1)),
            ("Tesla Earnings Success", "https://wsj.com/3", datetime.now() + timedelta(hours=2))
        ]
        
        headlines_data = {
            'TSLA': {
                'headlines': high_conf_headlines,
                'count': len(high_conf_headlines)
            }
        }
        
        enhanced_data = self.service.cluster_headlines(headlines_data)
        cluster = enhanced_data['TSLA']['clusters'][0]
        
        # Should have high confidence (≥0.8)
        self.assertGreaterEqual(cluster['confidence'], 0.8)
        self.assertTrue(cluster['has_primary_source'])
    
    def test_low_confidence_single_source(self):
        """Test low confidence for single source stories"""
        single_source_headlines = [
            ("Tesla Rumored New Product", "https://unknown-blog.com/rumor", datetime.now())
        ]
        
        headlines_data = {
            'TSLA': {
                'headlines': single_source_headlines,
                'count': len(single_source_headlines)
            }
        }
        
        enhanced_data = self.service.cluster_headlines(headlines_data)
        cluster = enhanced_data['TSLA']['clusters'][0]
        
        # Should have low confidence
        self.assertLessEqual(cluster['confidence'], 0.5)
        self.assertFalse(cluster['has_primary_source'])
    
    def test_consensus_sentiment_calculation(self):
        """Test weighted consensus sentiment calculation"""
        mixed_sentiment_headlines = [
            ("Tesla Beats Earnings Expectations", "https://reuters.com/1", datetime.now()),  # Positive
            ("Tesla Earnings Strong Performance", "https://bloomberg.com/2", datetime.now() + timedelta(hours=1)),  # Positive
            ("Tesla Results Mixed on Revenue", "https://unknown.com/3", datetime.now() + timedelta(hours=2))  # Neutral/Negative
        ]
        
        headlines_data = {
            'TSLA': {
                'headlines': mixed_sentiment_headlines,
                'count': len(mixed_sentiment_headlines)
            }
        }
        
        enhanced_data = self.service.cluster_headlines(headlines_data)
        cluster = enhanced_data['TSLA']['clusters'][0]
        
        # Should have positive consensus due to high-reputation positive sources
        self.assertGreater(cluster['consensus'], 0.0)
    
    def test_cluster_storage(self):
        """Test that clusters are stored persistently"""
        headlines_data = {
            'AAPL': {
                'headlines': [
                    ("Apple Strong Quarter", "https://reuters.com/apple", datetime.now()),
                    ("Apple Reports Growth", "https://bloomberg.com/apple", datetime.now() + timedelta(hours=1))
                ],
                'count': 2
            }
        }
        
        self.service.cluster_headlines(headlines_data)
        
        # Check that clusters file was created
        self.assertTrue(self.service.clusters_path.exists())
        
        # Check file contents
        clusters_df = pd.read_csv(self.service.clusters_path)
        self.assertFalse(clusters_df.empty)
        self.assertIn('AAPL', clusters_df['symbol'].values)
    
    def test_source_extraction_and_reputation(self):
        """Test source extraction from URLs and reputation mapping"""
        # Test known source
        source = self.service._extract_source_from_url("https://www.reuters.com/business/tesla-earnings")
        self.assertEqual(source, "reuters")
        
        # Test unknown source
        source = self.service._extract_source_from_url("https://unknown-site.com/news")
        self.assertEqual(source, "unknown-site")
        
        # Test reputation scoring
        self.assertEqual(self.service.source_reputation.get('reuters'), 0.9)
        self.assertEqual(self.service.source_reputation.get('unknown', 0.5), 0.5)

class TestNewsConsensusEnhancer(unittest.TestCase):
    """Test cases for NewsConsensusEnhancer integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.enhancer = NewsConsensusEnhancer()
    
    @patch('config.feature_flags.is_news_corroboration_enabled')
    def test_feature_flag_disabled(self, mock_flag):
        """Test that enhancement is skipped when feature flag is disabled"""
        mock_flag.return_value = False
        
        original_data = {
            'TSLA': {
                'headlines': [("Test headline", "https://example.com", datetime.now())],
                'count': 1
            }
        }
        
        enhanced_data = self.enhancer.enhance_news_with_consensus(original_data)
        
        # Should return original data unchanged
        self.assertEqual(enhanced_data, original_data)
        self.assertNotIn('clusters', enhanced_data['TSLA'])
    
    @patch('config.feature_flags.is_news_corroboration_enabled')
    def test_feature_flag_enabled(self, mock_flag):
        """Test that enhancement runs when feature flag is enabled"""
        mock_flag.return_value = True
        
        original_data = {
            'TSLA': {
                'headlines': [
                    ("Tesla Earnings Beat", "https://reuters.com/1", datetime.now()),
                    ("Tesla Strong Results", "https://bloomberg.com/2", datetime.now() + timedelta(hours=1))
                ],
                'count': 2
            }
        }
        
        with patch.object(self.enhancer.clustering_service, 'cluster_headlines') as mock_cluster:
            mock_cluster.return_value = original_data  # Mock return
            
            enhanced_data = self.enhancer.enhance_news_with_consensus(original_data)
            
            # Should have called clustering service
            mock_cluster.assert_called_once_with(original_data)

class TestRapidFuzzIntegration(unittest.TestCase):
    """Test rapidfuzz token_set_ratio functionality"""
    
    def test_similarity_detection(self):
        """Test that rapidfuzz correctly identifies similar headlines"""
        from rapidfuzz import fuzz
        
        title1 = "Tesla Reports Record Q3 Earnings Beat"
        title2 = "TSLA Beats Q3 Earnings Expectations"
        title3 = "Apple Launches New iPhone Models"
        
        # Similar titles should have high similarity
        similarity_similar = fuzz.token_set_ratio(title1, title2)
        self.assertGreater(similarity_similar, 70)
        
        # Dissimilar titles should have low similarity
        similarity_different = fuzz.token_set_ratio(title1, title3)
        self.assertLess(similarity_different, 50)
    
    def test_edge_cases(self):
        """Test edge cases for similarity detection"""
        from rapidfuzz import fuzz
        
        # Identical titles
        title1 = "Tesla Earnings Beat"
        title2 = "Tesla Earnings Beat"
        self.assertEqual(fuzz.token_set_ratio(title1, title2), 100)
        
        # Empty titles
        self.assertEqual(fuzz.token_set_ratio("", ""), 0)
        
        # Very different lengths
        short = "Tesla up"
        long = "Tesla stock price rises significantly after strong quarterly earnings report exceeds analyst expectations"
        similarity = fuzz.token_set_ratio(short, long)
        self.assertGreater(similarity, 0)  # Should still find some similarity

if __name__ == '__main__':
    # Create test suite with synthetic fixtures
    unittest.main()