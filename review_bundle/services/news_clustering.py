#!/usr/bin/env python3
"""
News clustering service for identifying duplicate stories and building consensus
Uses rapidfuzz for similarity detection and implements confidence scoring
"""
import pandas as pd
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from rapidfuzz import fuzz
from pathlib import Path
from services.audit_logger import log_news_processing

logger = logging.getLogger(__name__)

class NewsClusteringService:
    """Service for clustering similar news stories and calculating consensus"""
    
    def __init__(self):
        self.data_path = Path("data")
        self.clusters_path = self.data_path / "news_clusters.csv"
        self.similarity_threshold = 75  # rapidfuzz token_set_ratio threshold
        self.time_window_hours = 6
        
        # Source reputation mapping (can be expanded)
        self.source_reputation = {
            'reuters': 0.9,
            'bloomberg': 0.9,
            'wsj': 0.85,
            'cnbc': 0.8,
            'marketwatch': 0.75,
            'yahoo': 0.7,
            'google': 0.6,
            'unknown': 0.5
        }
        
        # Primary sources for corroboration
        self.primary_sources = {'reuters', 'bloomberg', 'wsj', 'ap', 'dow_jones'}
    
    def cluster_headlines(self, headlines_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Cluster headlines within time windows and calculate consensus scores
        Returns enhanced headlines_data with clustering information
        """
        with log_news_processing("clustering", metadata={'total_tickers': len(headlines_data)}) as op:
            try:
                enhanced_data = {}
                total_headlines_in = sum(len(data.get('headlines', [])) for data in headlines_data.values())
                op.step("start_clustering", count_in=total_headlines_in)
                
                for ticker, data in headlines_data.items():
                    headlines = data.get('headlines', [])
                    if not headlines:
                        enhanced_data[ticker] = data
                        continue
                    
                    with log_news_processing("clustering_ticker", ticker, {'ticker': ticker, 'headline_count': len(headlines)}) as ticker_op:
                        # Create clusters for this ticker
                        clusters = self._create_clusters(headlines, ticker)
                        ticker_op.step("clusters_created", count_out=len(clusters), 
                                     metadata={'similarity_threshold': self.similarity_threshold})
                        
                        # Calculate consensus scores for each cluster
                        enhanced_headlines = []
                        for headline in headlines:
                            enhanced_headline = self._enhance_headline_with_cluster_info(
                                headline, clusters, ticker
                            )
                            enhanced_headlines.append(enhanced_headline)
                        
                        ticker_op.step("headlines_enhanced", count_out=len(enhanced_headlines))
                        
                        # Update data with enhanced information
                        enhanced_data[ticker] = data.copy()
                        enhanced_data[ticker]['headlines'] = enhanced_headlines
                        enhanced_data[ticker]['clusters'] = clusters
                        
                        # Store clusters persistently
                        self._store_clusters(clusters, ticker)
                        ticker_op.step("clusters_stored", count_out=len(clusters), 
                                     source_links=[str(self.clusters_path)])
                
                total_headlines_out = sum(len(data.get('headlines', [])) for data in enhanced_data.values())
                total_clusters = sum(len(data.get('clusters', [])) for data in enhanced_data.values())
                op.step("clustering_complete", count_out=total_headlines_out, 
                       metadata={'total_clusters': total_clusters})
                
                return enhanced_data
                
            except Exception as e:
                logger.exception(f"Error clustering headlines: {e}")
                return headlines_data  # Return original data on error
    
    def _create_clusters(self, headlines: List[Tuple], ticker: str) -> List[Dict[str, Any]]:
        """Create clusters of similar headlines within time windows"""
        clusters = []
        processed_indices = set()
        
        for i, (title_i, url_i, date_i) in enumerate(headlines):
            if i in processed_indices:
                continue
            
            # Start new cluster with this headline
            cluster_members = [(i, title_i, url_i, date_i)]
            processed_indices.add(i)
            
            # Find similar headlines within time window
            for j, (title_j, url_j, date_j) in enumerate(headlines):
                if j in processed_indices or i == j:
                    continue
                
                # Check time window (6 hours)
                if self._within_time_window(date_i, date_j):
                    # Check similarity
                    similarity = fuzz.token_set_ratio(title_i, title_j)
                    if similarity >= self.similarity_threshold:
                        cluster_members.append((j, title_j, url_j, date_j))
                        processed_indices.add(j)
            
            # Create cluster if it has members
            if cluster_members:
                cluster = self._create_cluster_record(cluster_members, ticker)
                clusters.append(cluster)
        
        return clusters
    
    def _create_cluster_record(self, members: List[Tuple], ticker: str) -> Dict[str, Any]:
        """Create a cluster record from member headlines"""
        # Generate cluster ID
        member_titles = [title for _, title, _, _ in members]
        cluster_id = self._generate_cluster_id(ticker, member_titles)
        
        # Extract dates and sources
        dates = [date for _, _, _, date in members]
        urls = [url for _, _, url, _ in members]
        sources = [self._extract_source_from_url(url) for url in urls]
        
        # Calculate time span
        start_time = min(dates)
        end_time = max(dates)
        
        # Calculate consensus sentiment (placeholder - would integrate with sentiment analysis)
        consensus_sentiment = self._calculate_consensus_sentiment(members, sources)
        
        # Calculate confidence based on source diversity and count
        confidence = self._calculate_confidence(sources, len(members))
        
        cluster = {
            'cluster_id': cluster_id,
            'symbol': ticker,
            'start_ts': start_time.isoformat() if hasattr(start_time, 'isoformat') else str(start_time),
            'end_ts': end_time.isoformat() if hasattr(end_time, 'isoformat') else str(end_time),
            'members': json.dumps([{
                'title': title,
                'url': url,
                'date': date.isoformat() if hasattr(date, 'isoformat') else str(date),
                'source': self._extract_source_from_url(url)
            } for _, title, url, date in members]),
            'consensus': consensus_sentiment,
            'confidence': confidence,
            'member_count': len(members),
            'sources': list(set(sources)),
            'has_primary_source': any(source in self.primary_sources for source in sources)
        }
        
        return cluster
    
    def _calculate_consensus_sentiment(self, members: List[Tuple], sources: List[str]) -> float:
        """Calculate weighted consensus sentiment across cluster members"""
        if not members:
            return 0.0
        
        # For now, use simple sentiment based on title keywords
        # In practice, this would integrate with the FinBERT pipeline
        total_weighted_sentiment = 0.0
        total_weight = 0.0
        
        for (_, title, _, _), source in zip(members, sources):
            sentiment = self._basic_sentiment_score(title)
            weight = self.source_reputation.get(source, 0.5)
            
            total_weighted_sentiment += sentiment * weight
            total_weight += weight
        
        return total_weighted_sentiment / total_weight if total_weight > 0 else 0.0
    
    def _basic_sentiment_score(self, title: str) -> float:
        """Basic sentiment scoring (placeholder for FinBERT integration)"""
        positive_words = ['gain', 'rise', 'up', 'bull', 'buy', 'strong', 'growth', 'profit', 'beat', 'outperform']
        negative_words = ['loss', 'fall', 'down', 'bear', 'sell', 'weak', 'decline', 'risk', 'miss', 'underperform']
        
        title_lower = title.lower()
        pos_score = sum(2 if word in title_lower else 0 for word in positive_words)
        neg_score = sum(2 if word in title_lower else 0 for word in negative_words)
        
        # Normalize to [-1, 1] range
        total_words = len(title.split())
        return (pos_score - neg_score) / max(total_words, 1)
    
    def _calculate_confidence(self, sources: List[str], member_count: int) -> float:
        """
        Calculate confidence based on source diversity and corroboration
        High: ≥3 sources with primary corroboration
        Medium: Multiple sources without primary
        Low: Single source or rumor wording
        """
        unique_sources = set(sources)
        has_primary = any(source in self.primary_sources for source in unique_sources)
        
        if len(unique_sources) >= 3 and has_primary:
            confidence = 0.9  # High confidence
        elif len(unique_sources) >= 2:
            confidence = 0.7 if has_primary else 0.6  # Medium confidence
        else:
            confidence = 0.4  # Low confidence
        
        # Adjust based on member count
        if member_count >= 5:
            confidence = min(0.95, confidence + 0.1)
        elif member_count >= 3:
            confidence = min(0.9, confidence + 0.05)
        
        return round(confidence, 2)
    
    def _generate_cluster_id(self, ticker: str, titles: List[str]) -> str:
        """Generate unique cluster ID based on ticker and representative titles"""
        combined_text = f"{ticker}_{titles[0][:50]}_{len(titles)}"
        return hashlib.md5(combined_text.encode()).hexdigest()[:12]
    
    def _within_time_window(self, date1, date2) -> bool:
        """Check if two dates are within the clustering time window"""
        try:
            # Convert to datetime if they aren't already
            if isinstance(date1, str):
                date1 = pd.to_datetime(date1)
            if isinstance(date2, str):
                date2 = pd.to_datetime(date2)
            
            time_diff = abs((date1 - date2).total_seconds()) / 3600  # Convert to hours
            return time_diff <= self.time_window_hours
            
        except Exception as e:
            logger.warning(f"Error comparing dates {date1} and {date2}: {e}")
            return False
    
    def _extract_source_from_url(self, url: str) -> str:
        """Extract source name from URL"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            domain = domain.replace('www.', '')
            
            # Map known domains to standardized source names
            domain_map = {
                'reuters.com': 'reuters',
                'bloomberg.com': 'bloomberg',
                'wsj.com': 'wsj',
                'cnbc.com': 'cnbc',
                'marketwatch.com': 'marketwatch',
                'finance.yahoo.com': 'yahoo',
                'news.google.com': 'google'
            }
            
            return domain_map.get(domain, domain.split('.')[0] if domain else 'unknown')
            
        except Exception:
            return 'unknown'
    
    def _enhance_headline_with_cluster_info(self, headline: Tuple, clusters: List[Dict], ticker: str) -> Dict[str, Any]:
        """Enhance individual headline with cluster information"""
        title, url, date = headline
        
        # Find which cluster this headline belongs to
        cluster_info = None
        for cluster in clusters:
            members = json.loads(cluster['members'])
            if any(member['url'] == url for member in members):
                cluster_info = {
                    'cluster_id': cluster['cluster_id'],
                    'consensus': cluster['consensus'],
                    'confidence': cluster['confidence'],
                    'member_count': cluster['member_count'],
                    'has_primary_source': cluster['has_primary_source']
                }
                break
        
        enhanced_headline = {
            'title': title,
            'url': url,
            'date': date.isoformat() if hasattr(date, 'isoformat') else str(date),
            'source': self._extract_source_from_url(url),
            'cluster_info': cluster_info
        }
        
        return enhanced_headline
    
    def _store_clusters(self, clusters: List[Dict], ticker: str):
        """Store clusters persistently in news_clusters.csv"""
        try:
            if not clusters:
                return
            
            # Ensure clusters file exists
            if not self.clusters_path.exists():
                self.data_path.mkdir(exist_ok=True)
                empty_df = pd.DataFrame(columns=[
                    'cluster_id', 'symbol', 'start_ts', 'end_ts', 
                    'members', 'consensus', 'confidence'
                ])
                empty_df.to_csv(self.clusters_path, index=False)
            
            # Load existing clusters
            existing_df = pd.read_csv(self.clusters_path)
            
            # Create new clusters DataFrame
            new_clusters = []
            for cluster in clusters:
                new_clusters.append({
                    'cluster_id': cluster['cluster_id'],
                    'symbol': cluster['symbol'],
                    'start_ts': cluster['start_ts'],
                    'end_ts': cluster['end_ts'],
                    'members': cluster['members'],
                    'consensus': cluster['consensus'],
                    'confidence': cluster['confidence']
                })
            
            new_df = pd.DataFrame(new_clusters)
            
            # Remove existing clusters for this ticker to avoid duplicates
            existing_df = existing_df[existing_df['symbol'] != ticker]
            
            # Combine and save
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(self.clusters_path, index=False)
            
            logger.info(f"Stored {len(clusters)} clusters for {ticker}")
            
        except Exception as e:
            logger.exception(f"Error storing clusters for {ticker}: {e}")

class NewsConsensusEnhancer:
    """Service for enhancing news sentiment with consensus scoring"""
    
    def __init__(self):
        self.clustering_service = NewsClusteringService()
        
        # Import corroboration service
        try:
            from .news_corroboration import NewsCorroborationService
            self.corroboration_service = NewsCorroborationService()
        except ImportError:
            logger.warning("News corroboration service not available")
            self.corroboration_service = None
    
    def enhance_news_with_consensus(self, headlines_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Enhance news data with clustering and consensus scoring
        Only runs if news corroboration feature flag is enabled
        """
        from config.feature_flags import is_news_corroboration_enabled
        
        if not is_news_corroboration_enabled():
            logger.debug("News corroboration disabled, returning original data")
            return headlines_data
        
        logger.info("Enhancing news with consensus scoring and clustering")
        
        # Step 1: Apply corroboration pass
        enhanced_data = headlines_data
        if self.corroboration_service:
            try:
                enhanced_data = self.corroboration_service.corroborate_headlines(headlines_data)
                logger.debug("Applied corroboration pass to news data")
            except Exception as e:
                logger.warning(f"Error in corroboration pass: {e}")
        
        # Step 2: Apply clustering and consensus scoring
        try:
            enhanced_data = self.clustering_service.cluster_headlines(enhanced_data)
            logger.debug("Applied clustering and consensus scoring")
        except Exception as e:
            logger.warning(f"Error in clustering pass: {e}")
        
        return enhanced_data