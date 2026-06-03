#!/usr/bin/env python3
"""
News corroboration service for matching stories to primary sources
Uses heuristics to identify official company communications and filings
"""
import re
import logging
from typing import List, Dict, Any, Tuple, Set
from urllib.parse import urlparse
from config.tickers import TICKER_COMPANY

logger = logging.getLogger(__name__)

class NewsCorroborationService:
    """Service for validating news stories against primary sources"""
    
    def __init__(self):
        # Primary source domains for corroboration
        self.primary_domains = {
            'sec.gov',
            'edgar.sec.gov',
            'investor.sec.gov'
        }
        
        # Company IR and newsroom patterns
        self.company_ir_patterns = [
            r'investor\..*\.com',
            r'ir\..*\.com',
            r'newsroom\..*\.com',
            r'press\..*\.com',
            r'.*\.com/investor',
            r'.*\.com/newsroom',
            r'.*\.com/press',
            r'.*\.com/news'
        ]
        
        # Rumor and speculative language indicators
        self.rumor_keywords = {
            'rumor', 'rumors', 'rumored', 'allegedly', 'sources say', 
            'unconfirmed', 'speculation', 'speculative', 'reportedly',
            'may be', 'could be', 'might', 'possibly', 'potentially',
            'according to sources', 'insider claims', 'leaked'
        }
        
        # Build company domain mapping
        self.company_domains = self._build_company_domain_map()
    
    def _build_company_domain_map(self) -> Dict[str, Set[str]]:
        """Build mapping of tickers to their known official domains"""
        # This would typically be maintained in a config file
        # For now, we'll use common patterns
        domain_map = {}
        
        known_domains = {
            'AAPL': {'apple.com'},
            'MSFT': {'microsoft.com'},
            'GOOGL': {'abc.xyz', 'google.com'},
            'TSLA': {'tesla.com'},
            'NVDA': {'nvidia.com'},
            'META': {'meta.com', 'facebook.com'},
            'AMZN': {'amazon.com'},
            'NFLX': {'netflix.com'},
            'RTX': {'rtx.com'},
            'PFE': {'pfizer.com'},
            'LLY': {'lilly.com'},
            'AMD': {'amd.com'},
            'INTC': {'intel.com'}
        }
        
        for ticker, domains in known_domains.items():
            domain_map[ticker] = domains
        
        return domain_map
    
    def corroborate_headlines(self, headlines_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Add corroboration information to headlines data
        Identifies primary source matches and speculative content
        """
        try:
            corroborated_data = {}
            
            for ticker, data in headlines_data.items():
                headlines = data.get('headlines', [])
                if not headlines:
                    corroborated_data[ticker] = data
                    continue
                
                # Process each headline for corroboration
                enhanced_headlines = []
                for headline in headlines:
                    enhanced_headline = self._corroborate_single_headline(headline, ticker)
                    enhanced_headlines.append(enhanced_headline)
                
                # Update data with corroboration info
                enhanced_data = data.copy()
                enhanced_data['headlines'] = enhanced_headlines
                
                # Add aggregate corroboration metrics
                enhanced_data['corroboration_stats'] = self._calculate_corroboration_stats(enhanced_headlines)
                
                corroborated_data[ticker] = enhanced_data
            
            return corroborated_data
            
        except Exception as e:
            logger.exception(f"Error in news corroboration: {e}")
            return headlines_data  # Return original data on error
    
    def _corroborate_single_headline(self, headline: Any, ticker: str) -> Dict[str, Any]:
        """Corroborate a single headline and add metadata"""
        
        # Handle different headline formats (tuple vs dict)
        if isinstance(headline, tuple):
            title, url, date = headline
            original_headline = {
                'title': title,
                'url': url,
                'date': date
            }
        elif isinstance(headline, dict):
            original_headline = headline.copy()
            title = headline.get('title', '')
            url = headline.get('url', '')
        else:
            logger.warning(f"Unexpected headline format: {type(headline)}")
            return headline
        
        # Initialize corroboration data
        corroboration_urls = []
        flags = []
        confidence_boost = 0.0
        
        # Check for primary source corroboration
        is_primary_source = self._is_primary_source(url, ticker)
        if is_primary_source:
            corroboration_urls.append(url)
            confidence_boost += 0.3
            flags.append("primary_source")
        
        # Check for company IR/newsroom
        is_company_official = self._is_company_official_source(url, ticker)
        if is_company_official:
            if url not in corroboration_urls:
                corroboration_urls.append(url)
            confidence_boost += 0.2
            flags.append("official_company")
        
        # Check for SEC filing references
        is_sec_related = self._is_sec_related(url, title)
        if is_sec_related:
            if url not in corroboration_urls:
                corroboration_urls.append(url)
            confidence_boost += 0.25
            flags.append("sec_filing")
        
        # Check for speculative language
        is_speculative = self._contains_speculation(title)
        if is_speculative and not corroboration_urls:
            flags.append("speculative")
            confidence_boost -= 0.2
        
        # Build enhanced headline
        enhanced_headline = original_headline.copy()
        enhanced_headline['corroboration_urls'] = corroboration_urls
        enhanced_headline['flags'] = flags
        enhanced_headline['confidence_boost'] = confidence_boost
        enhanced_headline['is_corroborated'] = len(corroboration_urls) > 0
        
        return enhanced_headline
    
    def _is_primary_source(self, url: str, ticker: str) -> bool:
        """Check if URL is from a primary source (SEC, company official)"""
        try:
            domain = urlparse(url).netloc.lower()
            domain = domain.replace('www.', '')
            
            # Check SEC domains
            if any(sec_domain in domain for sec_domain in self.primary_domains):
                return True
            
            # Check known company domains
            company_domains = self.company_domains.get(ticker, set())
            if any(comp_domain in domain for comp_domain in company_domains):
                return True
            
            return False
            
        except Exception:
            return False
    
    def _is_company_official_source(self, url: str, ticker: str) -> bool:
        """Check if URL is from company IR, newsroom, or press pages"""
        try:
            full_url = url.lower()
            
            # Check for IR/newsroom patterns in URL path
            for pattern in self.company_ir_patterns:
                if re.search(pattern, full_url):
                    return True
            
            # Check for company-specific patterns
            company_domains = self.company_domains.get(ticker, set())
            for domain in company_domains:
                if domain in full_url and any(keyword in full_url for keyword in 
                    ['investor', 'newsroom', 'press', 'news', 'ir']):
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _is_sec_related(self, url: str, title: str) -> bool:
        """Check if story references SEC filings or regulatory documents"""
        try:
            url_lower = url.lower()
            title_lower = title.lower()
            
            # SEC filing keywords
            sec_keywords = [
                '8-k', '10-k', '10-q', 'form 8-k', 'form 10-k', 'form 10-q',
                'sec filing', 'securities filing', 'regulatory filing',
                'edgar', 'sec.gov'
            ]
            
            # Check URL
            if 'sec.gov' in url_lower or 'edgar' in url_lower:
                return True
            
            # Check title for SEC filing references
            return any(keyword in title_lower for keyword in sec_keywords)
            
        except Exception:
            return False
    
    def _contains_speculation(self, title: str) -> bool:
        """Check if headline contains speculative or rumor language"""
        try:
            title_lower = title.lower()
            return any(keyword in title_lower for keyword in self.rumor_keywords)
        except Exception:
            return False
    
    def _calculate_corroboration_stats(self, headlines: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate corroboration statistics"""
        if not headlines:
            return {}
        
        total_headlines = len(headlines)
        corroborated_count = sum(1 for h in headlines if h.get('is_corroborated', False))
        speculative_count = sum(1 for h in headlines if 'speculative' in h.get('flags', []))
        primary_source_count = sum(1 for h in headlines if 'primary_source' in h.get('flags', []))
        
        stats = {
            'total_headlines': total_headlines,
            'corroborated_count': corroborated_count,
            'corroborated_pct': round((corroborated_count / total_headlines) * 100, 1),
            'speculative_count': speculative_count,
            'speculative_pct': round((speculative_count / total_headlines) * 100, 1),
            'primary_source_count': primary_source_count,
            'overall_confidence': self._calculate_overall_confidence(headlines)
        }
        
        return stats
    
    def _calculate_overall_confidence(self, headlines: List[Dict]) -> float:
        """Calculate overall confidence score for the headline set"""
        if not headlines:
            return 0.0
        
        confidence_scores = []
        for headline in headlines:
            base_confidence = 0.5  # Base confidence
            boost = headline.get('confidence_boost', 0.0)
            confidence = max(0.0, min(1.0, base_confidence + boost))
            confidence_scores.append(confidence)
        
        return round(sum(confidence_scores) / len(confidence_scores), 2)