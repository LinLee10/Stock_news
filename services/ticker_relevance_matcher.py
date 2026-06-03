"""
TickerRelevanceMatcher - Advanced stock ticker relevance matching using NLP.

This module provides sophisticated matching of financial news articles to stock tickers using:
1. Named Entity Recognition for organization extraction
2. Fuzzy matching against SEC EDGAR company database
3. Context analysis to determine mention significance
4. Relevance scoring based on financial keywords proximity
"""

import re
import json
import time
import logging
from typing import Dict, List, Set, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import defaultdict, Counter
from pathlib import Path

import spacy
import requests
import structlog
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

logger = structlog.get_logger(__name__)


class RelevanceScore(NamedTuple):
    """Relevance score for a ticker match"""
    ticker: str
    relevance_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    mention_count: int
    context_significance: float  # 0.0 to 1.0
    financial_keywords_nearby: int
    company_name_matches: List[str]
    key_sentences: List[str]


@dataclass 
class CompanyInfo:
    """Company information from SEC EDGAR database"""
    ticker: str
    company_name: str
    alternative_names: List[str]
    cik: str
    sic_code: Optional[str] = None
    industry: Optional[str] = None


class TickerRelevanceMatcher:
    """
    Advanced ticker relevance matching using NLP and financial context analysis.
    
    Features:
    - spaCy NER for organization extraction
    - SEC EDGAR company database integration
    - Context-aware relevance scoring
    - Financial keyword proximity analysis
    - Multi-strategy fuzzy matching
    - Real-time and batch processing
    """
    
    def __init__(self, model_name: str = "en_core_web_sm", sec_data_path: Optional[str] = None):
        self.model_name = model_name
        self.sec_data_path = sec_data_path
        
        # Initialize spaCy model
        try:
            self.nlp = spacy.load(model_name)
            logger.info("spaCy model loaded", model=model_name)
        except OSError:
            logger.error(f"spaCy model '{model_name}' not found. Install with: python -m spacy download {model_name}")
            raise
        
        # Initialize NLTK components
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK stopwords not found, downloading...")
            nltk.download('stopwords')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words('english'))
        
        # Company database
        self.company_db: Dict[str, CompanyInfo] = {}
        self.company_name_index: Dict[str, str] = {}  # company_name -> ticker
        self.alternative_names_index: Dict[str, str] = {}  # alt_name -> ticker
        
        # Initialize company database
        self._initialize_company_database()
        
        # Financial context keywords with weights
        self.financial_keywords = {
            # High significance - direct financial impact
            'earnings': 1.0, 'revenue': 1.0, 'profit': 1.0, 'loss': 1.0,
            'eps': 1.0, 'guidance': 1.0, 'forecast': 1.0, 'outlook': 1.0,
            'dividend': 0.9, 'split': 0.9, 'buyback': 0.9, 'repurchase': 0.9,
            'merger': 0.9, 'acquisition': 0.9, 'ipo': 0.9, 'secondary': 0.8,
            
            # Medium significance - business operations
            'sales': 0.8, 'growth': 0.7, 'expansion': 0.7, 'partnership': 0.7,
            'deal': 0.7, 'contract': 0.7, 'investment': 0.8, 'funding': 0.7,
            'launch': 0.6, 'product': 0.6, 'service': 0.6, 'technology': 0.5,
            
            # Financial metrics
            'margin': 0.8, 'ebitda': 0.9, 'cash_flow': 0.8, 'debt': 0.7,
            'equity': 0.7, 'assets': 0.6, 'liability': 0.6, 'book_value': 0.7,
            'market_cap': 0.8, 'valuation': 0.8, 'ratio': 0.6,
            
            # Market-related terms
            'stock': 0.8, 'share': 0.8, 'price': 0.7, 'target': 0.7,
            'upgrade': 0.8, 'downgrade': 0.8, 'rating': 0.7, 'analyst': 0.6,
            'consensus': 0.6, 'estimate': 0.6, 'beat': 0.8, 'miss': 0.8,
        }
        
        # Context patterns that boost relevance
        self.high_relevance_patterns = [
            r'\b(?:CEO|CFO|president|chairman|executive)\b',
            r'\bquarterly\s+(?:results|earnings|report)\b',
            r'\bannual\s+(?:results|earnings|report)\b',
            r'\b(?:beats?|misses?)\s+(?:estimates?|expectations?)\b',
            r'\b(?:announces?|reports?|posts?)\s+(?:earnings|revenue|profit|loss)\b',
            r'\b(?:raises?|cuts?|maintains?)\s+guidance\b',
            r'\b(?:declares?|pays?)\s+dividend\b',
            r'\bstock\s+(?:price|split|buyback)\b'
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.high_relevance_patterns]
    
    def _initialize_company_database(self):
        """Initialize company database from various sources"""
        
        # Load from local SEC data if available
        if self.sec_data_path and Path(self.sec_data_path).exists():
            self._load_sec_database()
        
        # Fallback to built-in ticker database from config
        self._load_builtin_tickers()
        
        # Build search indexes
        self._build_search_indexes()
        
        logger.info("Company database initialized", 
                   company_count=len(self.company_db),
                   index_entries=len(self.company_name_index))
    
    def _load_sec_database(self):
        """Load company information from SEC EDGAR database"""
        try:
            with open(self.sec_data_path, 'r') as f:
                sec_data = json.load(f)
            
            for entry in sec_data:
                ticker = entry.get('ticker', '').upper()
                if not ticker:
                    continue
                
                company_info = CompanyInfo(
                    ticker=ticker,
                    company_name=entry.get('title', ''),
                    alternative_names=entry.get('alternative_names', []),
                    cik=entry.get('cik_str', ''),
                    sic_code=entry.get('sic_code'),
                    industry=entry.get('industry')
                )
                
                self.company_db[ticker] = company_info
                
            logger.info("SEC database loaded", companies=len(self.company_db))
            
        except Exception as e:
            logger.warning("Failed to load SEC database", path=self.sec_data_path, error=str(e))
    
    def _load_builtin_tickers(self):
        """Load built-in ticker database from config"""
        try:
            from config.tickers import TICKER_COMPANY
            
            for ticker, company_name in TICKER_COMPANY.items():
                if ticker not in self.company_db:
                    # Generate alternative names
                    alt_names = self._generate_alternative_names(company_name)
                    
                    company_info = CompanyInfo(
                        ticker=ticker.upper(),
                        company_name=company_name,
                        alternative_names=alt_names,
                        cik='',  # Not available in built-in data
                    )
                    
                    self.company_db[ticker.upper()] = company_info
            
            logger.info("Built-in ticker database loaded", companies=len(TICKER_COMPANY))
            
        except ImportError:
            logger.warning("Built-in ticker database not available")
    
    def _generate_alternative_names(self, company_name: str) -> List[str]:
        """Generate alternative names for a company"""
        alternatives = []
        
        # Remove common suffixes
        base_name = re.sub(r'\b(Inc|Corp|Corporation|Ltd|Limited|Company|Co|Group|Holdings?|Plc)\b\.?$', 
                          '', company_name, flags=re.IGNORECASE).strip()
        
        if base_name != company_name:
            alternatives.append(base_name)
        
        # Add with different suffixes
        if not any(suffix in company_name.lower() for suffix in ['inc', 'corp', 'ltd', 'co']):
            alternatives.extend([
                f"{base_name} Inc",
                f"{base_name} Corp",
                f"{base_name} Corporation"
            ])
        
        # Handle acronyms (e.g., "International Business Machines" -> "IBM")
        words = base_name.split()
        if len(words) >= 2:
            acronym = ''.join(word[0].upper() for word in words if word[0].isalpha())
            if len(acronym) >= 2:
                alternatives.append(acronym)
        
        return list(set(alternatives))
    
    def _build_search_indexes(self):
        """Build search indexes for fast company name lookups"""
        for ticker, company_info in self.company_db.items():
            # Index company name
            name_normalized = company_info.company_name.lower().strip()
            self.company_name_index[name_normalized] = ticker
            
            # Index alternative names
            for alt_name in company_info.alternative_names:
                alt_normalized = alt_name.lower().strip()
                self.alternative_names_index[alt_normalized] = ticker
        
        logger.debug("Search indexes built",
                    name_index_size=len(self.company_name_index),
                    alt_index_size=len(self.alternative_names_index))
    
    def analyze_article_relevance(self, title: str, content: str, target_tickers: Optional[List[str]] = None) -> List[RelevanceScore]:
        """
        Analyze article relevance for stock tickers.
        
        Args:
            title: Article title
            content: Article content
            target_tickers: Optional list of specific tickers to analyze
            
        Returns:
            List of RelevanceScore objects sorted by relevance
        """
        start_time = time.time()
        
        try:
            # Combine title and content for analysis
            full_text = f"{title}\n\n{content}"
            
            # Extract organizations using spaCy NER
            organizations = self._extract_organizations(full_text)
            
            # Find ticker matches
            ticker_matches = self._find_ticker_matches(full_text, organizations, target_tickers)
            
            # Calculate relevance scores
            relevance_scores = []
            
            for ticker in ticker_matches:
                score = self._calculate_relevance_score(
                    ticker, title, content, organizations, ticker_matches[ticker]
                )
                relevance_scores.append(score)
            
            # Sort by relevance score
            relevance_scores.sort(key=lambda x: x.relevance_score, reverse=True)
            
            processing_time = time.time() - start_time
            
            logger.debug("Article relevance analysis completed",
                        ticker_count=len(relevance_scores),
                        processing_time=f"{processing_time:.3f}s")
            
            return relevance_scores
            
        except Exception as e:
            logger.error("Article relevance analysis failed", error=str(e))
            return []
    
    def _extract_organizations(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Extract organization entities using spaCy NER.
        
        Args:
            text: Input text
            
        Returns:
            List of (organization_name, start_pos, end_pos) tuples
        """
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            organizations = []
            
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PERSON']:  # Organizations and key people
                    org_name = ent.text.strip()
                    
                    # Filter out common false positives
                    if (len(org_name) > 2 and 
                        not org_name.lower() in self.stop_words and
                        not re.match(r'^[A-Z]{1,2}$', org_name)):  # Avoid single/double letter matches
                        
                        organizations.append((org_name, ent.start_char, ent.end_char))
            
            # Also extract potential ticker symbols ($AAPL, AAPL:, etc.)
            ticker_pattern = r'\b(?:\$)?([A-Z]{2,5})\b(?=\s|:|$|\.)'
            for match in re.finditer(ticker_pattern, text):
                ticker = match.group(1)
                if ticker in self.company_db:
                    organizations.append((ticker, match.start(), match.end()))
            
            return organizations
            
        except Exception as e:
            logger.warning("Organization extraction failed", error=str(e))
            return []
    
    def _find_ticker_matches(self, text: str, organizations: List[Tuple[str, int, int]], 
                           target_tickers: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Find ticker matches from extracted organizations and text analysis.
        
        Args:
            text: Full text content
            organizations: Extracted organizations
            target_tickers: Optional specific tickers to focus on
            
        Returns:
            Dictionary of ticker -> match_info
        """
        ticker_matches = defaultdict(lambda: {
            'mention_count': 0,
            'company_name_matches': [],
            'positions': [],
            'match_types': []
        })
        
        text_lower = text.lower()
        
        # Strategy 1: Direct ticker symbol matches
        for ticker in self.company_db.keys():
            if target_tickers and ticker not in target_tickers:
                continue
            
            # Look for ticker mentions
            ticker_pattern = rf'\b{re.escape(ticker)}\b'
            matches = list(re.finditer(ticker_pattern, text, re.IGNORECASE))
            
            if matches:
                ticker_matches[ticker]['mention_count'] += len(matches)
                ticker_matches[ticker]['positions'].extend([(m.start(), m.end()) for m in matches])
                ticker_matches[ticker]['match_types'].extend(['ticker_symbol'] * len(matches))
        
        # Strategy 2: Company name matching (exact and fuzzy)
        for org_name, start_pos, end_pos in organizations:
            best_ticker = None
            best_score = 0.0
            
            # Exact match check
            org_lower = org_name.lower()
            if org_lower in self.company_name_index:
                best_ticker = self.company_name_index[org_lower]
                best_score = 1.0
            elif org_lower in self.alternative_names_index:
                best_ticker = self.alternative_names_index[org_lower]
                best_score = 0.9
            
            # Fuzzy matching for partial matches
            if best_score < 0.8:
                # Search all company names
                all_names = list(self.company_name_index.keys()) + list(self.alternative_names_index.keys())
                fuzzy_matches = process.extract(org_lower, all_names, limit=3, scorer=fuzz.partial_ratio)
                
                for match_name, score in fuzzy_matches:
                    if score >= 85:  # 85% similarity threshold
                        if match_name in self.company_name_index:
                            candidate_ticker = self.company_name_index[match_name]
                        else:
                            candidate_ticker = self.alternative_names_index[match_name]
                        
                        if target_tickers and candidate_ticker not in target_tickers:
                            continue
                        
                        if score/100.0 > best_score:
                            best_ticker = candidate_ticker
                            best_score = score/100.0
            
            # Record match if found
            if best_ticker and best_score >= 0.7:
                ticker_matches[best_ticker]['mention_count'] += 1
                ticker_matches[best_ticker]['company_name_matches'].append(org_name)
                ticker_matches[best_ticker]['positions'].append((start_pos, end_pos))
                ticker_matches[best_ticker]['match_types'].append(f'company_name_{best_score:.2f}')
        
        # Strategy 3: Context-based discovery for target tickers
        if target_tickers:
            for ticker in target_tickers:
                if ticker in self.company_db and ticker not in ticker_matches:
                    company_info = self.company_db[ticker]
                    
                    # Check for partial company name mentions
                    company_words = company_info.company_name.lower().split()
                    significant_words = [w for w in company_words 
                                       if len(w) > 3 and w not in self.stop_words]
                    
                    if significant_words:
                        context_score = 0.0
                        for word in significant_words:
                            if word in text_lower:
                                context_score += 0.3
                        
                        if context_score >= 0.5:  # At least 2 significant words
                            ticker_matches[ticker]['mention_count'] = 1
                            ticker_matches[ticker]['match_types'].append(f'context_{context_score:.2f}')
        
        return dict(ticker_matches)
    
    def _calculate_relevance_score(self, ticker: str, title: str, content: str, 
                                 organizations: List, match_info: Dict) -> RelevanceScore:
        """
        Calculate comprehensive relevance score for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            title: Article title
            content: Article content
            organizations: Extracted organizations
            match_info: Match information from _find_ticker_matches
            
        Returns:
            RelevanceScore object
        """
        try:
            # Base score from mention count
            mention_count = match_info['mention_count']
            base_score = min(1.0, mention_count * 0.3)  # Diminishing returns
            
            # Title boost (mentions in title are more significant)
            title_boost = 0.0
            company_info = self.company_db.get(ticker)
            if company_info:
                title_lower = title.lower()
                if ticker.lower() in title_lower:
                    title_boost += 0.3
                if company_info.company_name.lower() in title_lower:
                    title_boost += 0.4
                for alt_name in company_info.alternative_names:
                    if alt_name.lower() in title_lower:
                        title_boost += 0.2
                        break
            
            # Context significance analysis
            context_significance = self._analyze_context_significance(ticker, content, match_info['positions'])
            
            # Financial keywords proximity
            financial_keywords_count = self._count_financial_keywords_nearby(content, match_info['positions'])
            financial_boost = min(0.4, financial_keywords_count * 0.1)
            
            # High-relevance pattern matching
            pattern_boost = 0.0
            for pattern in self.compiled_patterns:
                if pattern.search(content):
                    pattern_boost += 0.1
            pattern_boost = min(0.3, pattern_boost)
            
            # Calculate final relevance score
            relevance_score = base_score + title_boost + (context_significance * 0.3) + financial_boost + pattern_boost
            relevance_score = min(1.0, relevance_score)
            
            # Calculate confidence based on match quality
            confidence = self._calculate_confidence(match_info, relevance_score)
            
            # Extract key sentences containing the ticker/company
            key_sentences = self._extract_key_sentences(content, match_info['positions'])
            
            return RelevanceScore(
                ticker=ticker,
                relevance_score=relevance_score,
                confidence=confidence,
                mention_count=mention_count,
                context_significance=context_significance,
                financial_keywords_nearby=financial_keywords_count,
                company_name_matches=match_info['company_name_matches'],
                key_sentences=key_sentences
            )
            
        except Exception as e:
            logger.error("Relevance score calculation failed", ticker=ticker, error=str(e))
            return RelevanceScore(
                ticker=ticker,
                relevance_score=0.0,
                confidence=0.0,
                mention_count=0,
                context_significance=0.0,
                financial_keywords_nearby=0,
                company_name_matches=[],
                key_sentences=[]
            )
    
    def _analyze_context_significance(self, ticker: str, content: str, positions: List[Tuple[int, int]]) -> float:
        """Analyze the significance of ticker mentions based on surrounding context"""
        if not positions:
            return 0.0
        
        significance_scores = []
        sentences = sent_tokenize(content)
        
        for start_pos, end_pos in positions:
            # Find the sentence containing this mention
            sentence_significance = 0.0
            
            for sentence in sentences:
                sentence_start = content.find(sentence)
                sentence_end = sentence_start + len(sentence)
                
                # Check if mention is in this sentence
                if sentence_start <= start_pos <= sentence_end:
                    sentence_lower = sentence.lower()
                    
                    # Check for financial keywords in the sentence
                    for keyword, weight in self.financial_keywords.items():
                        if keyword.replace('_', ' ') in sentence_lower:
                            sentence_significance += weight * 0.1
                    
                    # Check for high-relevance patterns
                    for pattern in self.compiled_patterns:
                        if pattern.search(sentence):
                            sentence_significance += 0.2
                    
                    # Boost for numerical data (prices, percentages, etc.)
                    numbers = re.findall(r'\$[\d,]+(?:\.\d{2})?|\d+(?:\.\d+)?%', sentence)
                    sentence_significance += min(0.3, len(numbers) * 0.1)
                    
                    break
            
            significance_scores.append(min(1.0, sentence_significance))
        
        return sum(significance_scores) / len(significance_scores) if significance_scores else 0.0
    
    def _count_financial_keywords_nearby(self, content: str, positions: List[Tuple[int, int]], window: int = 100) -> int:
        """Count financial keywords near ticker mentions"""
        if not positions:
            return 0
        
        content_lower = content.lower()
        keyword_count = 0
        
        for start_pos, end_pos in positions:
            # Define window around mention
            window_start = max(0, start_pos - window)
            window_end = min(len(content), end_pos + window)
            window_text = content_lower[window_start:window_end]
            
            # Count financial keywords in window
            for keyword in self.financial_keywords:
                keyword_clean = keyword.replace('_', ' ')
                if keyword_clean in window_text:
                    keyword_count += 1
        
        return keyword_count
    
    def _calculate_confidence(self, match_info: Dict, relevance_score: float) -> float:
        """Calculate confidence in the relevance score"""
        confidence_factors = []
        
        # Factor 1: Match type quality
        match_types = match_info.get('match_types', [])
        exact_matches = sum(1 for mt in match_types if 'ticker_symbol' in mt or '1.00' in mt)
        fuzzy_matches = sum(1 for mt in match_types if 'company_name' in mt and '1.00' not in mt)
        
        if exact_matches > 0:
            confidence_factors.append(0.9)
        elif fuzzy_matches > 0:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Factor 2: Mention frequency
        mention_count = match_info.get('mention_count', 0)
        if mention_count >= 3:
            confidence_factors.append(0.9)
        elif mention_count >= 2:
            confidence_factors.append(0.8)
        elif mention_count >= 1:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        
        # Factor 3: Relevance score itself
        if relevance_score >= 0.8:
            confidence_factors.append(0.9)
        elif relevance_score >= 0.6:
            confidence_factors.append(0.8)
        elif relevance_score >= 0.4:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _extract_key_sentences(self, content: str, positions: List[Tuple[int, int]], max_sentences: int = 3) -> List[str]:
        """Extract key sentences containing ticker mentions"""
        if not positions:
            return []
        
        sentences = sent_tokenize(content)
        key_sentences = []
        
        for sentence in sentences:
            sentence_start = content.find(sentence)
            sentence_end = sentence_start + len(sentence)
            
            # Check if any mention overlaps with this sentence
            for start_pos, end_pos in positions:
                if (sentence_start <= start_pos <= sentence_end or 
                    sentence_start <= end_pos <= sentence_end):
                    
                    if len(sentence) > 20:  # Avoid very short sentences
                        key_sentences.append(sentence.strip())
                    break
            
            if len(key_sentences) >= max_sentences:
                break
        
        return key_sentences
    
    def batch_analyze_relevance(self, articles: List[Dict], target_tickers: Optional[List[str]] = None) -> List[Tuple[Dict, List[RelevanceScore]]]:
        """
        Batch process multiple articles for ticker relevance.
        
        Args:
            articles: List of article dicts with 'title' and 'content' keys
            target_tickers: Optional list of tickers to focus on
            
        Returns:
            List of (article, relevance_scores) tuples
        """
        results = []
        
        for article in articles:
            try:
                relevance_scores = self.analyze_article_relevance(
                    article.get('title', ''),
                    article.get('content', ''),
                    target_tickers
                )
                results.append((article, relevance_scores))
                
            except Exception as e:
                logger.error("Batch relevance analysis failed for article", 
                           title=article.get('title', 'unknown'), error=str(e))
                results.append((article, []))
        
        return results
    
    def get_supported_tickers(self) -> List[str]:
        """Get list of supported ticker symbols"""
        return list(self.company_db.keys())
    
    def get_company_info(self, ticker: str) -> Optional[CompanyInfo]:
        """Get company information for a ticker"""
        return self.company_db.get(ticker.upper())


# Utility functions for easy integration
def analyze_article_for_tickers(title: str, content: str, target_tickers: List[str] = None) -> List[RelevanceScore]:
    """
    Convenience function to analyze article relevance for tickers.
    
    Args:
        title: Article title
        content: Article content
        target_tickers: Optional specific tickers to analyze
        
    Returns:
        List of RelevanceScore objects
    """
    matcher = TickerRelevanceMatcher()
    return matcher.analyze_article_relevance(title, content, target_tickers)


def filter_relevant_articles(articles: List[Dict], ticker: str, min_relevance: float = 0.3) -> List[Tuple[Dict, RelevanceScore]]:
    """
    Filter articles relevant to a specific ticker.
    
    Args:
        articles: List of article dictionaries
        ticker: Target ticker symbol
        min_relevance: Minimum relevance threshold
        
    Returns:
        List of (article, relevance_score) tuples for relevant articles
    """
    matcher = TickerRelevanceMatcher()
    relevant_articles = []
    
    for article in articles:
        relevance_scores = matcher.analyze_article_relevance(
            article.get('title', ''),
            article.get('content', ''),
            [ticker]
        )
        
        for score in relevance_scores:
            if score.ticker == ticker and score.relevance_score >= min_relevance:
                relevant_articles.append((article, score))
                break
    
    return relevant_articles


if __name__ == "__main__":
    # Test the ticker relevance matcher
    matcher = TickerRelevanceMatcher()
    
    # Test article
    test_article = {
        'title': 'Apple Reports Strong Q4 Earnings, iPhone Sales Beat Expectations',
        'content': '''
        Apple Inc. (AAPL) reported fourth-quarter earnings that exceeded analyst expectations,
        driven by strong iPhone sales and services revenue. The tech giant posted earnings per share 
        of $1.46, beating the consensus estimate of $1.39. Revenue for the quarter reached $89.5 billion,
        up 4.5% year-over-year. CEO Tim Cook highlighted the company's continued growth in emerging markets
        and the success of the new iPhone 15 lineup. Apple also announced a quarterly dividend of $0.24
        per share and authorized an additional $90 billion share buyback program.
        '''
    }
    
    # Analyze relevance
    relevance_scores = matcher.analyze_article_relevance(
        test_article['title'], 
        test_article['content'],
        ['AAPL', 'MSFT', 'GOOGL']
    )
    
    print("Ticker Relevance Analysis Results:")
    print("=" * 50)
    
    for score in relevance_scores:
        print(f"\nTicker: {score.ticker}")
        print(f"Relevance Score: {score.relevance_score:.3f}")
        print(f"Confidence: {score.confidence:.3f}")
        print(f"Mentions: {score.mention_count}")
        print(f"Financial Keywords Nearby: {score.financial_keywords_nearby}")
        print(f"Company Name Matches: {score.company_name_matches}")
        print(f"Key Sentences: {score.key_sentences[:2]}")  # Show first 2 sentences
        print("-" * 30)
    
    # Test company database
    print(f"\nSupported Tickers: {len(matcher.get_supported_tickers())}")
    print(f"Sample Tickers: {matcher.get_supported_tickers()[:10]}")
    
    aapl_info = matcher.get_company_info('AAPL')
    if aapl_info:
        print(f"\nAAPL Company Info:")
        print(f"Name: {aapl_info.company_name}")
        print(f"Alternative Names: {aapl_info.alternative_names}")