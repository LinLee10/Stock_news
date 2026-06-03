#!/usr/bin/env python3
"""
Enhanced News Scraper with Deep Article Analysis

This module provides state-of-the-art news scraping with:
1. Full article content extraction
2. Advanced sentiment analysis with multiple models
3. Numerical data extraction (prices, percentages, revenues)
4. Content quality assessment
5. Source credibility scoring

Research-based implementation using latest 2024-2025 techniques.
"""

import os
import re
import time
import logging
import requests
import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import json

# Enhanced libraries for content extraction and analysis
from newspaper import Article, Config
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from rapidfuzz import fuzz
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    def flesch_reading_ease(text): return 50.0
    def flesch_kincaid_grade(text): return 10.0

# Configuration
from config.config import ALPHA_VANTAGE_KEY, NEWS_LOOKBACK_DAYS
from services.audit_logger import log_news_processing

logger = logging.getLogger(__name__)

# Data structures for enhanced analysis
@dataclass
class ArticleContent:
    """Enhanced article content with metadata"""
    title: str
    url: str
    publish_date: datetime
    full_text: str
    summary: str
    authors: List[str]
    images: List[str]
    keywords: List[str]
    content_length: int
    reading_time_minutes: int
    language: str
    source_domain: str

@dataclass
class SentimentScore:
    """Multi-model sentiment analysis results"""
    finbert_label: str
    finbert_confidence: float
    finbert_score: float  # -1 to 1 normalized
    gpt4_sentiment: Optional[str] = None  # For future GPT integration
    gpt4_confidence: Optional[float] = None
    ensemble_score: float = 0.0  # Combined score from multiple models
    uncertainty: float = 0.0  # Measure of disagreement between models

@dataclass
class NumericalData:
    """Extracted numerical information from articles"""
    stock_prices: List[Tuple[str, float, str]]  # (ticker, price, context)
    percentages: List[Tuple[float, str]]  # (percentage, context)
    revenues: List[Tuple[float, str, str]]  # (amount, currency, period)
    earnings: List[Tuple[float, str, str]]  # (eps, currency, period)
    market_caps: List[Tuple[float, str]]  # (value, currency)
    volumes: List[Tuple[int, str]]  # (volume, context)
    ratios: List[Tuple[str, float, str]]  # (ratio_type, value, context)

@dataclass
class ContentQuality:
    """Article content quality metrics"""
    readability_score: float  # Flesch reading ease (0-100)
    grade_level: float  # Flesch-Kincaid grade level
    word_count: int
    sentence_count: int
    avg_sentence_length: float
    credibility_score: float  # Based on source, authors, references
    factual_density: float  # Ratio of facts/numbers to total content
    
class EnhancedNewsAnalyzer:
    """State-of-the-art news analysis with deep content extraction"""
    
    def __init__(self):
        self.setup_models()
        self.setup_extractors()
        
        # Source credibility mapping (can be enhanced with real-time scoring)
        self.source_credibility = {
            'reuters.com': 0.95, 'bloomberg.com': 0.95, 'wsj.com': 0.90,
            'cnbc.com': 0.85, 'marketwatch.com': 0.80, 'yahoo.com': 0.75,
            'forbes.com': 0.85, 'ft.com': 0.90, 'businessinsider.com': 0.70,
            'seekingalpha.com': 0.65, 'fool.com': 0.60, 'zacks.com': 0.65
        }
        
        # Financial keywords for enhanced matching
        self.financial_keywords = {
            'earnings', 'revenue', 'profit', 'loss', 'eps', 'guidance', 'outlook',
            'forecast', 'beat', 'miss', 'estimate', 'analyst', 'consensus',
            'dividend', 'split', 'merger', 'acquisition', 'ipo', 'secondary',
            'buyback', 'repurchase', 'investment', 'partnership', 'deal'
        }
        
    def setup_models(self):
        """Initialize sentiment analysis models"""
        try:
            # Primary model: FinBERT (established baseline)
            import torch
            self.finbert_model = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Backup financial sentiment model
            try:
                self.alt_finbert = pipeline(
                    "sentiment-analysis", 
                    model="ahmedrachid/FinancialBERT-Sentiment-Analysis",
                    device=0 if torch.cuda.is_available() else -1
                )
            except:
                self.alt_finbert = None
                
            # Initialize spaCy for NER and advanced text processing
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("spaCy en_core_web_sm not found. Running without advanced NLP features.")
                    self.nlp = None
            else:
                logger.info("spaCy not available. Running without advanced NLP features.")
                self.nlp = None
                
            logger.info("✅ Sentiment analysis models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading sentiment models: {e}")
            raise
    
    def setup_extractors(self):
        """Initialize content extraction tools"""
        # Configure newspaper3k for optimal content extraction
        self.newspaper_config = Config()
        self.newspaper_config.browser_user_agent = (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )
        self.newspaper_config.request_timeout = 10
        self.newspaper_config.number_threads = 1
        self.newspaper_config.fetch_images = True
        
        # Financial number extraction patterns
        self.financial_patterns = {
            'stock_price': re.compile(r'\$(\d+(?:\.\d{2})?)\s*(?:per\s+share|share|stock)?', re.IGNORECASE),
            'percentage': re.compile(r'(\d+(?:\.\d+)?)\s*%', re.IGNORECASE),
            'revenue': re.compile(r'revenue\s+of\s+\$(\d+(?:\.\d+)?)\s*(billion|million|thousand)?', re.IGNORECASE),
            'earnings': re.compile(r'earnings?\s+(?:per\s+share\s+)?of\s+\$(\d+(?:\.\d+)?)', re.IGNORECASE),
            'market_cap': re.compile(r'market\s+cap(?:italization)?\s+of\s+\$(\d+(?:\.\d+)?)\s*(billion|million)?', re.IGNORECASE),
            'pe_ratio': re.compile(r'p/e\s+ratio\s+of\s+(\d+(?:\.\d+)?)', re.IGNORECASE),
            'volume': re.compile(r'volume\s+of\s+(\d+(?:,\d{3})*)', re.IGNORECASE)
        }
    
    def extract_full_article(self, url: str) -> Optional[ArticleContent]:
        """Extract full article content using newspaper3k and BeautifulSoup fallback"""
        try:
            # Primary extraction with newspaper3k
            article = Article(url, config=self.newspaper_config)
            article.download()
            article.parse()
            article.nlp()
            
            if not article.text or len(article.text) < 100:
                # Fallback to BeautifulSoup for stubborn sites
                return self._extract_with_beautifulsoup(url)
            
            # Parse publish date
            pub_date = article.publish_date or datetime.now()
            if not isinstance(pub_date, datetime):
                pub_date = datetime.now()
            
            # Calculate reading time (average 200 words per minute)
            word_count = len(article.text.split())
            reading_time = max(1, round(word_count / 200))
            
            return ArticleContent(
                title=article.title or "",
                url=url,
                publish_date=pub_date,
                full_text=article.text or "",
                summary=article.summary or "",
                authors=article.authors or [],
                images=article.images or [],
                keywords=article.keywords or [],
                content_length=len(article.text or ""),
                reading_time_minutes=reading_time,
                language=article.meta_lang or "en",
                source_domain=urlparse(url).netloc.lower()
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract article from {url}: {e}")
            return self._extract_with_beautifulsoup(url)
    
    def _extract_with_beautifulsoup(self, url: str) -> Optional[ArticleContent]:
        """Fallback content extraction using BeautifulSoup"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'aside', 'advertisement']):
                element.decompose()
            
            # Extract title
            title = ""
            for selector in ['h1', 'title', '[class*="title"]', '[class*="headline"]']:
                title_elem = soup.select_one(selector)
                if title_elem and title_elem.get_text().strip():
                    title = title_elem.get_text().strip()
                    break
            
            # Extract main content
            content = ""
            for selector in ['[class*="article"]', '[class*="content"]', '[class*="story"]', 'main', 'article']:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Get all paragraphs
                    paragraphs = content_elem.find_all('p')
                    if paragraphs:
                        content = '\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                        break
            
            if not content or len(content) < 50:
                # Last resort: get all paragraph text
                paragraphs = soup.find_all('p')
                content = '\n'.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20])
            
            word_count = len(content.split())
            reading_time = max(1, round(word_count / 200))
            
            return ArticleContent(
                title=title,
                url=url,
                publish_date=datetime.now(),  # Fallback date
                full_text=content,
                summary="",  # Not available in fallback
                authors=[],
                images=[],
                keywords=[],
                content_length=len(content),
                reading_time_minutes=reading_time,
                language="en",
                source_domain=urlparse(url).netloc.lower()
            )
            
        except Exception as e:
            logger.error(f"BeautifulSoup fallback failed for {url}: {e}")
            return None
    
    def analyze_sentiment(self, text: str, title: str = "") -> SentimentScore:
        """Advanced multi-model sentiment analysis"""
        try:
            # Primary analysis with FinBERT
            finbert_result = self.finbert_model(text[:512])[0]  # FinBERT max length
            
            label = finbert_result['label'].lower()
            confidence = finbert_result['score']
            
            # Normalize to -1 to 1 scale
            if label == 'positive':
                score = confidence
            elif label == 'negative':
                score = -confidence
            else:  # neutral
                score = 0.0
                
            # Get alternative model sentiment if available
            alt_score = None
            alt_confidence = None
            if self.alt_finbert:
                try:
                    alt_result = self.alt_finbert(text[:512])[0]
                    alt_label = alt_result['label'].lower()
                    alt_confidence = alt_result['score']
                    if alt_label == 'positive':
                        alt_score = alt_confidence
                    elif alt_label == 'negative':
                        alt_score = -alt_confidence
                    else:
                        alt_score = 0.0
                except:
                    pass
            
            # Calculate ensemble score and uncertainty
            if alt_score is not None:
                ensemble_score = (score + alt_score) / 2
                uncertainty = abs(score - alt_score)
            else:
                ensemble_score = score
                uncertainty = 0.0
            
            # Boost sentiment if title strongly agrees
            if title:
                title_result = self.finbert_model(title)[0]
                title_label = title_result['label'].lower()
                title_confidence = title_result['score']
                
                if title_label == label and title_confidence > 0.8:
                    ensemble_score *= 1.1  # 10% boost for title agreement
                    ensemble_score = max(-1.0, min(1.0, ensemble_score))  # Keep in bounds
            
            return SentimentScore(
                finbert_label=label,
                finbert_confidence=confidence,
                finbert_score=score,
                ensemble_score=ensemble_score,
                uncertainty=uncertainty
            )
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return SentimentScore(
                finbert_label='neutral',
                finbert_confidence=0.5,
                finbert_score=0.0,
                ensemble_score=0.0,
                uncertainty=1.0
            )
    
    def extract_numerical_data(self, text: str, ticker: str) -> NumericalData:
        """Extract financial numbers and metrics from article text"""
        numerical_data = NumericalData(
            stock_prices=[], percentages=[], revenues=[],
            earnings=[], market_caps=[], volumes=[], ratios=[]
        )
        
        try:
            # Extract stock prices
            for match in self.financial_patterns['stock_price'].finditer(text):
                price = float(match.group(1))
                context = self._get_context(text, match.start(), match.end(), 50)
                numerical_data.stock_prices.append((ticker, price, context))
            
            # Extract percentages
            for match in self.financial_patterns['percentage'].finditer(text):
                percentage = float(match.group(1))
                context = self._get_context(text, match.start(), match.end(), 30)
                numerical_data.percentages.append((percentage, context))
            
            # Extract revenue figures
            for match in self.financial_patterns['revenue'].finditer(text):
                amount = float(match.group(1))
                unit = match.group(2) or "dollars"
                context = self._get_context(text, match.start(), match.end(), 50)
                
                # Convert to standard units (billions)
                if unit.lower() == 'million':
                    amount = amount / 1000
                elif unit.lower() == 'thousand':
                    amount = amount / 1000000
                    
                numerical_data.revenues.append((amount, "USD", context))
            
            # Extract earnings per share
            for match in self.financial_patterns['earnings'].finditer(text):
                eps = float(match.group(1))
                context = self._get_context(text, match.start(), match.end(), 40)
                numerical_data.earnings.append((eps, "USD", context))
            
            # Extract market cap
            for match in self.financial_patterns['market_cap'].finditer(text):
                value = float(match.group(1))
                unit = match.group(2) or "dollars"
                context = self._get_context(text, match.start(), match.end(), 50)
                
                # Convert to standard units (billions)
                if unit.lower() == 'million':
                    value = value / 1000
                    
                numerical_data.market_caps.append((value, "USD"))
            
            # Extract P/E ratios
            for match in self.financial_patterns['pe_ratio'].finditer(text):
                ratio = float(match.group(1))
                context = self._get_context(text, match.start(), match.end(), 30)
                numerical_data.ratios.append(("P/E", ratio, context))
            
            # Extract volume
            for match in self.financial_patterns['volume'].finditer(text):
                volume = int(match.group(1).replace(',', ''))
                context = self._get_context(text, match.start(), match.end(), 40)
                numerical_data.volumes.append((volume, context))
                
        except Exception as e:
            logger.error(f"Error extracting numerical data: {e}")
        
        return numerical_data
    
    def _get_context(self, text: str, start: int, end: int, window: int = 40) -> str:
        """Extract context around a matched pattern"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def assess_content_quality(self, content: ArticleContent) -> ContentQuality:
        """Assess the quality and credibility of article content"""
        text = content.full_text
        
        # Basic readability metrics
        readability = flesch_reading_ease(text) if text else 0
        grade_level = flesch_kincaid_grade(text) if text else 0
        
        # Count sentences and calculate average length
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        words = text.split()
        word_count = len(words)
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Source credibility score
        domain = content.source_domain
        credibility_score = self.source_credibility.get(domain, 0.5)  # Default neutral
        
        # Boost credibility if authors are listed
        if content.authors:
            credibility_score = min(1.0, credibility_score + 0.1)
        
        # Calculate factual density (numbers, financial terms, specific dates)
        fact_patterns = len(re.findall(r'\d+(?:\.\d+)?', text))  # Numbers
        financial_terms = sum(1 for keyword in self.financial_keywords if keyword in text.lower())
        date_patterns = len(re.findall(r'\b\d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b', text))
        
        factual_density = (fact_patterns + financial_terms + date_patterns) / max(1, word_count)
        
        return ContentQuality(
            readability_score=readability,
            grade_level=grade_level,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length,
            credibility_score=credibility_score,
            factual_density=min(1.0, factual_density * 100)  # Scale to 0-1
        )


def enhanced_scrape_headlines(tickers: List[str], days: int = None) -> Dict[str, Dict]:
    """
    Enhanced news scraping with full article analysis
    
    Returns comprehensive analysis including:
    - Full article content and metadata
    - Advanced multi-model sentiment analysis  
    - Extracted numerical data (prices, percentages, ratios)
    - Content quality assessment
    - Source credibility scoring
    """
    with log_news_processing("enhanced_scraping", metadata={'tickers': len(tickers), 'days': days}) as op:
        
        if days is None:
            days = NEWS_LOOKBACK_DAYS
            
        analyzer = EnhancedNewsAnalyzer()
        op.step("analyzer_initialized")
        
        # Get date window
        from news_scraper import get_date_window
        window_start_iso, today_iso = get_date_window(lookback_days=days, tz="America/Los_Angeles")
        today = datetime.fromisoformat(today_iso).date()
        window_start = datetime.fromisoformat(window_start_iso).date()
        
        op.step("date_window_calculated", metadata={'start': window_start_iso, 'end': today_iso})
        
        results = {}
        
        for ticker in tickers:
            with log_news_processing("enhanced_ticker_analysis", ticker, {'ticker': ticker}) as ticker_op:
                try:
                    # Get basic headlines first (reuse existing logic)
                    from news_scraper import fetch_google_rss, parse_rss, fetch_av_news, smart_match_with_alternatives
                    from news_scraper import TICKER_COMPANY
                    
                    comp_name = TICKER_COMPANY.get(ticker, ticker)
                    google_xml = fetch_google_rss(comp_name)
                    google_items = list(parse_rss(google_xml))
                    av_items = fetch_av_news(ticker)
                    
                    ticker_op.step("headlines_fetched", count_in=len(google_items) + len(av_items))
                    
                    # Filter and deduplicate
                    combined = []
                    for title, link, pub_dt in google_items + av_items:
                        pub_date = pub_dt.date()
                        if (smart_match_with_alternatives(ticker, title) and 
                            window_start <= pub_date <= today and link):
                            combined.append((title, link, pub_date))
                    
                    # Remove duplicates by URL
                    seen_urls = set()
                    unique_articles = []
                    for title, link, date in combined:
                        if link not in seen_urls:
                            seen_urls.add(link)
                            unique_articles.append((title, link, date))
                    
                    ticker_op.step("articles_filtered", count_out=len(unique_articles))
                    
                    # Enhanced analysis for each article
                    enhanced_articles = []
                    total_sentiment_scores = []
                    numerical_extracts = []
                    quality_scores = []
                    
                    for i, (title, url, date) in enumerate(unique_articles[:10]):  # Limit to prevent overload
                        try:
                            # Extract full article content
                            article_content = analyzer.extract_full_article(url)
                            if not article_content or len(article_content.full_text) < 100:
                                continue
                            
                            # Perform advanced sentiment analysis
                            sentiment = analyzer.analyze_sentiment(
                                article_content.full_text, article_content.title
                            )
                            
                            # Extract numerical data
                            numerical_data = analyzer.extract_numerical_data(
                                article_content.full_text, ticker
                            )
                            
                            # Assess content quality
                            content_quality = analyzer.assess_content_quality(article_content)
                            
                            enhanced_article = {
                                'title': article_content.title,
                                'url': url,
                                'date': date.isoformat(),
                                'full_content': article_content.full_text[:1000],  # Truncate for storage
                                'summary': article_content.summary,
                                'authors': article_content.authors,
                                'reading_time_minutes': article_content.reading_time_minutes,
                                'source_domain': article_content.source_domain,
                                'sentiment': {
                                    'label': sentiment.finbert_label,
                                    'confidence': sentiment.finbert_confidence,
                                    'score': sentiment.finbert_score,
                                    'ensemble_score': sentiment.ensemble_score,
                                    'uncertainty': sentiment.uncertainty
                                },
                                'numerical_data': {
                                    'stock_prices': numerical_data.stock_prices,
                                    'percentages': numerical_data.percentages,
                                    'revenues': numerical_data.revenues,
                                    'earnings': numerical_data.earnings,
                                    'market_caps': numerical_data.market_caps,
                                    'ratios': numerical_data.ratios
                                },
                                'quality_metrics': {
                                    'readability_score': content_quality.readability_score,
                                    'credibility_score': content_quality.credibility_score,
                                    'factual_density': content_quality.factual_density,
                                    'word_count': content_quality.word_count
                                }
                            }
                            
                            enhanced_articles.append(enhanced_article)
                            total_sentiment_scores.append(sentiment.ensemble_score)
                            numerical_extracts.extend(numerical_data.stock_prices)
                            quality_scores.append(content_quality.credibility_score)
                            
                            if i % 3 == 0:  # Log progress every 3 articles
                                ticker_op.step(f"analyzed_article_{i+1}", count_out=1)
                        
                        except Exception as e:
                            logger.warning(f"Error analyzing article {url}: {e}")
                            continue
                    
                    ticker_op.step("enhanced_analysis_complete", count_out=len(enhanced_articles))
                    
                    # Aggregate metrics
                    avg_sentiment = np.mean(total_sentiment_scores) if total_sentiment_scores else 0.0
                    sentiment_std = np.std(total_sentiment_scores) if len(total_sentiment_scores) > 1 else 0.0
                    avg_credibility = np.mean(quality_scores) if quality_scores else 0.5
                    
                    # Count sentiment distribution
                    positive_count = sum(1 for score in total_sentiment_scores if score > 0.1)
                    negative_count = sum(1 for score in total_sentiment_scores if score < -0.1)
                    neutral_count = len(total_sentiment_scores) - positive_count - negative_count
                    
                    results[ticker] = {
                        'enhanced_articles': enhanced_articles,
                        'article_count': len(enhanced_articles),
                        'sentiment_analysis': {
                            'average_sentiment': avg_sentiment,
                            'sentiment_volatility': sentiment_std,
                            'positive_count': positive_count,
                            'negative_count': negative_count,
                            'neutral_count': neutral_count
                        },
                        'numerical_insights': {
                            'total_price_mentions': len(numerical_extracts),
                            'price_data': numerical_extracts[:5]  # Top 5 price mentions
                        },
                        'quality_metrics': {
                            'average_credibility': avg_credibility,
                            'total_words_analyzed': sum(a['quality_metrics']['word_count'] 
                                                      for a in enhanced_articles),
                            'sources_count': len(set(a['source_domain'] for a in enhanced_articles))
                        }
                    }
                    
                    logger.info(
                        f"[{ticker}] Enhanced analysis: {len(enhanced_articles)} articles, "
                        f"avg_sentiment={avg_sentiment:.3f}, credibility={avg_credibility:.3f} "
                        f"(tz=America/Los_Angeles, lookback={days}d)"
                    )
                    
                except Exception as e:
                    logger.error(f"Error in enhanced analysis for {ticker}: {e}")
                    results[ticker] = {
                        'enhanced_articles': [],
                        'article_count': 0,
                        'error': str(e)
                    }
        
        op.step("enhanced_scraping_complete", count_out=len(results))
        return results


# Compatibility function for backward compatibility
def scrape_headlines_enhanced(tickers: List[str], days: int = None) -> Dict[str, Dict]:
    """Backward compatible function that returns enhanced results in original format"""
    enhanced_results = enhanced_scrape_headlines(tickers, days)
    
    # Convert to original format for compatibility
    compatible_results = {}
    for ticker, data in enhanced_results.items():
        if 'enhanced_articles' in data:
            articles = data['enhanced_articles']
            headlines = [(a['title'], a['url'], datetime.fromisoformat(a['date']).date()) 
                        for a in articles]
            
            # Create daily sentiment mapping
            daily_sentiment = defaultdict(list)
            for article in articles:
                date = datetime.fromisoformat(article['date']).date()
                sentiment_score = article['sentiment']['ensemble_score']
                daily_sentiment[date].append(sentiment_score)
            
            daily_avg_sentiment = {
                date: sum(scores)/len(scores) 
                for date, scores in daily_sentiment.items()
            }
            
            compatible_results[ticker] = {
                'headlines': headlines,
                'count': len(headlines),
                'daily_sentiment': daily_avg_sentiment,
                'count_positive': data['sentiment_analysis']['positive_count'],
                'count_negative': data['sentiment_analysis']['negative_count'], 
                'count_neutral': data['sentiment_analysis']['neutral_count'],
                # Enhanced data
                'enhanced_analysis': data
            }
        else:
            # Error case
            compatible_results[ticker] = {
                'headlines': [],
                'count': 0,
                'daily_sentiment': {},
                'count_positive': 0,
                'count_negative': 0,
                'count_neutral': 0,
                'error': data.get('error', 'Unknown error')
            }
    
    return compatible_results


if __name__ == "__main__":
    # Test the enhanced scraper
    import torch
    test_results = enhanced_scrape_headlines(['AAPL'], days=7)
    
    for ticker, data in test_results.items():
        print(f"\n=== {ticker} Enhanced Analysis ===")
        print(f"Articles analyzed: {data.get('article_count', 0)}")
        if 'sentiment_analysis' in data:
            print(f"Average sentiment: {data['sentiment_analysis']['average_sentiment']:.3f}")
            print(f"Sentiment counts: +{data['sentiment_analysis']['positive_count']} "
                  f"-{data['sentiment_analysis']['negative_count']} "
                  f"~{data['sentiment_analysis']['neutral_count']}")
        if 'quality_metrics' in data:
            print(f"Average credibility: {data['quality_metrics']['average_credibility']:.3f}")
            print(f"Total words analyzed: {data['quality_metrics']['total_words_analyzed']:,}")