"""
ArticleQualityAssessment - Comprehensive framework for evaluating financial news article quality.

This module provides multi-dimensional quality assessment including:
1. FinBERT-based financial relevance scoring
2. Article length and structure validation
3. Source credibility weighting
4. Timeliness scoring with decay functions
5. Content originality and uniqueness analysis
"""

import re
import time
import math
import logging
from typing import Dict, List, Optional, NamedTuple, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from collections import Counter
from pathlib import Path

import structlog
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from textstat import flesch_reading_ease, flesch_kincaid_grade, automated_readability_index
from rapidfuzz import fuzz
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

logger = structlog.get_logger(__name__)


class QualityScore(NamedTuple):
    """Comprehensive quality score for an article"""
    overall_score: float  # 0.0 to 1.0
    financial_relevance: float
    content_quality: float
    source_credibility: float
    timeliness_score: float
    structure_score: float
    originality_score: float
    readability_score: float
    factual_density: float
    confidence: float
    quality_issues: List[str]
    quality_strengths: List[str]


@dataclass
class SourceCredibility:
    """Source credibility information"""
    domain: str
    base_credibility: float
    tier: str  # "tier1", "tier2", "tier3", "unknown"
    reputation_factors: List[str]
    penalty_factors: List[str]


@dataclass
class ContentMetrics:
    """Content structure and quality metrics"""
    word_count: int
    sentence_count: int
    paragraph_count: int
    avg_sentence_length: float
    avg_paragraph_length: float
    readability_score: float
    grade_level: float
    numerical_data_count: int
    quote_count: int
    financial_terms_count: int


class ArticleQualityAssessment:
    """
    Comprehensive article quality assessment framework for financial news.
    
    Features:
    - Multi-model financial relevance scoring
    - Source credibility analysis
    - Content structure and readability assessment
    - Timeliness scoring with decay functions
    - Originality and uniqueness detection
    - Configurable scoring weights and thresholds
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = "auto"):
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self._setup_models()
        
        # Initialize NLTK components
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK stopwords not found, downloading...")
            nltk.download('stopwords')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words('english'))
        
        # Source credibility database
        self.source_credibility_db = self._initialize_credibility_database()
        
        # Financial relevance keywords with weights
        self.financial_keywords = {
            # Core financial terms (high weight)
            'earnings': 3.0, 'revenue': 3.0, 'profit': 3.0, 'loss': 3.0,
            'eps': 2.5, 'ebitda': 2.5, 'cash_flow': 2.5, 'dividend': 2.5,
            'merger': 3.0, 'acquisition': 3.0, 'ipo': 3.0, 'buyback': 2.5,
            
            # Market terms (medium weight)
            'stock': 2.0, 'shares': 2.0, 'market': 2.0, 'trading': 2.0,
            'price': 2.0, 'volume': 1.5, 'volatility': 2.0, 'index': 1.5,
            'analyst': 1.5, 'forecast': 2.0, 'guidance': 2.5, 'outlook': 2.0,
            
            # Business terms (lower weight)
            'growth': 1.5, 'expansion': 1.5, 'partnership': 1.5, 'deal': 1.5,
            'investment': 1.8, 'funding': 1.8, 'valuation': 2.0, 'startup': 1.2,
            'innovation': 1.0, 'technology': 1.0, 'product': 1.0, 'service': 1.0,
        }
        
        # Quality assessment weights
        self.score_weights = {
            'financial_relevance': 0.25,
            'content_quality': 0.20,
            'source_credibility': 0.15,
            'timeliness': 0.10,
            'structure': 0.10,
            'originality': 0.10,
            'readability': 0.05,
            'factual_density': 0.05
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_word_count': 150,
            'max_word_count': 5000,
            'min_sentences': 5,
            'min_paragraphs': 2,
            'max_avg_sentence_length': 35,
            'min_financial_terms': 3,
            'max_hours_for_full_timeliness': 24,
            'timeliness_decay_days': 7
        }
    
    def _setup_models(self):
        """Initialize FinBERT and other ML models"""
        try:
            logger.info("Loading FinBERT model", model=self.model_name, device=self.device)
            
            # Primary FinBERT model
            self.finbert_classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            
            # Backup general financial model
            try:
                self.backup_classifier = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if self.device == "cuda" else -1,
                    return_all_scores=True
                )
            except:
                self.backup_classifier = None
                logger.warning("Backup financial classifier not available")
            
            logger.info("✅ Quality assessment models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading quality assessment models: {e}")
            raise
    
    def _initialize_credibility_database(self) -> Dict[str, SourceCredibility]:
        """Initialize source credibility database"""
        credibility_db = {}
        
        # Tier 1: Premium financial news sources
        tier1_sources = {
            'reuters.com': 0.95, 'bloomberg.com': 0.95, 'wsj.com': 0.90,
            'ft.com': 0.90, 'marketwatch.com': 0.85, 'cnbc.com': 0.85,
            'finance.yahoo.com': 0.80, 'yahoo.com': 0.78, 'economist.com': 0.88
        }
        
        for domain, score in tier1_sources.items():
            credibility_db[domain] = SourceCredibility(
                domain=domain,
                base_credibility=score,
                tier="tier1",
                reputation_factors=["established_brand", "professional_journalism", "fact_checking"],
                penalty_factors=[]
            )
        
        # Tier 2: Business and technology publications
        tier2_sources = {
            'forbes.com': 0.75, 'businessinsider.com': 0.70, 'fortune.com': 0.75,
            'techcrunch.com': 0.70, 'venturebeat.com': 0.65, 'thenextweb.com': 0.60,
            'nytimes.com': 0.80, 'washingtonpost.com': 0.78, 'usatoday.com': 0.72
        }
        
        for domain, score in tier2_sources.items():
            credibility_db[domain] = SourceCredibility(
                domain=domain,
                base_credibility=score,
                tier="tier2",
                reputation_factors=["recognized_brand", "editorial_standards"],
                penalty_factors=[]
            )
        
        # Tier 3: Financial blogs and specialized sites
        tier3_sources = {
            'seekingalpha.com': 0.60, 'fool.com': 0.55, 'zacks.com': 0.58,
            'investopedia.com': 0.70, 'thestreet.com': 0.65, 'gurufocus.com': 0.55,
            'stockhouse.com': 0.45, 'smallcappower.com': 0.40
        }
        
        for domain, score in tier3_sources.items():
            credibility_db[domain] = SourceCredibility(
                domain=domain,
                base_credibility=score,
                tier="tier3",
                reputation_factors=["specialized_content", "community_driven"],
                penalty_factors=["opinion_based", "variable_quality"]
            )
        
        logger.info("Source credibility database initialized", 
                   total_sources=len(credibility_db))
        
        return credibility_db
    
    def assess_article_quality(self, title: str, content: str, source_url: str,
                             publish_date: Optional[datetime] = None,
                             author: Optional[str] = None) -> QualityScore:
        """
        Perform comprehensive quality assessment of a financial news article.
        
        Args:
            title: Article title
            content: Article content
            source_url: Source URL for credibility assessment
            publish_date: Publication date for timeliness scoring
            author: Author name (optional)
            
        Returns:
            QualityScore with comprehensive assessment
        """
        start_time = time.time()
        
        try:
            # Extract content metrics
            content_metrics = self._analyze_content_structure(title, content)
            
            # Financial relevance scoring
            financial_relevance = self._assess_financial_relevance(title, content)
            
            # Content quality scoring
            content_quality = self._assess_content_quality(content_metrics, title, content)
            
            # Source credibility scoring
            source_credibility = self._assess_source_credibility(source_url, author)
            
            # Timeliness scoring
            timeliness_score = self._assess_timeliness(publish_date)
            
            # Structure scoring
            structure_score = self._assess_structure_quality(content_metrics)
            
            # Originality scoring (placeholder for future enhancement)
            originality_score = self._assess_originality(title, content)
            
            # Readability scoring
            readability_score = self._normalize_readability(content_metrics.readability_score)
            
            # Factual density scoring
            factual_density = self._assess_factual_density(content_metrics, content)
            
            # Calculate weighted overall score
            overall_score = (
                financial_relevance * self.score_weights['financial_relevance'] +
                content_quality * self.score_weights['content_quality'] +
                source_credibility * self.score_weights['source_credibility'] +
                timeliness_score * self.score_weights['timeliness'] +
                structure_score * self.score_weights['structure'] +
                originality_score * self.score_weights['originality'] +
                readability_score * self.score_weights['readability'] +
                factual_density * self.score_weights['factual_density']
            )
            
            # Identify quality issues and strengths
            quality_issues, quality_strengths = self._identify_quality_factors(
                content_metrics, financial_relevance, source_credibility, 
                timeliness_score, structure_score
            )
            
            # Calculate confidence based on various factors
            confidence = self._calculate_assessment_confidence(
                content_metrics, financial_relevance, source_credibility
            )
            
            processing_time = time.time() - start_time
            
            logger.debug("Article quality assessment completed",
                        overall_score=f"{overall_score:.3f}",
                        processing_time=f"{processing_time:.3f}s")
            
            return QualityScore(
                overall_score=overall_score,
                financial_relevance=financial_relevance,
                content_quality=content_quality,
                source_credibility=source_credibility,
                timeliness_score=timeliness_score,
                structure_score=structure_score,
                originality_score=originality_score,
                readability_score=readability_score,
                factual_density=factual_density,
                confidence=confidence,
                quality_issues=quality_issues,
                quality_strengths=quality_strengths
            )
            
        except Exception as e:
            logger.error("Article quality assessment failed", error=str(e))
            return QualityScore(
                overall_score=0.0,
                financial_relevance=0.0,
                content_quality=0.0,
                source_credibility=0.5,
                timeliness_score=0.5,
                structure_score=0.0,
                originality_score=0.5,
                readability_score=0.5,
                factual_density=0.0,
                confidence=0.0,
                quality_issues=["assessment_failed"],
                quality_strengths=[]
            )
    
    def _analyze_content_structure(self, title: str, content: str) -> ContentMetrics:
        """Analyze content structure and basic metrics"""
        try:
            # Basic counts
            words = content.split()
            word_count = len(words)
            
            sentences = sent_tokenize(content)
            sentence_count = len(sentences)
            
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            paragraph_count = len(paragraphs)
            
            # Calculate averages
            avg_sentence_length = word_count / max(1, sentence_count)
            avg_paragraph_length = sentence_count / max(1, paragraph_count)
            
            # Readability metrics
            readability_score = flesch_reading_ease(content) if content else 0
            grade_level = flesch_kincaid_grade(content) if content else 0
            
            # Count special content types
            numerical_data_count = len(re.findall(r'\$[\d,]+(?:\.\d{2})?|\d+(?:\.\d+)?%|\d+(?:,\d{3})*(?:\.\d+)?', content))
            quote_count = len(re.findall(r'"[^"]*"', content))
            
            # Count financial terms
            content_lower = content.lower()
            financial_terms_count = sum(
                content_lower.count(term.replace('_', ' ')) * weight
                for term, weight in self.financial_keywords.items()
            )
            
            return ContentMetrics(
                word_count=word_count,
                sentence_count=sentence_count,
                paragraph_count=paragraph_count,
                avg_sentence_length=avg_sentence_length,
                avg_paragraph_length=avg_paragraph_length,
                readability_score=readability_score,
                grade_level=grade_level,
                numerical_data_count=numerical_data_count,
                quote_count=quote_count,
                financial_terms_count=int(financial_terms_count)
            )
            
        except Exception as e:
            logger.warning("Content structure analysis failed", error=str(e))
            return ContentMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _assess_financial_relevance(self, title: str, content: str) -> float:
        """Assess financial relevance using FinBERT and keyword analysis"""
        try:
            # Combine title and content for analysis
            combined_text = f"{title}. {content}"
            
            # FinBERT-based relevance scoring
            finbert_scores = self.finbert_classifier(combined_text[:512])  # Limit token length
            
            # Extract sentiment strength as relevance indicator
            if finbert_scores and len(finbert_scores) > 0:
                scores = finbert_scores[0]
                # Strong sentiment (positive or negative) indicates financial relevance
                max_score = max(score['score'] for score in scores)
                finbert_relevance = max_score
            else:
                finbert_relevance = 0.0
            
            # Keyword-based relevance scoring
            keyword_relevance = self._calculate_keyword_relevance(combined_text)
            
            # Pattern-based relevance (financial events, numbers, etc.)
            pattern_relevance = self._calculate_pattern_relevance(combined_text)
            
            # Weighted combination
            relevance_score = (
                finbert_relevance * 0.5 +
                keyword_relevance * 0.3 +
                pattern_relevance * 0.2
            )
            
            return min(1.0, relevance_score)
            
        except Exception as e:
            logger.warning("Financial relevance assessment failed", error=str(e))
            return 0.5
    
    def _calculate_keyword_relevance(self, text: str) -> float:
        """Calculate financial relevance based on keyword density and importance"""
        text_lower = text.lower()
        word_count = len(text.split())
        
        if word_count == 0:
            return 0.0
        
        # Calculate weighted keyword score
        keyword_score = 0.0
        for keyword, weight in self.financial_keywords.items():
            keyword_clean = keyword.replace('_', ' ')
            occurrences = text_lower.count(keyword_clean)
            keyword_score += occurrences * weight
        
        # Normalize by document length
        keyword_density = keyword_score / word_count
        
        # Apply sigmoid function to get 0-1 score
        relevance = 1 / (1 + math.exp(-5 * (keyword_density - 0.02)))
        
        return min(1.0, relevance)
    
    def _calculate_pattern_relevance(self, text: str) -> float:
        """Calculate relevance based on financial patterns and structures"""
        pattern_score = 0.0
        
        # Financial number patterns
        financial_numbers = [
            r'\$[\d,]+(?:\.\d{2})?',  # Currency
            r'\d+(?:\.\d+)?%',  # Percentages
            r'\d+(?:\.\d+)?[BMK]?',  # Large numbers with suffixes
            r'Q[1-4]\s+\d{4}',  # Quarters
            r'FY\s*\d{4}',  # Fiscal years
        ]
        
        for pattern in financial_numbers:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            pattern_score += matches * 0.1
        
        # Financial event patterns
        financial_events = [
            r'earnings\s+(?:report|call|release)',
            r'(?:beats?|misses?)\s+(?:estimates?|expectations?)',
            r'(?:quarterly|annual)\s+results?',
            r'(?:dividend|stock)\s+(?:increase|cut|split)',
            r'merger|acquisition|buyout|takeover',
            r'IPO|initial\s+public\s+offering',
        ]
        
        for pattern in financial_events:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            pattern_score += matches * 0.2
        
        # Normalize score
        return min(1.0, pattern_score / 10.0)
    
    def _assess_content_quality(self, metrics: ContentMetrics, title: str, content: str) -> float:
        """Assess overall content quality based on structure and composition"""
        quality_score = 0.0
        
        # Word count scoring
        if metrics.word_count >= self.quality_thresholds['min_word_count']:
            if metrics.word_count <= self.quality_thresholds['max_word_count']:
                quality_score += 0.3  # Ideal length
            else:
                quality_score += 0.2  # Too long
        else:
            quality_score += 0.1  # Too short
        
        # Structure scoring
        if metrics.sentence_count >= self.quality_thresholds['min_sentences']:
            quality_score += 0.2
        
        if metrics.paragraph_count >= self.quality_thresholds['min_paragraphs']:
            quality_score += 0.15
        
        # Sentence length scoring
        if metrics.avg_sentence_length <= self.quality_thresholds['max_avg_sentence_length']:
            quality_score += 0.1
        
        # Content richness scoring
        if metrics.numerical_data_count >= 3:
            quality_score += 0.1
        
        if metrics.quote_count >= 1:
            quality_score += 0.05
        
        # Financial terms density
        if metrics.financial_terms_count >= self.quality_thresholds['min_financial_terms']:
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _assess_source_credibility(self, source_url: str, author: Optional[str] = None) -> float:
        """Assess source credibility based on domain and author"""
        try:
            # Extract domain from URL
            from urllib.parse import urlparse
            domain = urlparse(source_url).netloc.lower()
            domain = domain.replace('www.', '')
            
            # Look up in credibility database
            if domain in self.source_credibility_db:
                credibility_info = self.source_credibility_db[domain]
                base_score = credibility_info.base_credibility
                
                # Author boost (if known reputable author)
                author_boost = 0.0
                if author and len(author) > 5:  # Has meaningful author
                    author_boost = 0.05
                
                return min(1.0, base_score + author_boost)
            
            # Unknown source - neutral credibility
            return 0.5
            
        except Exception as e:
            logger.warning("Source credibility assessment failed", error=str(e))
            return 0.5
    
    def _assess_timeliness(self, publish_date: Optional[datetime]) -> float:
        """Assess timeliness with decay function"""
        if not publish_date:
            return 0.5  # Unknown publish date gets neutral score
        
        try:
            now = datetime.now(timezone.utc)
            if publish_date.tzinfo is None:
                publish_date = publish_date.replace(tzinfo=timezone.utc)
            
            age_hours = (now - publish_date).total_seconds() / 3600
            
            # Full score for recent articles
            if age_hours <= self.quality_thresholds['max_hours_for_full_timeliness']:
                return 1.0
            
            # Exponential decay function
            decay_days = self.quality_thresholds['timeliness_decay_days']
            age_days = age_hours / 24
            
            # Score decays exponentially over time
            timeliness = math.exp(-age_days / decay_days)
            
            return max(0.1, timeliness)  # Minimum score of 0.1
            
        except Exception as e:
            logger.warning("Timeliness assessment failed", error=str(e))
            return 0.5
    
    def _assess_structure_quality(self, metrics: ContentMetrics) -> float:
        """Assess article structure quality"""
        structure_score = 0.0
        
        # Paragraph structure
        if 2 <= metrics.paragraph_count <= 15:
            structure_score += 0.3
        elif metrics.paragraph_count > 15:
            structure_score += 0.2
        else:
            structure_score += 0.1
        
        # Sentence structure
        if 15 <= metrics.avg_sentence_length <= 25:
            structure_score += 0.3  # Ideal sentence length
        elif 10 <= metrics.avg_sentence_length <= 30:
            structure_score += 0.2  # Acceptable
        else:
            structure_score += 0.1  # Too short or too long
        
        # Content balance
        sentences_per_paragraph = metrics.sentence_count / max(1, metrics.paragraph_count)
        if 3 <= sentences_per_paragraph <= 6:
            structure_score += 0.2
        else:
            structure_score += 0.1
        
        # Data inclusion
        if metrics.numerical_data_count > 0:
            structure_score += 0.1
        
        if metrics.quote_count > 0:
            structure_score += 0.1
        
        return min(1.0, structure_score)
    
    def _assess_originality(self, title: str, content: str) -> float:
        """Assess content originality (placeholder for future enhancement)"""
        # This is a simplified originality assessment
        # In a full implementation, this would check against a database of known articles
        
        originality_score = 0.8  # Default assumption of originality
        
        # Check for common boilerplate patterns
        boilerplate_patterns = [
            r'this\s+is\s+a\s+breaking\s+news',
            r'developing\s+story',
            r'more\s+details\s+to\s+follow',
            r'stay\s+tuned\s+for\s+updates'
        ]
        
        content_lower = content.lower()
        for pattern in boilerplate_patterns:
            if re.search(pattern, content_lower):
                originality_score -= 0.1
        
        return max(0.3, originality_score)
    
    def _normalize_readability(self, flesch_score: float) -> float:
        """Normalize Flesch reading ease score to 0-1 scale"""
        if flesch_score <= 0:
            return 0.0
        elif flesch_score >= 100:
            return 1.0
        else:
            # Convert Flesch score (0-100) to normalized score
            # Optimal readability is around 60-70 Flesch score
            if 60 <= flesch_score <= 70:
                return 1.0
            elif 40 <= flesch_score <= 80:
                return 0.8
            elif 20 <= flesch_score <= 90:
                return 0.6
            else:
                return 0.4
    
    def _assess_factual_density(self, metrics: ContentMetrics, content: str) -> float:
        """Assess density of factual information"""
        if metrics.word_count == 0:
            return 0.0
        
        # Count factual elements
        factual_elements = (
            metrics.numerical_data_count +  # Numbers, percentages, currencies
            metrics.quote_count +  # Direct quotes
            len(re.findall(r'\b\d{4}\b', content)) +  # Years
            len(re.findall(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b', content, re.IGNORECASE)) +  # Months
            metrics.financial_terms_count / 3  # Financial terms (weighted)
        )
        
        # Calculate density
        density = factual_elements / metrics.word_count * 100
        
        # Normalize to 0-1 scale
        return min(1.0, density / 5.0)  # 5% density = full score
    
    def _identify_quality_factors(self, metrics: ContentMetrics, financial_relevance: float,
                                source_credibility: float, timeliness: float,
                                structure_score: float) -> Tuple[List[str], List[str]]:
        """Identify specific quality issues and strengths"""
        issues = []
        strengths = []
        
        # Check for issues
        if metrics.word_count < self.quality_thresholds['min_word_count']:
            issues.append(f"Article too short ({metrics.word_count} words)")
        
        if metrics.word_count > self.quality_thresholds['max_word_count']:
            issues.append(f"Article too long ({metrics.word_count} words)")
        
        if metrics.sentence_count < self.quality_thresholds['min_sentences']:
            issues.append("Too few sentences")
        
        if metrics.paragraph_count < self.quality_thresholds['min_paragraphs']:
            issues.append("Poor paragraph structure")
        
        if financial_relevance < 0.3:
            issues.append("Low financial relevance")
        
        if source_credibility < 0.6:
            issues.append("Unknown or low-credibility source")
        
        if timeliness < 0.3:
            issues.append("Article is outdated")
        
        if metrics.financial_terms_count < self.quality_thresholds['min_financial_terms']:
            issues.append("Insufficient financial content")
        
        # Check for strengths
        if financial_relevance >= 0.8:
            strengths.append("High financial relevance")
        
        if source_credibility >= 0.85:
            strengths.append("High-credibility source")
        
        if timeliness >= 0.9:
            strengths.append("Very recent news")
        
        if metrics.numerical_data_count >= 5:
            strengths.append("Rich in numerical data")
        
        if metrics.quote_count >= 2:
            strengths.append("Contains multiple quotes")
        
        if structure_score >= 0.8:
            strengths.append("Well-structured content")
        
        return issues, strengths
    
    def _calculate_assessment_confidence(self, metrics: ContentMetrics, 
                                       financial_relevance: float,
                                       source_credibility: float) -> float:
        """Calculate confidence in the quality assessment"""
        confidence_factors = []
        
        # Word count factor
        if metrics.word_count >= 200:
            confidence_factors.append(0.9)
        elif metrics.word_count >= 100:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Financial relevance factor
        if financial_relevance >= 0.7:
            confidence_factors.append(0.9)
        elif financial_relevance >= 0.5:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Source credibility factor
        if source_credibility >= 0.8:
            confidence_factors.append(0.9)
        elif source_credibility >= 0.6:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.6)
        
        # Content richness factor
        content_richness = (metrics.numerical_data_count + metrics.quote_count) / 10.0
        if content_richness >= 0.5:
            confidence_factors.append(0.8)
        elif content_richness >= 0.3:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.6)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def batch_assess_quality(self, articles: List[Dict]) -> List[Tuple[Dict, QualityScore]]:
        """
        Batch assess quality for multiple articles.
        
        Args:
            articles: List of article dictionaries with required fields
            
        Returns:
            List of (article, QualityScore) tuples
        """
        results = []
        
        for article in articles:
            try:
                quality_score = self.assess_article_quality(
                    title=article.get('title', ''),
                    content=article.get('content', ''),
                    source_url=article.get('source_url', ''),
                    publish_date=article.get('publish_date'),
                    author=article.get('author')
                )
                results.append((article, quality_score))
                
            except Exception as e:
                logger.error("Batch quality assessment failed for article",
                           title=article.get('title', 'unknown'), error=str(e))
                results.append((article, QualityScore(
                    overall_score=0.0, financial_relevance=0.0, content_quality=0.0,
                    source_credibility=0.5, timeliness_score=0.5, structure_score=0.0,
                    originality_score=0.5, readability_score=0.5, factual_density=0.0,
                    confidence=0.0, quality_issues=["assessment_failed"], quality_strengths=[]
                )))
        
        return results
    
    def get_quality_summary(self, quality_scores: List[QualityScore]) -> Dict:
        """Get summary statistics for a batch of quality scores"""
        if not quality_scores:
            return {}
        
        overall_scores = [qs.overall_score for qs in quality_scores]
        
        return {
            'count': len(quality_scores),
            'average_score': np.mean(overall_scores),
            'median_score': np.median(overall_scores),
            'std_score': np.std(overall_scores),
            'high_quality_count': sum(1 for score in overall_scores if score >= 0.8),
            'low_quality_count': sum(1 for score in overall_scores if score <= 0.4),
            'avg_financial_relevance': np.mean([qs.financial_relevance for qs in quality_scores]),
            'avg_source_credibility': np.mean([qs.source_credibility for qs in quality_scores]),
            'avg_timeliness': np.mean([qs.timeliness_score for qs in quality_scores]),
            'common_issues': Counter([issue for qs in quality_scores for issue in qs.quality_issues]).most_common(5),
            'common_strengths': Counter([strength for qs in quality_scores for strength in qs.quality_strengths]).most_common(5)
        }


# Utility functions for easy integration
def assess_single_article(title: str, content: str, source_url: str, 
                         publish_date: Optional[datetime] = None) -> QualityScore:
    """Convenience function to assess a single article"""
    assessor = ArticleQualityAssessment()
    return assessor.assess_article_quality(title, content, source_url, publish_date)


def filter_high_quality_articles(articles: List[Dict], min_score: float = 0.7) -> List[Tuple[Dict, QualityScore]]:
    """Filter articles by minimum quality score"""
    assessor = ArticleQualityAssessment()
    results = assessor.batch_assess_quality(articles)
    
    return [(article, score) for article, score in results if score.overall_score >= min_score]


if __name__ == "__main__":
    # Test the quality assessment system
    assessor = ArticleQualityAssessment()
    
    # Test article
    test_article = {
        'title': 'Apple Reports Record Q4 Earnings, Beats Revenue Expectations',
        'content': '''
        Apple Inc. reported record fourth-quarter earnings of $1.46 per share, significantly beating
        analyst expectations of $1.39. The tech giant's revenue reached $89.5 billion for the quarter,
        representing a 4.5% increase year-over-year and surpassing the consensus estimate of $89.3 billion.
        
        "We're pleased to report another record-breaking quarter," said CEO Tim Cook during the earnings call.
        "Our strong performance was driven by robust iPhone sales, continued growth in our services business,
        and expanding market share in emerging economies."
        
        iPhone revenue totaled $43.8 billion, up 3.5% from the previous year, while services revenue grew
        16% to $22.3 billion. The company also announced a quarterly dividend of $0.24 per share and
        authorized an additional $90 billion share repurchase program.
        
        Looking ahead, Apple provided guidance for Q1 2024 revenue between $89-93 billion, slightly above
        analyst projections. The company expects continued momentum from the iPhone 15 launch and growing
        adoption of Apple Intelligence features.
        ''',
        'source_url': 'https://www.reuters.com/business/apple-earnings-q4-2023',
        'publish_date': datetime.now(timezone.utc) - timedelta(hours=2),
        'author': 'Jane Smith'
    }
    
    # Assess quality
    quality_score = assessor.assess_article_quality(
        test_article['title'],
        test_article['content'],
        test_article['source_url'],
        test_article['publish_date'],
        test_article['author']
    )
    
    print("Article Quality Assessment Results:")
    print("=" * 50)
    print(f"Overall Score: {quality_score.overall_score:.3f}")
    print(f"Financial Relevance: {quality_score.financial_relevance:.3f}")
    print(f"Content Quality: {quality_score.content_quality:.3f}")
    print(f"Source Credibility: {quality_score.source_credibility:.3f}")
    print(f"Timeliness: {quality_score.timeliness_score:.3f}")
    print(f"Structure Score: {quality_score.structure_score:.3f}")
    print(f"Readability: {quality_score.readability_score:.3f}")
    print(f"Factual Density: {quality_score.factual_density:.3f}")
    print(f"Confidence: {quality_score.confidence:.3f}")
    
    print("\nQuality Issues:")
    for issue in quality_score.quality_issues:
        print(f"  - {issue}")
    
    print("\nQuality Strengths:")
    for strength in quality_score.quality_strengths:
        print(f"  + {strength}")
    
    # Test batch assessment
    print(f"\nTesting batch assessment...")
    articles = [test_article] * 3  # Test with 3 copies
    results = assessor.batch_assess_quality(articles)
    summary = assessor.get_quality_summary([score for _, score in results])
    
    print(f"Batch Summary: {summary}")