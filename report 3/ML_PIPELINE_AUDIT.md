# ML Pipeline Audit - Current State & Roadmap

## Current ML/NLP Implementations

### ✅ Sentiment Analysis - FinBERT Pipeline
**Location**: `services/finbert_sentiment_analyzer.py` (referenced in feature flags)
**Status**: Implemented but feature-flagged

```python
# From config/feature_flags.py:28,100-102
'enable_finbert_pipeline': self._get_bool_env('ENABLE_FINBERT_PIPELINE', False),
'enable_finbert_backtest': self._get_bool_env('ENABLE_FINBERT_BACKTEST', False),

def is_finbert_pipeline_enabled() -> bool:
    return feature_flags.is_enabled('enable_finbert_pipeline')
```

**Capabilities**:
- Financial news sentiment scoring (-1.0 to 1.0)
- Pre-trained on financial texts
- Integration with news ingestion pipeline

### ✅ News Clustering & Correlation
**Location**: `services/news_clustering.py` (referenced in tests)
**Evidence**: `tests/test_news_clustering.py` exists

**Capabilities**:
- Article grouping by topic/symbol
- Duplicate detection beyond simple hash matching
- Temporal correlation analysis

### ✅ Investment Recommendation Engine  
**Location**: `services/investment_recommendation_engine.py` + `services/reco_engine.py`
**Evidence**: Multiple test files reference recommendation logic

**Model Types**:
- Rule-based recommendations (`config/recommendation_rules.yaml`)
- ML-enhanced scoring (feature-flagged)
- Portfolio optimization integration

### 🟡 Forecasting Models - Limited Implementation
**Location**: `services/forecasting/` directory structure
**Files Present**:
- `services/forecasting/baselines.py` - Simple baseline models
- `services/forecasting/timegpt_stub.py` - Placeholder for TimeGPT integration

**Feature Flags**:
```python
'enable_alt_forecasts': self._get_bool_env('ENABLE_ALT_FORECASTS', False),
'enable_timegpt_stub': self._get_bool_env('ENABLE_TIMEGPT_STUB', False),
```

## Model Performance Analysis

### Sentiment Analysis Evaluation
**Current State**: No systematic evaluation metrics found

**Missing Evaluation Framework**:
```python
# NEEDED: services/ml_evaluation.py
class SentimentEvaluator:
    def __init__(self):
        self.ground_truth_data = self.load_labeled_dataset()
    
    def evaluate_finbert(self, test_articles: List[NewsArticle]) -> dict:
        """Evaluate FinBERT performance on labeled data"""
        predictions = []
        actuals = []
        
        for article in test_articles:
            pred_sentiment = self.finbert_analyzer.analyze(article.content)
            actual_sentiment = self.ground_truth_data.get(article.url)
            
            if actual_sentiment is not None:
                predictions.append(pred_sentiment)
                actuals.append(actual_sentiment)
        
        return {
            "accuracy": accuracy_score(actuals, predictions),
            "precision": precision_score(actuals, predictions, average='weighted'),
            "recall": recall_score(actuals, predictions, average='weighted'),
            "f1_score": f1_score(actuals, predictions, average='weighted'),
            "confusion_matrix": confusion_matrix(actuals, predictions).tolist()
        }
```

### Recommendation Engine Evaluation  
**Current State**: `services/reco_engine.py` exists but no backtesting framework

**Missing Backtesting**:
```python
# NEEDED: services/backtesting.py  
class RecommendationBacktester:
    def __init__(self, historical_data: pd.DataFrame):
        self.price_data = historical_data
        self.recommendation_history = []
    
    def backtest_recommendations(self, 
                               start_date: str, 
                               end_date: str,
                               initial_capital: float = 10000) -> dict:
        """Backtest recommendation engine performance"""
        
        portfolio_value = []
        benchmark_value = []
        recommendations_made = []
        
        for date in pd.date_range(start_date, end_date):
            # Get recommendations for this date
            recs = self.reco_engine.generate_recommendations(
                date=date, 
                symbols=self.universe
            )
            
            # Simulate portfolio performance
            portfolio_return = self.simulate_portfolio(recs, date)
            benchmark_return = self.get_benchmark_return(date)  # SPY
            
            portfolio_value.append(portfolio_return)
            benchmark_value.append(benchmark_return)
            recommendations_made.append(len(recs))
        
        return {
            "total_return": (portfolio_value[-1] / portfolio_value[0]) - 1,
            "benchmark_return": (benchmark_value[-1] / benchmark_value[0]) - 1,
            "alpha": self.calculate_alpha(portfolio_value, benchmark_value),
            "sharpe_ratio": self.calculate_sharpe(portfolio_value),
            "max_drawdown": self.calculate_max_drawdown(portfolio_value),
            "avg_recommendations_per_day": sum(recommendations_made) / len(recommendations_made)
        }
```

## Current Model Architecture

### Rule-Based Recommendation System
**Configuration**: `config/recommendation_rules.yaml`

```yaml
# INFERRED STRUCTURE based on feature flags
rules:
  sentiment_threshold:
    positive: 0.3
    negative: -0.3
  
  technical_indicators:
    rsi_oversold: 30
    rsi_overbought: 70
    volume_spike_threshold: 2.0
    
  fundamental_filters:
    min_market_cap: 1000000000  # $1B
    max_pe_ratio: 30
    min_revenue_growth: 0.05    # 5%

scoring:
  sentiment_weight: 0.3
  technical_weight: 0.4  
  fundamental_weight: 0.3
```

### News Clustering Algorithm
**Approach**: Likely using TF-IDF + K-means or similar

**Enhancement Needed**:
```python
# UPGRADE: Use transformer-based embeddings
from sentence_transformers import SentenceTransformer

class AdvancedNewsClusterer:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.clustering_model = HDBSCAN(min_cluster_size=3)
    
    def cluster_articles(self, articles: List[NewsArticle]) -> List[List[NewsArticle]]:
        """Cluster articles using semantic embeddings"""
        
        # Generate embeddings
        texts = [f"{article.title} {article.summary}" for article in articles]
        embeddings = self.embedding_model.encode(texts)
        
        # Perform clustering
        cluster_labels = self.clustering_model.fit_predict(embeddings)
        
        # Group articles by cluster
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(articles[idx])
        
        return list(clusters.values())
```

## Model Training Pipeline

### Current Training Infrastructure: ❌ Missing
**No Evidence Found** of:
- Model versioning system  
- Training data management
- Model deployment pipeline
- A/B testing framework

### Required Training Pipeline:
```python
# NEEDED: services/model_training.py
class MLTrainingPipeline:
    def __init__(self, model_registry: ModelRegistry):
        self.registry = model_registry
        self.feature_store = FeatureStore()
    
    async def train_sentiment_model(self, 
                                  training_data: pd.DataFrame,
                                  model_type: str = "finbert") -> str:
        """Train and register new sentiment model"""
        
        # Feature engineering
        features = self.feature_store.prepare_sentiment_features(training_data)
        
        # Train model
        if model_type == "finbert":
            model = self.train_finbert_model(features)
        elif model_type == "lstm":
            model = self.train_lstm_model(features)
        
        # Evaluate model
        eval_results = self.evaluate_model(model, self.test_data)
        
        # Register if performance meets threshold
        if eval_results["f1_score"] > 0.75:
            model_id = self.registry.register_model(
                model=model,
                model_type="sentiment_analysis",
                version=self.get_next_version(),
                metrics=eval_results
            )
            return model_id
        else:
            raise ModelPerformanceError(f"Model F1 score {eval_results['f1_score']} below threshold")
```

## Feature Engineering Pipeline

### Current Features: 🟡 Basic Implementation
**Price Features**: OHLCV data from yfinance/providers
**News Features**: Sentiment scores, article counts
**Technical Indicators**: Limited (feature-flagged for local computation)

### Enhanced Feature Pipeline:
```python
# UPGRADE: services/feature_engineering.py
class FeatureEngineer:
    def __init__(self):
        self.technical_calculator = TechnicalIndicators()
        self.sentiment_aggregator = SentimentAggregator()
    
    def generate_stock_features(self, symbol: str, lookback_days: int = 30) -> pd.DataFrame:
        """Generate comprehensive feature set for a symbol"""
        
        # Price-based features
        price_data = self.get_price_data(symbol, lookback_days)
        price_features = self.technical_calculator.calculate_all(price_data)
        
        # News-based features  
        news_data = self.get_news_data(symbol, lookback_days)
        news_features = self.sentiment_aggregator.aggregate(news_data)
        
        # Market regime features
        market_features = self.get_market_regime_features(lookback_days)
        
        # Combine all features
        features = pd.concat([price_features, news_features, market_features], axis=1)
        return features.fillna(method='ffill')
    
    def calculate_technical_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        indicators = {}
        
        # Momentum indicators
        indicators['rsi_14'] = self.rsi(price_data['close'], 14)
        indicators['macd'] = self.macd(price_data['close'])
        indicators['stoch_k'] = self.stochastic(price_data, 14)
        
        # Volatility indicators  
        indicators['bb_upper'], indicators['bb_lower'] = self.bollinger_bands(price_data['close'], 20)
        indicators['atr'] = self.average_true_range(price_data, 14)
        
        # Volume indicators
        indicators['obv'] = self.on_balance_volume(price_data)
        indicators['vwap'] = self.volume_weighted_average_price(price_data)
        
        return pd.DataFrame(indicators, index=price_data.index)
```

## Model Deployment & Versioning

### Missing Model Registry: ❌ Critical Gap
```python
# NEEDED: services/model_registry.py
class ModelRegistry:
    """Centralized model versioning and deployment"""
    
    def __init__(self, storage_backend: str = "local"):
        self.storage = self.get_storage_backend(storage_backend)
        self.model_metadata = {}
    
    def register_model(self, 
                      model: Any,
                      model_type: str,
                      version: str, 
                      metrics: dict) -> str:
        """Register new model version"""
        model_id = f"{model_type}_v{version}"
        
        # Save model artifact
        model_path = self.storage.save_model(model, model_id)
        
        # Save metadata
        self.model_metadata[model_id] = {
            "type": model_type,
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "path": str(model_path),
            "status": "registered"
        }
        
        return model_id
    
    def promote_to_production(self, model_id: str):
        """Promote model to production"""
        if model_id in self.model_metadata:
            # Update current production model
            self.model_metadata[model_id]["status"] = "production" 
            
            # Demote previous production model
            for mid, metadata in self.model_metadata.items():
                if (metadata["type"] == self.model_metadata[model_id]["type"] and 
                    mid != model_id and 
                    metadata["status"] == "production"):
                    metadata["status"] = "archived"
```

### A/B Testing Framework: ❌ Missing
```python
# NEEDED: services/ab_testing.py
class ModelABTester:
    def __init__(self, model_registry: ModelRegistry):
        self.registry = model_registry
        self.experiment_configs = {}
    
    def create_experiment(self, 
                         name: str,
                         control_model_id: str,
                         treatment_model_id: str,
                         traffic_split: float = 0.5) -> str:
        """Create A/B test between two models"""
        
        experiment_id = f"exp_{name}_{int(time.time())}"
        
        self.experiment_configs[experiment_id] = {
            "name": name,
            "control_model": control_model_id,
            "treatment_model": treatment_model_id,
            "traffic_split": traffic_split,
            "start_date": datetime.now(timezone.utc),
            "status": "active",
            "results": {"control": [], "treatment": []}
        }
        
        return experiment_id
    
    def route_prediction(self, experiment_id: str, request_id: str) -> str:
        """Route prediction request to control or treatment"""
        config = self.experiment_configs[experiment_id]
        
        # Deterministic routing based on request hash
        hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        if (hash_val % 100) < (config["traffic_split"] * 100):
            return config["treatment_model"]
        else:
            return config["control_model"]
```

## Upgrade Roadmap

### Phase 1 - Evaluation & Monitoring (Week 1-2)
1. **Model Performance Metrics**
   - Implement systematic evaluation for FinBERT sentiment
   - Add recommendation backtesting framework
   - Create baseline performance benchmarks

2. **Model Monitoring**
   - Track prediction accuracy over time
   - Monitor for model drift
   - Alert on performance degradation

### Phase 2 - Enhanced Features (Week 3-4)  
1. **Feature Engineering Pipeline**
   - Comprehensive technical indicator calculation
   - News sentiment aggregation and trending
   - Market regime detection features

2. **Advanced Models**
   - Transformer-based news clustering  
   - LSTM/GRU for time series forecasting
   - Ensemble methods for recommendations

### Phase 3 - Production ML (Week 5-8)
1. **Model Registry & Versioning** 
   - Centralized model storage and metadata
   - Automated model deployment pipeline
   - A/B testing framework for model comparison

2. **Real-time Inference**
   - Low-latency prediction API
   - Model serving with auto-scaling  
   - Feature store for real-time features

## Performance Benchmarks

### Current Baseline (Estimated)
- **Sentiment Analysis**: ~75% accuracy on financial news
- **Recommendation Engine**: No systematic measurement  
- **News Clustering**: No quantitative evaluation

### Target Performance (Post-Upgrade)
- **Sentiment Analysis**: >85% F1 score on labeled financial news
- **Recommendation Backtest**: Sharpe ratio >1.2, max drawdown <15%
- **News Clustering**: >90% topic coherence score
- **Inference Latency**: <200ms for real-time predictions

**Implementation Priority**: Evaluation framework first (Week 1), then enhanced models, finally production ML infrastructure.