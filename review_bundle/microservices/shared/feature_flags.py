"""
Feature Flag Management System with A/B Testing
Integration with LaunchDarkly/Unleash for enterprise features
"""

import json
import time
import hashlib
import random
from typing import Dict, Any, Optional, List, Union, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

import redis
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class RolloutStrategy(Enum):
    """Rollout strategies for feature flags"""
    IMMEDIATE = "immediate"
    GRADUAL = "gradual"
    USER_SEGMENT = "user_segment"
    A_B_TEST = "a_b_test"
    CANARY = "canary"


class UserSegment(Enum):
    """User segmentation categories"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    PREMIUM = "premium"
    FREE = "free"
    BETA_TESTER = "beta_tester"
    HIGH_VOLUME = "high_volume"


@dataclass
class User:
    """User context for feature flag evaluation"""
    user_id: str
    risk_profile: UserSegment
    account_type: UserSegment
    subscription_tier: str
    region: str
    signup_date: datetime
    portfolio_value: float
    trading_frequency: str
    beta_tester: bool = False
    custom_attributes: Optional[Dict[str, Any]] = None


@dataclass
class VariationConfig:
    """Configuration for A/B test variations"""
    name: str
    description: str
    weight: float  # Percentage allocation (0-100)
    config: Dict[str, Any]
    is_control: bool = False


@dataclass
class FeatureFlagConfig:
    """Feature flag configuration"""
    name: str
    description: str
    enabled: bool
    rollout_strategy: RolloutStrategy
    rollout_percentage: float
    target_segments: List[UserSegment]
    variations: List[VariationConfig]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    prerequisites: Optional[List[str]] = None
    custom_rules: Optional[List[Dict[str, Any]]] = None


class ExperimentResult(BaseModel):
    """A/B test experiment result"""
    flag_name: str
    variation: str
    user_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    conversion: bool = False


class FeatureFlagProvider(ABC):
    """Abstract base class for feature flag providers"""
    
    @abstractmethod
    async def get_flag_value(self, flag_name: str, user: User, default_value: Any = False) -> Any:
        """Get feature flag value for user"""
        pass
    
    @abstractmethod
    async def get_variation(self, flag_name: str, user: User) -> Optional[str]:
        """Get A/B test variation for user"""
        pass
    
    @abstractmethod
    async def track_event(self, flag_name: str, user: User, event: str, value: float = 1.0):
        """Track event for analytics"""
        pass


class LocalFeatureFlagProvider(FeatureFlagProvider):
    """Local feature flag provider for development/testing"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.flags: Dict[str, FeatureFlagConfig] = {}
        self.user_assignments: Dict[str, Dict[str, str]] = {}  # user_id -> {flag_name: variation}
        
    async def register_flag(self, flag_config: FeatureFlagConfig):
        """Register a new feature flag"""
        self.flags[flag_config.name] = flag_config
        
        # Store in Redis for persistence
        await self._store_flag_config(flag_config)
        
        logger.info("Feature flag registered", 
                   flag_name=flag_config.name,
                   strategy=flag_config.rollout_strategy.value)
    
    async def get_flag_value(self, flag_name: str, user: User, default_value: Any = False) -> Any:
        """Get feature flag value for user"""
        try:
            flag_config = await self._get_flag_config(flag_name)
            if not flag_config or not flag_config.enabled:
                return default_value
            
            # Check prerequisites
            if flag_config.prerequisites:
                for prereq in flag_config.prerequisites:
                    if not await self.get_flag_value(prereq, user, False):
                        logger.debug("Prerequisite not met", flag=flag_name, prerequisite=prereq)
                        return default_value
            
            # Check date constraints
            now = datetime.utcnow()
            if flag_config.start_date and now < flag_config.start_date:
                return default_value
            if flag_config.end_date and now > flag_config.end_date:
                return default_value
            
            # Evaluate rollout strategy
            if flag_config.rollout_strategy == RolloutStrategy.IMMEDIATE:
                return await self._evaluate_immediate(flag_config, user)
            elif flag_config.rollout_strategy == RolloutStrategy.GRADUAL:
                return await self._evaluate_gradual(flag_config, user)
            elif flag_config.rollout_strategy == RolloutStrategy.USER_SEGMENT:
                return await self._evaluate_user_segment(flag_config, user)
            elif flag_config.rollout_strategy == RolloutStrategy.A_B_TEST:
                variation = await self.get_variation(flag_name, user)
                return await self._get_variation_config(flag_config, variation)
            elif flag_config.rollout_strategy == RolloutStrategy.CANARY:
                return await self._evaluate_canary(flag_config, user)
            
            return default_value
            
        except Exception as e:
            logger.error("Error evaluating feature flag", 
                        flag_name=flag_name, 
                        user_id=user.user_id, 
                        error=str(e))
            return default_value
    
    async def get_variation(self, flag_name: str, user: User) -> Optional[str]:
        """Get A/B test variation for user"""
        try:
            # Check if user already has an assignment
            user_key = f"user_assignment:{user.user_id}:{flag_name}"
            existing_assignment = await self.redis.get(user_key)
            
            if existing_assignment:
                return existing_assignment.decode('utf-8')
            
            flag_config = await self._get_flag_config(flag_name)
            if not flag_config or not flag_config.variations:
                return None
            
            # Calculate hash for consistent assignment
            hash_input = f"{flag_name}:{user.user_id}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            percentage = hash_value % 100
            
            # Assign variation based on weights
            cumulative_weight = 0
            for variation in flag_config.variations:
                cumulative_weight += variation.weight
                if percentage < cumulative_weight:
                    # Store assignment for consistency
                    await self.redis.setex(user_key, 86400 * 30, variation.name)  # 30 days TTL
                    
                    logger.info("User assigned to variation",
                               user_id=user.user_id,
                               flag_name=flag_name,
                               variation=variation.name)
                    return variation.name
            
            # Fallback to control
            control_variation = next((v for v in flag_config.variations if v.is_control), flag_config.variations[0])
            await self.redis.setex(user_key, 86400 * 30, control_variation.name)
            return control_variation.name
            
        except Exception as e:
            logger.error("Error getting variation", 
                        flag_name=flag_name, 
                        user_id=user.user_id, 
                        error=str(e))
            return None
    
    async def track_event(self, flag_name: str, user: User, event: str, value: float = 1.0):
        """Track event for analytics"""
        try:
            variation = await self.get_variation(flag_name, user)
            if not variation:
                return
            
            event_data = {
                'flag_name': flag_name,
                'variation': variation,
                'user_id': user.user_id,
                'event': event,
                'value': value,
                'timestamp': datetime.utcnow().isoformat(),
                'user_segment': user.risk_profile.value,
                'account_type': user.account_type.value
            }
            
            # Store in Redis for analytics processing
            event_key = f"ab_test_events:{flag_name}:{datetime.utcnow().strftime('%Y-%m-%d')}"
            await self.redis.lpush(event_key, json.dumps(event_data))
            await self.redis.expire(event_key, 86400 * 90)  # 90 days retention
            
            logger.info("A/B test event tracked",
                       flag_name=flag_name,
                       variation=variation,
                       event=event,
                       user_id=user.user_id)
            
        except Exception as e:
            logger.error("Error tracking event", error=str(e))
    
    async def _get_flag_config(self, flag_name: str) -> Optional[FeatureFlagConfig]:
        """Get flag configuration from cache or Redis"""
        if flag_name in self.flags:
            return self.flags[flag_name]
        
        # Try Redis
        config_key = f"feature_flag_config:{flag_name}"
        config_data = await self.redis.get(config_key)
        if config_data:
            config_dict = json.loads(config_data.decode('utf-8'))
            flag_config = self._deserialize_flag_config(config_dict)
            self.flags[flag_name] = flag_config
            return flag_config
        
        return None
    
    async def _store_flag_config(self, flag_config: FeatureFlagConfig):
        """Store flag configuration in Redis"""
        config_key = f"feature_flag_config:{flag_config.name}"
        config_data = self._serialize_flag_config(flag_config)
        await self.redis.setex(config_key, 86400 * 7, json.dumps(config_data))  # 7 days TTL
    
    def _serialize_flag_config(self, flag_config: FeatureFlagConfig) -> Dict[str, Any]:
        """Serialize flag config to dictionary"""
        config_dict = asdict(flag_config)
        config_dict['rollout_strategy'] = flag_config.rollout_strategy.value
        config_dict['target_segments'] = [seg.value for seg in flag_config.target_segments]
        
        if flag_config.start_date:
            config_dict['start_date'] = flag_config.start_date.isoformat()
        if flag_config.end_date:
            config_dict['end_date'] = flag_config.end_date.isoformat()
        
        return config_dict
    
    def _deserialize_flag_config(self, config_dict: Dict[str, Any]) -> FeatureFlagConfig:
        """Deserialize flag config from dictionary"""
        # Convert enums and dates back
        config_dict['rollout_strategy'] = RolloutStrategy(config_dict['rollout_strategy'])
        config_dict['target_segments'] = [UserSegment(seg) for seg in config_dict['target_segments']]
        
        if config_dict.get('start_date'):
            config_dict['start_date'] = datetime.fromisoformat(config_dict['start_date'])
        if config_dict.get('end_date'):
            config_dict['end_date'] = datetime.fromisoformat(config_dict['end_date'])
        
        # Convert variations
        variations = []
        for var_dict in config_dict['variations']:
            variations.append(VariationConfig(**var_dict))
        config_dict['variations'] = variations
        
        return FeatureFlagConfig(**config_dict)
    
    async def _evaluate_immediate(self, flag_config: FeatureFlagConfig, user: User) -> bool:
        """Evaluate immediate rollout strategy"""
        return True
    
    async def _evaluate_gradual(self, flag_config: FeatureFlagConfig, user: User) -> bool:
        """Evaluate gradual rollout strategy"""
        hash_input = f"{flag_config.name}:{user.user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        percentage = hash_value % 100
        return percentage < flag_config.rollout_percentage
    
    async def _evaluate_user_segment(self, flag_config: FeatureFlagConfig, user: User) -> bool:
        """Evaluate user segment rollout strategy"""
        return (user.risk_profile in flag_config.target_segments or 
                user.account_type in flag_config.target_segments)
    
    async def _evaluate_canary(self, flag_config: FeatureFlagConfig, user: User) -> bool:
        """Evaluate canary rollout strategy"""
        # Canary for beta testers and high-value users first
        if user.beta_tester or user.portfolio_value > 100000:
            return await self._evaluate_gradual(flag_config, user)
        return False
    
    async def _get_variation_config(self, flag_config: FeatureFlagConfig, variation_name: Optional[str]) -> Any:
        """Get configuration for a specific variation"""
        if not variation_name:
            return False
        
        variation = next((v for v in flag_config.variations if v.name == variation_name), None)
        if not variation:
            return False
        
        return variation.config


class FeatureFlagManager:
    """Main feature flag manager with A/B testing capabilities"""
    
    def __init__(self, provider: FeatureFlagProvider, redis_client: redis.Redis):
        self.provider = provider
        self.redis = redis_client
        self.logger = structlog.get_logger(__name__)
        
        # Gradual rollout percentages
        self.gradual_rollout_stages = [1, 5, 10, 25, 50, 75, 100]
        
    async def initialize_default_flags(self):
        """Initialize default feature flags for the system"""
        default_flags = [
            # Recommendation Algorithm A/B Test
            FeatureFlagConfig(
                name="recommendation_algorithm_v2",
                description="A/B test for new recommendation algorithm with enhanced ML features",
                enabled=True,
                rollout_strategy=RolloutStrategy.A_B_TEST,
                rollout_percentage=50.0,
                target_segments=[UserSegment.MODERATE, UserSegment.AGGRESSIVE],
                variations=[
                    VariationConfig(
                        name="control",
                        description="Current recommendation algorithm",
                        weight=50.0,
                        config={"algorithm_version": "v1", "ml_features": False},
                        is_control=True
                    ),
                    VariationConfig(
                        name="enhanced_ml",
                        description="Enhanced ML recommendation algorithm",
                        weight=50.0,
                        config={"algorithm_version": "v2", "ml_features": True, "feature_count": 50}
                    )
                ]
            ),
            
            # Real-time notifications
            FeatureFlagConfig(
                name="real_time_notifications",
                description="Real-time push notifications for price alerts",
                enabled=True,
                rollout_strategy=RolloutStrategy.GRADUAL,
                rollout_percentage=25.0,
                target_segments=[UserSegment.PREMIUM, UserSegment.HIGH_VOLUME],
                variations=[]
            ),
            
            # Advanced charting
            FeatureFlagConfig(
                name="advanced_charting",
                description="Advanced charting features with technical indicators",
                enabled=True,
                rollout_strategy=RolloutStrategy.USER_SEGMENT,
                rollout_percentage=100.0,
                target_segments=[UserSegment.PREMIUM],
                variations=[]
            ),
            
            # Portfolio optimization
            FeatureFlagConfig(
                name="portfolio_optimization",
                description="AI-powered portfolio optimization suggestions",
                enabled=True,
                rollout_strategy=RolloutStrategy.CANARY,
                rollout_percentage=10.0,
                target_segments=[UserSegment.AGGRESSIVE, UserSegment.PREMIUM],
                variations=[]
            )
        ]
        
        for flag in default_flags:
            await self.provider.register_flag(flag)
    
    async def is_enabled(self, flag_name: str, user: User) -> bool:
        """Check if feature flag is enabled for user"""
        return await self.provider.get_flag_value(flag_name, user, False)
    
    async def get_config(self, flag_name: str, user: User, default_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get feature flag configuration for user"""
        config = await self.provider.get_flag_value(flag_name, user, default_config or {})
        return config if isinstance(config, dict) else {}
    
    async def get_variation(self, flag_name: str, user: User) -> Optional[str]:
        """Get A/B test variation for user"""
        return await self.provider.get_variation(flag_name, user)
    
    async def track_conversion(self, flag_name: str, user: User, conversion_value: float = 1.0):
        """Track conversion event for A/B testing"""
        await self.provider.track_event(flag_name, user, "conversion", conversion_value)
    
    async def track_custom_event(self, flag_name: str, user: User, event_name: str, value: float = 1.0):
        """Track custom event for A/B testing"""
        await self.provider.track_event(flag_name, user, event_name, value)
    
    async def gradual_rollout(self, flag_name: str, current_percentage: float) -> float:
        """Calculate next stage for gradual rollout"""
        next_stages = [stage for stage in self.gradual_rollout_stages if stage > current_percentage]
        return next_stages[0] if next_stages else current_percentage
    
    async def get_experiment_results(self, flag_name: str, days: int = 7) -> Dict[str, Any]:
        """Get A/B test experiment results"""
        try:
            results = {"variations": {}, "summary": {}}
            
            for day_offset in range(days):
                date = (datetime.utcnow() - timedelta(days=day_offset)).strftime('%Y-%m-%d')
                event_key = f"ab_test_events:{flag_name}:{date}"
                
                events = await self.redis.lrange(event_key, 0, -1)
                
                for event_data in events:
                    try:
                        event = json.loads(event_data.decode('utf-8'))
                        variation = event['variation']
                        
                        if variation not in results["variations"]:
                            results["variations"][variation] = {
                                "total_users": set(),
                                "events": {},
                                "conversions": 0,
                                "total_events": 0
                            }
                        
                        results["variations"][variation]["total_users"].add(event['user_id'])
                        results["variations"][variation]["total_events"] += 1
                        
                        event_type = event['event']
                        if event_type not in results["variations"][variation]["events"]:
                            results["variations"][variation]["events"][event_type] = 0
                        results["variations"][variation]["events"][event_type] += 1
                        
                        if event_type == "conversion":
                            results["variations"][variation]["conversions"] += event.get('value', 1.0)
                    
                    except json.JSONDecodeError:
                        continue
            
            # Calculate summary statistics
            for variation, data in results["variations"].items():
                user_count = len(data["total_users"])
                conversion_rate = (data["conversions"] / user_count) if user_count > 0 else 0
                
                results["variations"][variation] = {
                    "unique_users": user_count,
                    "total_events": data["total_events"],
                    "conversions": data["conversions"],
                    "conversion_rate": conversion_rate,
                    "events_breakdown": data["events"]
                }
            
            return results
            
        except Exception as e:
            self.logger.error("Error getting experiment results", error=str(e))
            return {"variations": {}, "summary": {}}


# Context manager for feature flag evaluation
class FeatureFlagContext:
    """Context manager for feature flag evaluation with automatic event tracking"""
    
    def __init__(self, flag_manager: FeatureFlagManager, flag_name: str, user: User):
        self.flag_manager = flag_manager
        self.flag_name = flag_name
        self.user = user
        self.enabled = False
        self.variation = None
        self.start_time = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        self.enabled = await self.flag_manager.is_enabled(self.flag_name, self.user)
        if self.enabled:
            self.variation = await self.flag_manager.get_variation(self.flag_name, self.user)
            await self.flag_manager.track_custom_event(self.flag_name, self.user, "feature_used")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self.start_time:
            duration = time.time() - self.start_time
            await self.flag_manager.track_custom_event(self.flag_name, self.user, "feature_duration", duration)
        
        if exc_type is not None:
            await self.flag_manager.track_custom_event(self.flag_name, self.user, "feature_error")