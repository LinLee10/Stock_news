# Prioritized Roadmap - 2 & 4 Week Implementation Plan

## 2-Week Plan (Critical Path)

### Week 1: Critical Bug Fixes & Foundation

#### Day 1-2: Alpha Vantage Infinite Loop Fix 🔴 URGENT
**Effort**: 8 hours  
**Owner**: Senior Engineer  
**Files to Touch**: 
- `services/alpha_vantage_manager.py` (lines 129-835)
- `tests/unit/test_alpha_vantage_loop_fix.py` (NEW)

**Tasks**:
1. Add daily exhaustion guard (`_daily_closed` flag)
2. Fix backoff re-queueing logic (lines 394-402) 
3. Add `DailyQuotaExhaustedError` exception handling
4. Bound rate limit enforcement waits (max 5 minutes)
5. Create reproduction test suite

**Success Criteria**: 
- No infinite loops under quota exhaustion
- CPU usage <10% during rate limiting
- All existing tests pass

#### Day 3-4: Finnhub Manager Implementation 🟡 HIGH
**Effort**: 12 hours  
**Owner**: Senior Engineer  
**Files to Touch**:
- `services/finnhub_manager.py` (NEW)
- `tests/unit/test_finnhub_manager.py` (NEW)
- `config/feature_flags.py` (add ENABLE_FINNHUB)

**Tasks**:
1. Implement async Finnhub client (60/min rate limit)
2. Add endpoints: `/company-news`, `/stock/social-sentiment`, `/quote`  
3. Integrate with existing audit logger
4. Add structured error handling and retries
5. Create unit tests with mocked responses

**Success Criteria**:
- 60/min rate limiting enforced
- Clean JSON response parsing  
- Graceful 429 handling with backoff

#### Day 5: Basic Deduplication 🟡 HIGH
**Effort**: 6 hours  
**Owner**: Senior Engineer  
**Files to Touch**:
- `services/deduplication.py` (NEW)
- `tests/unit/test_deduplication.py` (NEW)

**Tasks**:
1. Implement content hash-based news deduplication
2. Add URL normalization (remove tracking params)
3. Create merge logic for duplicate articles
4. Basic performance testing (1000 articles)

**Success Criteria**:
- >90% duplicate detection accuracy
- <1 second processing for 1000 articles
- Preserve best data from merged duplicates

### Week 2: Pipeline & Multi-Provider Integration

#### Day 8-10: Pipeline Runner 🟡 HIGH  
**Effort**: 16 hours
**Owner**: Senior Engineer
**Files to Touch**:
- `pipeline_runner.py` (NEW)
- `services/quota_manager.py` (NEW)
- `tests/integration/test_pipeline_runner.py` (NEW)

**Tasks**:
1. Create daily orchestration with 15-minute time budget
2. Implement quota pre-sizing (calculate max tasks before scheduling)
3. Add provider priority system (Portfolio > Watchlist > Research)
4. Integrate all existing providers (YFinance, NewsAPI, Alpha Vantage)
5. Add comprehensive logging and monitoring

**Success Criteria**:
- Completes within 15 minutes for 50-ticker universe
- Never exceeds any provider's daily quota
- Graceful fallback when providers fail

#### Day 11-12: Twelve Data Manager 🟡 MEDIUM
**Effort**: 10 hours
**Owner**: Senior Engineer  
**Files to Touch**:
- `services/twelve_data_manager.py` (NEW)
- `tests/unit/test_twelve_data_manager.py` (NEW)
- `config/feature_flags.py` (add USE_LOCAL_INDICATORS)

**Tasks**:
1. Implement 8/min, 800/day rate limiting
2. Add intraday and indicator endpoints  
3. Create Track A (remote indicators) vs Track B (local computation) logic
4. Pre-cap daily tasks to ≤800 with safety buffer
5. Integration with pipeline runner

**Success Criteria**:  
- 8/min rate limiting verified
- Daily task count never exceeds 720 (10% buffer)
- Local indicator computation as fallback

#### Day 13-14: Enhanced Monitoring & Cloud Prep 🟠 MEDIUM
**Effort**: 8 hours
**Owner**: Senior Engineer
**Files to Touch**:
- `services/audit_logger.py` (enhance per-provider counters)
- `deployment/cloud_run_config.yaml` (NEW)
- `deployment/github_actions_pipeline.yml` (NEW)

**Tasks**:
1. Add per-provider usage rollups to audit logger
2. Create cloud deployment configuration (Cloud Run/Lambda)
3. Set up GitHub Actions workflow for daily execution  
4. Add alert thresholds for quota exhaustion
5. Document deployment runbook

**Success Criteria**:
- Structured JSONL logging with rollups
- Cloud deployment ready with secrets management
- Automated daily scheduling configured

## 4-Week Plan (Complete System)

### Week 3: Additional Providers & Advanced Features

#### Day 15-17: Alpaca & Government Data Clients
**Effort**: 12 hours
**Files to Touch**:
- `services/alpaca_manager.py` (NEW)
- `services/macro/fred_client.py` (NEW)
- `services/macro/bea_client.py` (NEW)
- `services/macro/bls_client.py` (NEW)

**Tasks**:
1. Alpaca US intraday implementation (200/min, 15-min delay)
2. FRED economic indicators client (120/min, 100k/day)
3. BEA economic analysis client (1000/day)
4. BLS labor statistics client (25/day registered)
5. Integration with pipeline runner

#### Day 18-19: Advanced Deduplication & Content Analysis  
**Effort**: 10 hours
**Files to Touch**:
- `services/deduplication.py` (enhance)
- `services/content_similarity.py` (NEW)
- `services/symbol_extraction.py` (NEW)

**Tasks**:
1. Content similarity matching (fuzzy dedup)
2. Automatic symbol extraction from news content
3. Cross-provider precedence rules 
4. Performance optimization for large batches

#### Day 20-21: Multi-Key Compliance Implementation
**Effort**: 8 hours  
**Files to Touch**:
- `services/api_key_manager.py` (NEW)
- `services/quota_enforcer.py` (NEW)
- `services/compliance_auditor.py` (NEW)

**Tasks**:
1. Key routing based on usage context
2. Per-key usage ledgers with Redis persistence
3. Compliance audit trails and reporting
4. Vendor clarification for ambiguous ToS cases

### Week 4: Testing, Optimization & Production Readiness

#### Day 22-24: Comprehensive Testing Suite
**Effort**: 16 hours
**Files to Touch**:
- `tests/unit/test_rate_limit_comprehensive.py` (expand)
- `tests/integration/test_fallback_chains.py` (NEW)
- `tests/performance/test_concurrent_load.py` (NEW)
- `tests/e2e/test_full_pipeline.py` (NEW)

**Tasks**:
1. Rate limit edge case testing (all providers)
2. Fallback chain correctness verification
3. Load testing with concurrent requests
4. End-to-end pipeline testing with real quotas
5. Memory leak detection for long-running processes

#### Day 25-26: Performance Optimization  
**Effort**: 10 hours
**Files to Touch**:
- `services/caching_layer.py` (NEW)
- `services/batch_processor.py` (NEW)
- Multiple files for optimization

**Tasks**:
1. Redis caching layer for frequently accessed data
2. Batch processing optimization for news/prices
3. Database query optimization
4. Memory usage profiling and optimization
5. Response time benchmarking

#### Day 27-28: Production Deployment & Monitoring
**Effort**: 8 hours  
**Files to Touch**:
- `monitoring/dashboards.py` (NEW)
- `monitoring/alerting.py` (NEW)  
- `deployment/production_config.py` (NEW)

**Tasks**:
1. Production monitoring dashboards
2. Real-time alerting for quota/performance issues
3. Production deployment with zero-downtime
4. Backup and disaster recovery procedures
5. Documentation and runbooks

## Resource Allocation

### Personnel Requirements
- **1 Senior Engineer (Full-time)**: Core development and architecture
- **0.5 DevOps Engineer (Part-time)**: Cloud deployment and monitoring
- **0.25 QA Engineer (Part-time)**: Test strategy and edge case validation

### Infrastructure Costs
- **Cloud Run/Lambda**: ~$20-30/month
- **Redis Cache**: ~$10-15/month  
- **Monitoring & Logging**: ~$5-10/month
- **Total**: $35-55/month

### Risk Mitigation Timeline

| Week | Risk | Mitigation | Success Metric |
|------|------|------------|----------------|
| 1 | Alpha Vantage loop causes outage | Fix first, comprehensive testing | No infinite loops under load |
| 2 | Pipeline exceeds cloud time limits | Optimize batch sizes, parallel processing | <15 min completion time |
| 3 | New providers violate ToS | Conservative quotas, compliance tracking | Zero violations detected |
| 4 | Production deployment issues | Staged rollout, rollback procedures | <1 hour deployment time |

## Success Criteria by Week

### Week 2 Deliverables:
- [ ] Alpha Vantage infinite loop eliminated
- [ ] Finnhub news integration working  
- [ ] Basic deduplication reducing duplicates by >80%
- [ ] Pipeline runner orchestrating daily execution
- [ ] All existing functionality preserved

### Week 4 Deliverables:  
- [ ] 11+ API providers integrated with compliant quotas
- [ ] Multi-key strategy implemented where allowed
- [ ] Comprehensive test coverage (>85% critical path)
- [ ] Production-ready cloud deployment
- [ ] Full monitoring and alerting operational

## Maintenance & Evolution (Post-4 Weeks)

### Monthly Tasks:
- Review provider ToS for changes
- Analyze quota utilization and optimization opportunities  
- Update compliance documentation
- Performance benchmarking and optimization

### Quarterly Tasks:
- Evaluate new data provider opportunities
- Review and update test coverage
- Disaster recovery testing
- Cost optimization review

**Total Estimated Effort**: 140 hours over 4 weeks = 3.5 weeks of dedicated senior engineer time with some parallel tasks.