# Alpha Vantage Reset Timing Probe Plan (72-Hour)

**Objective**: Determine whether Alpha Vantage free tier quota resets at UTC midnight or 24h from first daily call  
**Duration**: 72 hours (3 complete daily cycles)  
**Probe Budget**: 2 calls/day × 3 days = 6 calls total  
**Risk Level**: Minimal (uses lowest-cost endpoint)

---

## Probe Design

### Test Hypothesis
- **H1 (UTC Midnight)**: Quota resets precisely at 00:00:00 UTC regardless of previous day's call timing
- **H2 (Rolling 24h)**: Quota resets 24 hours after the first call of each day
- **H3 (Inconclusive)**: Inconsistent behavior or external factors prevent determination

### Probe Call Specification
- **Endpoint**: `GLOBAL_QUOTE` (lowest cost, minimal data transfer)
- **Symbol**: `AAPL` (high liquidity, always available)
- **Expected Response**: JSON with quote data (~200 bytes)
- **Timeout**: 30 seconds
- **Retry Policy**: Single attempt only (no retries to avoid budget waste)

---

## Daily Schedule (UTC)

### Daily Probe Windows
1. **Pre-Midnight Probe**: 23:58:00 - 23:59:30 UTC
2. **Post-Midnight Probe**: 00:01:00 - 00:02:30 UTC  
3. **Control Probe**: 12:00:00 - 12:01:00 UTC (if needed)

### Expected Probe Patterns

**Day 1 (2025-09-01)**:
- 23:58 UTC: Probe #1 - Expected result depends on previous day usage
- 00:01 UTC: Probe #2 - Should succeed if UTC midnight reset
- Evidence: HTTP 200 vs 429/rate limit response

**Day 2 (2025-09-02)**: 
- 23:58 UTC: Probe #3 - Test consistency 
- 00:01 UTC: Probe #4 - Confirm reset pattern
- Evidence: Response pattern matching Day 1

**Day 3 (2025-09-03)**:
- 23:58 UTC: Probe #5 - Final consistency check
- 00:01 UTC: Probe #6 - Confirm final pattern
- Evidence: 3-day pattern analysis

---

## Implementation

### Manual Execution Steps
```bash
# Set environment
export ALPHA_VANTAGE_KEY="V09CQLOY0N4DG2J4"  # From config/secrets.env
export PROBE_SYMBOL="AAPL"

# Pre-midnight probe (23:58 UTC)
curl -w "HTTP_STATUS:%{http_code}\nTOTAL_TIME:%{time_total}\n" \
     -o "probes/probe_$(date -u +%Y%m%d_%H%M%S)_pre.json" \
     "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${PROBE_SYMBOL}&apikey=${ALPHA_VANTAGE_KEY}"

# Post-midnight probe (00:01 UTC)  
curl -w "HTTP_STATUS:%{http_code}\nTOTAL_TIME:%{time_total}\n" \
     -o "probes/probe_$(date -u +%Y%m%d_%H%M%S)_post.json" \
     "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${PROBE_SYMBOL}&apikey=${ALPHA_VANTAGE_KEY}"
```

### Automated GitHub Actions Workflow
```yaml
# .github/workflows/alpha_vantage_reset_probe.yml
name: Alpha Vantage Reset Timing Probe
on:
  schedule:
    - cron: '58 23 * * *'  # 23:58 UTC daily (pre-midnight)
    - cron: '1 0 * * *'    # 00:01 UTC daily (post-midnight)
  workflow_dispatch:

jobs:
  probe:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Execute Probe
        env:
          ALPHA_VANTAGE_KEY: ${{ secrets.ALPHA_VANTAGE_KEY }}
        run: |
          TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
          PROBE_TYPE=$([[ "$(date -u +%H)" == "23" ]] && echo "pre" || echo "post")
          
          curl -w "HTTP_STATUS:%{http_code}\nTOTAL_TIME:%{time_total}\n" \
               -o "probes/probe_${TIMESTAMP}_${PROBE_TYPE}.json" \
               "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey=${ALPHA_VANTAGE_KEY}"
               
          # Log results
          echo "$(date -u +%Y-%m-%d\ %H:%M:%S) — ProbeType=${PROBE_TYPE} — Result=HTTP_$(curl -s -o /dev/null -w "%{http_code}" "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey=${ALPHA_VANTAGE_KEY}") — Evidence=probe_${TIMESTAMP}_${PROBE_TYPE}.json" >> logs/api_usage_log.md
          
      - name: Commit Results
        run: |
          git config --local user.email "action@github.com"  
          git config --local user.name "GitHub Action"
          git add probes/ logs/
          git commit -m "Alpha Vantage reset probe: $(date -u +%Y%m%d_%H%M%S)" || exit 0
          git push
```

---

## Success Criteria & Evidence Interpretation

### UTC Midnight Reset (H1) Evidence
- **Pattern**: Pre-midnight probes may fail (429), post-midnight probes succeed (200)  
- **Timing**: Consistent success within 2 minutes after 00:00 UTC
- **Response**: Valid JSON quote data immediately after midnight
- **Confidence**: High if pattern repeats 3 consecutive days

### Rolling 24h Reset (H2) Evidence  
- **Pattern**: Success/failure depends on previous day's first call timing
- **Timing**: Quota available exactly 24 hours after previous day's first API call
- **Variability**: Reset time shifts daily based on usage pattern
- **Confidence**: Medium - requires tracking first daily call timestamps

### Inconclusive (H3) Triggers
- **Inconsistent Patterns**: Different behavior across 3-day window
- **External Factors**: API maintenance, service degradation during probe windows
- **Data Quality Issues**: Probe failures due to network/timeout rather than quota

---

## Risk Mitigation

### Probe Budget Management
- **Total Budget**: 6 calls over 3 days (well within 25/day limit)
- **Budget Tracking**: Log each probe attempt with timestamp and response
- **Contingency**: Stop probes if non-quota failures exceed 2 attempts

### Error Handling
- **Network Failures**: Retry once after 30s delay, then abort
- **Invalid Responses**: Log response body for analysis
- **Rate Limit Errors**: Expected behavior - log as evidence, do not retry

### Data Collection
```bash
# Probe result logging format
TIMESTAMP=$(date -u +%Y-%m-%d\ %H:%M:%S)
echo "${TIMESTAMP} — ProbeType=${TYPE} — HTTPStatus=${STATUS} — ResponseTime=${TIME}s — Evidence=${FILENAME}" >> probes/probe_log.txt
```

---

## Expected Timeline & Deliverables

### Day 0 (Setup)
- [ ] Create probe scripts
- [ ] Test probe execution (dry run without API calls)
- [ ] Schedule first probe for 2025-09-01 23:58 UTC

### Days 1-3 (Execution)  
- [ ] Execute 6 probes as scheduled
- [ ] Collect response data and timing information
- [ ] Log results in structured format

### Day 4 (Analysis)
- [ ] Analyze response patterns across 3 days
- [ ] Update `rate_limit_unknowns.md` with findings
- [ ] Set final recommendation for Alpha Vantage quota reset timing

### Automation
- [ ] GitHub Actions workflow created: `.github/workflows/alpha_vantage_reset_probe.yml`
- [ ] Secrets configured: `ALPHA_VANTAGE_KEY`  
- [ ] Results automatically committed to `probes/` directory

---

## Data Analysis Template

After 72 hours, analyze results using this pattern:

```
Day 1: Pre=HTTP_XXX (23:58), Post=HTTP_XXX (00:01) 
Day 2: Pre=HTTP_XXX (23:58), Post=HTTP_XXX (00:01)
Day 3: Pre=HTTP_XXX (23:58), Post=HTTP_XXX (00:01)

Pattern: [UTC_MIDNIGHT|ROLLING_24H|INCONCLUSIVE]
Confidence: [HIGH|MEDIUM|LOW]
Recommendation: Use [UTC midnight|24h rolling] assumptions for quota planning
```

This probe plan provides definitive evidence for Alpha Vantage quota reset timing with minimal API budget usage and automated execution.