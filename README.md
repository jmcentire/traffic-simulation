# Traffic Flow Optimization: Per-Lane Speed Ranges

An agent-based simulation demonstrating that **per-lane speed ranges with enforced minimums** produces dramatically better traffic flow than traditional single speed limits.

## The Hypothesis

Traffic congestion isn't caused by high speeds—it's caused by **speed variance**. The slow driver in the fast lane creates cascading brake events and forces constant lane changes.

**Proposed solution:**
- Non-overlapping speed ranges per lane (55-65, 65-75, 75-85, 85-95 mph)
- Harsh enforcement of **minimums** (ticket slow drivers, tolerate fast drivers)
- License and vehicle certification tiers for higher-speed lane access

## Results (Bay Area Projections)

| Metric | Current | Proposed | Change |
|--------|---------|----------|--------|
| Throughput | 1,175/hr | 1,574/hr | **+34%** |
| Lane changes/car | 7.49 | 0.46 | **-94%** |
| Accident risk | 9.06/1000 | 1.11/1000 | **-84%** |
| Effective MPG | 17.7 | 27.1 | **+53%** |

### Annual Impact (SF Bay Area)
- **$3.3 billion** in benefits (time, fuel, safety)
- **$1.8 billion** in new government revenue
- **334 million gallons** of fuel saved
- **3.3 million tons** of CO2 reduced
- **84%** fewer accidents
- **54x ROI** on $62M implementation cost

## Files

- `traffic_model.py` - Basic simulation comparing baseline vs proposed
- `traffic_model_v2.py` - Realistic model with mid-road exits and imperfect compliance
- `traffic_results.png` - Visualization of simulation results
- `linkedin_post.md` - Full writeup with revenue model and policy analysis

## Running the Simulation

```bash
# Basic comparison
python3 traffic_model.py

# Realistic model with visualization
python3 traffic_model_v2.py
```

Requires: `numpy`, `matplotlib`

## The Mechanism

1. **Self-selection**: Drivers sort by desired speed at entry
2. **Structural enforcement**: Can't enter faster lane below its minimum
3. **Natural discipline**: "Accelerate then merge left" becomes mandatory
4. **Reduced variance**: Within-lane speeds are uniform
5. **Fewer lane changes**: 94% reduction eliminates accordion effect

## Revenue Model

The proposal **increases** government revenue:
- Un-speeding tickets (slow-traffic cameras): $660M/year
- License tier fees: $688M/year
- Vehicle certification: $488M/year
- **Total: $1.8B/year** (vs $280M current)

Plus creates $206M driving school market.

## Why It Works

Accidents correlate with **speed differential**, not absolute speed. The current system optimizes for the wrong variable. A left lane where everyone goes 80 is safer than a left lane with cars going 60, 70, and 80 mixed together.

## Author

Jeremy McEntire

## License

MIT - Take it, use it, implement it. Credit appreciated but not required.
