"""
Traffic Flow Simulation: Testing Per-Lane Speed Ranges
Hypothesis: Non-overlapping speed ranges with harsh minimum enforcement
produces near-optimal flow from simple rules.

Author: Jeremy McEntire
Date: 2025-12-29
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

@dataclass
class Car:
    id: int
    desired_speed: float  # What the driver wants to go (mph)
    current_speed: float  # What they're actually going
    position: float       # Position on road (miles)
    lane: int            # Current lane (0 = rightmost/slowest)

    def __hash__(self):
        return self.id


@dataclass
class LaneConfig:
    min_speed: float  # Minimum allowed (enforced harshly)
    max_speed: float  # Maximum allowed (enforced loosely)


class TrafficSimulation:
    def __init__(
        self,
        num_lanes: int,
        lane_configs: List[LaneConfig],
        road_length: float = 10.0,  # miles
        dt: float = 1/3600,  # time step in hours (1 second)
        min_following_distance: float = 0.02,  # miles (~100 feet)
        entry_rate: float = 1800,  # cars per hour entering
        desired_speed_mean: float = 72,  # mph
        desired_speed_std: float = 10,   # mph
        enforce_minimums: bool = True,
        minimum_violation_penalty: float = 0.95,  # slow down harshly if below min
    ):
        self.num_lanes = num_lanes
        self.lane_configs = lane_configs
        self.road_length = road_length
        self.dt = dt
        self.min_following_distance = min_following_distance
        self.entry_rate = entry_rate
        self.desired_speed_mean = desired_speed_mean
        self.desired_speed_std = desired_speed_std
        self.enforce_minimums = enforce_minimums
        self.minimum_violation_penalty = minimum_violation_penalty

        self.cars: List[Car] = []
        self.next_car_id = 0
        self.time = 0.0

        # Metrics
        self.cars_completed = 0
        self.total_travel_time = 0.0
        self.entry_times = {}  # car_id -> entry_time
        self.speed_samples = []
        self.lane_change_count = 0
        self.congestion_events = 0  # times a car had to brake significantly

    def get_best_entry_lane(self, desired_speed: float) -> int:
        """Find the appropriate lane for a car's desired speed."""
        for i, config in enumerate(self.lane_configs):
            if config.min_speed <= desired_speed <= config.max_speed:
                return i
            # If desired speed is below all lanes, use slowest
            if desired_speed < config.min_speed and i == 0:
                return 0
        # If desired speed exceeds all lanes, use fastest
        return self.num_lanes - 1

    def get_car_ahead(self, car: Car) -> Optional[Car]:
        """Find the car immediately ahead in the same lane."""
        cars_in_lane = [c for c in self.cars if c.lane == car.lane and c.position > car.position]
        if not cars_in_lane:
            return None
        return min(cars_in_lane, key=lambda c: c.position)

    def get_car_ahead_in_lane(self, position: float, lane: int) -> Optional[Car]:
        """Find the car immediately ahead in a specific lane."""
        cars_in_lane = [c for c in self.cars if c.lane == lane and c.position > position]
        if not cars_in_lane:
            return None
        return min(cars_in_lane, key=lambda c: c.position)

    def get_car_behind_in_lane(self, position: float, lane: int) -> Optional[Car]:
        """Find the car immediately behind in a specific lane."""
        cars_in_lane = [c for c in self.cars if c.lane == lane and c.position < position]
        if not cars_in_lane:
            return None
        return max(cars_in_lane, key=lambda c: c.position)

    def can_change_to_lane(self, car: Car, target_lane: int) -> bool:
        """Check if lane change is safe."""
        if target_lane < 0 or target_lane >= self.num_lanes:
            return False

        # Check if there's room
        car_ahead = self.get_car_ahead_in_lane(car.position, target_lane)
        car_behind = self.get_car_behind_in_lane(car.position, target_lane)

        if car_ahead and (car_ahead.position - car.position) < self.min_following_distance * 1.5:
            return False
        if car_behind and (car.position - car_behind.position) < self.min_following_distance * 1.5:
            return False

        return True

    def should_change_lane(self, car: Car) -> Optional[int]:
        """
        Determine if car should change lanes and which direction.
        Returns target lane or None.

        Key insight: With enforced minimums, you MUST be in the right lane
        for your speed. This naturally enforces "accelerate then merge left"
        and "merge right then decelerate".
        """
        current_config = self.lane_configs[car.lane]

        # Check if we're violating minimum (must move right)
        if self.enforce_minimums and car.desired_speed < current_config.min_speed:
            if car.lane > 0 and self.can_change_to_lane(car, car.lane - 1):
                return car.lane - 1

        # Check if we could go faster in a left lane
        if car.lane < self.num_lanes - 1:
            left_config = self.lane_configs[car.lane + 1]
            # Only move left if our desired speed fits that lane
            if (left_config.min_speed <= car.desired_speed <= left_config.max_speed * 1.1):
                car_ahead = self.get_car_ahead(car)
                # Move left if blocked or if it's our proper lane
                if car_ahead and (car_ahead.position - car.position) < self.min_following_distance * 2:
                    if self.can_change_to_lane(car, car.lane + 1):
                        return car.lane + 1
                # Move left if we're below our desired in current lane but could achieve it left
                if car.current_speed < car.desired_speed * 0.9:
                    if self.can_change_to_lane(car, car.lane + 1):
                        return car.lane + 1

        # Check if we should move right (going slower than lane expects)
        if car.lane > 0:
            right_config = self.lane_configs[car.lane - 1]
            if car.desired_speed <= right_config.max_speed:
                if self.can_change_to_lane(car, car.lane - 1):
                    return car.lane - 1

        return None

    def update_car_speed(self, car: Car):
        """Update car's speed based on conditions."""
        config = self.lane_configs[car.lane]
        car_ahead = self.get_car_ahead(car)

        # Target speed is desired, clamped to lane limits
        if self.enforce_minimums:
            target_speed = max(config.min_speed, min(car.desired_speed, config.max_speed * 1.1))
        else:
            target_speed = min(car.desired_speed, config.max_speed * 1.1)

        # But must slow down if car ahead is too close
        if car_ahead:
            gap = car_ahead.position - car.position
            if gap < self.min_following_distance:
                # Emergency braking
                target_speed = car_ahead.current_speed * 0.8
                self.congestion_events += 1
            elif gap < self.min_following_distance * 2:
                # Match speed of car ahead
                target_speed = min(target_speed, car_ahead.current_speed)
            elif gap < self.min_following_distance * 3:
                # Gradually slow if approaching
                if car.current_speed > car_ahead.current_speed:
                    target_speed = min(target_speed, car_ahead.current_speed * 1.05)

        # Smooth speed changes (can't instantly change speed)
        max_accel = 10  # mph per second
        max_decel = 15  # mph per second (can brake faster than accelerate)

        speed_diff = target_speed - car.current_speed
        if speed_diff > 0:
            car.current_speed += min(speed_diff, max_accel * (self.dt * 3600))
        else:
            car.current_speed += max(speed_diff, -max_decel * (self.dt * 3600))

        car.current_speed = max(0, car.current_speed)

    def step(self):
        """Advance simulation by one time step."""
        self.time += self.dt

        # Maybe spawn new car
        if random.random() < self.entry_rate * self.dt:
            desired_speed = np.random.normal(self.desired_speed_mean, self.desired_speed_std)
            desired_speed = max(45, min(95, desired_speed))  # Clamp to reasonable range

            entry_lane = self.get_best_entry_lane(desired_speed)

            # Check if there's room to enter
            car_ahead = self.get_car_ahead_in_lane(0, entry_lane)
            if car_ahead is None or car_ahead.position > self.min_following_distance * 2:
                new_car = Car(
                    id=self.next_car_id,
                    desired_speed=desired_speed,
                    current_speed=min(desired_speed, self.lane_configs[entry_lane].max_speed),
                    position=0.0,
                    lane=entry_lane
                )
                self.cars.append(new_car)
                self.entry_times[new_car.id] = self.time
                self.next_car_id += 1

        # Update each car
        for car in self.cars:
            # Check for lane change
            target_lane = self.should_change_lane(car)
            if target_lane is not None:
                car.lane = target_lane
                self.lane_change_count += 1

            # Update speed
            self.update_car_speed(car)

            # Update position
            car.position += car.current_speed * self.dt

            # Sample speed for metrics
            if random.random() < 0.01:  # Sample 1% of updates
                self.speed_samples.append(car.current_speed)

        # Remove cars that completed the road
        completed = [c for c in self.cars if c.position >= self.road_length]
        for car in completed:
            self.cars_completed += 1
            travel_time = self.time - self.entry_times[car.id]
            self.total_travel_time += travel_time
            del self.entry_times[car.id]

        self.cars = [c for c in self.cars if c.position < self.road_length]

    def run(self, duration_hours: float = 1.0):
        """Run simulation for specified duration."""
        steps = int(duration_hours / self.dt)
        for _ in range(steps):
            self.step()

    def get_metrics(self) -> dict:
        """Return simulation metrics."""
        avg_travel_time = self.total_travel_time / self.cars_completed if self.cars_completed > 0 else 0
        avg_speed = np.mean(self.speed_samples) if self.speed_samples else 0
        speed_std = np.std(self.speed_samples) if self.speed_samples else 0

        # Theoretical minimum travel time (road_length / desired_speed_mean)
        theoretical_min = self.road_length / self.desired_speed_mean
        efficiency = theoretical_min / avg_travel_time if avg_travel_time > 0 else 0

        return {
            'cars_completed': self.cars_completed,
            'throughput_per_hour': self.cars_completed / (self.time) if self.time > 0 else 0,
            'avg_travel_time_minutes': avg_travel_time * 60,
            'avg_speed_mph': avg_speed,
            'speed_std_mph': speed_std,
            'efficiency': efficiency,
            'lane_changes': self.lane_change_count,
            'lane_changes_per_car': self.lane_change_count / self.cars_completed if self.cars_completed > 0 else 0,
            'congestion_events': self.congestion_events,
        }


def run_comparison():
    """Compare baseline vs proposed lane rules."""

    # Simulation parameters
    num_lanes = 3
    duration = 1.0  # 1 hour
    entry_rate = 2400  # cars per hour (high traffic)

    print("=" * 60)
    print("TRAFFIC FLOW SIMULATION")
    print("Testing: Per-Lane Speed Ranges with Minimum Enforcement")
    print("=" * 60)

    # Baseline: Traditional single speed limit (65 mph), no enforced minimum
    print("\n--- BASELINE: Single Speed Limit (65 mph), No Minimums ---")
    baseline_configs = [
        LaneConfig(min_speed=0, max_speed=65),   # Right lane
        LaneConfig(min_speed=0, max_speed=65),   # Middle lane
        LaneConfig(min_speed=0, max_speed=65),   # Left lane
    ]

    baseline_sim = TrafficSimulation(
        num_lanes=num_lanes,
        lane_configs=baseline_configs,
        entry_rate=entry_rate,
        enforce_minimums=False,
    )
    baseline_sim.run(duration)
    baseline_metrics = baseline_sim.get_metrics()

    print(f"  Throughput: {baseline_metrics['throughput_per_hour']:.0f} cars/hour")
    print(f"  Avg Travel Time: {baseline_metrics['avg_travel_time_minutes']:.2f} minutes")
    print(f"  Avg Speed: {baseline_metrics['avg_speed_mph']:.1f} mph")
    print(f"  Speed Std Dev: {baseline_metrics['speed_std_mph']:.1f} mph")
    print(f"  Efficiency: {baseline_metrics['efficiency']*100:.1f}%")
    print(f"  Lane Changes/Car: {baseline_metrics['lane_changes_per_car']:.2f}")
    print(f"  Congestion Events: {baseline_metrics['congestion_events']}")

    # Proposed: Non-overlapping speed ranges with enforced minimums
    print("\n--- PROPOSED: Non-Overlapping Ranges, Enforced Minimums ---")
    proposed_configs = [
        LaneConfig(min_speed=55, max_speed=65),  # Right lane: 55-65
        LaneConfig(min_speed=65, max_speed=75),  # Middle lane: 65-75
        LaneConfig(min_speed=75, max_speed=85),  # Left lane: 75-85
    ]

    proposed_sim = TrafficSimulation(
        num_lanes=num_lanes,
        lane_configs=proposed_configs,
        entry_rate=entry_rate,
        enforce_minimums=True,
    )
    proposed_sim.run(duration)
    proposed_metrics = proposed_sim.get_metrics()

    print(f"  Throughput: {proposed_metrics['throughput_per_hour']:.0f} cars/hour")
    print(f"  Avg Travel Time: {proposed_metrics['avg_travel_time_minutes']:.2f} minutes")
    print(f"  Avg Speed: {proposed_metrics['avg_speed_mph']:.1f} mph")
    print(f"  Speed Std Dev: {proposed_metrics['speed_std_mph']:.1f} mph")
    print(f"  Efficiency: {proposed_metrics['efficiency']*100:.1f}%")
    print(f"  Lane Changes/Car: {proposed_metrics['lane_changes_per_car']:.2f}")
    print(f"  Congestion Events: {proposed_metrics['congestion_events']}")

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    throughput_change = (proposed_metrics['throughput_per_hour'] - baseline_metrics['throughput_per_hour']) / baseline_metrics['throughput_per_hour'] * 100
    travel_time_change = (proposed_metrics['avg_travel_time_minutes'] - baseline_metrics['avg_travel_time_minutes']) / baseline_metrics['avg_travel_time_minutes'] * 100
    lane_change_reduction = (baseline_metrics['lane_changes_per_car'] - proposed_metrics['lane_changes_per_car']) / baseline_metrics['lane_changes_per_car'] * 100 if baseline_metrics['lane_changes_per_car'] > 0 else 0
    congestion_reduction = (baseline_metrics['congestion_events'] - proposed_metrics['congestion_events']) / baseline_metrics['congestion_events'] * 100 if baseline_metrics['congestion_events'] > 0 else 0

    print(f"  Throughput Change: {throughput_change:+.1f}%")
    print(f"  Travel Time Change: {travel_time_change:+.1f}%")
    print(f"  Lane Change Reduction: {lane_change_reduction:.1f}%")
    print(f"  Congestion Event Reduction: {congestion_reduction:.1f}%")

    return baseline_metrics, proposed_metrics


def run_sensitivity_analysis():
    """Test how the proposed system performs under varying traffic loads."""

    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS: Varying Traffic Load")
    print("=" * 60)

    entry_rates = [1200, 1800, 2400, 3000, 3600]  # cars per hour

    baseline_results = []
    proposed_results = []

    proposed_configs = [
        LaneConfig(min_speed=55, max_speed=65),
        LaneConfig(min_speed=65, max_speed=75),
        LaneConfig(min_speed=75, max_speed=85),
    ]

    baseline_configs = [
        LaneConfig(min_speed=0, max_speed=65),
        LaneConfig(min_speed=0, max_speed=65),
        LaneConfig(min_speed=0, max_speed=65),
    ]

    for rate in entry_rates:
        # Baseline
        sim = TrafficSimulation(
            num_lanes=3,
            lane_configs=baseline_configs,
            entry_rate=rate,
            enforce_minimums=False,
        )
        sim.run(0.5)  # 30 minutes each for speed
        baseline_results.append(sim.get_metrics())

        # Proposed
        sim = TrafficSimulation(
            num_lanes=3,
            lane_configs=proposed_configs,
            entry_rate=rate,
            enforce_minimums=True,
        )
        sim.run(0.5)
        proposed_results.append(sim.get_metrics())

    print(f"\n{'Entry Rate':<12} {'Baseline Throughput':<20} {'Proposed Throughput':<20} {'Improvement':<12}")
    print("-" * 64)

    for i, rate in enumerate(entry_rates):
        b_tp = baseline_results[i]['throughput_per_hour']
        p_tp = proposed_results[i]['throughput_per_hour']
        improvement = (p_tp - b_tp) / b_tp * 100 if b_tp > 0 else 0
        print(f"{rate:<12} {b_tp:<20.0f} {p_tp:<20.0f} {improvement:+.1f}%")

    return entry_rates, baseline_results, proposed_results


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Run main comparison
    baseline, proposed = run_comparison()

    # Run sensitivity analysis
    rates, baseline_sens, proposed_sens = run_sensitivity_analysis()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
The simulation tests the hypothesis that:
1. Non-overlapping per-lane speed RANGES (not just limits)
2. Harsh enforcement of MINIMUMS (not just maximums)
3. Tolerance of exceeding maximums

...produces emergent efficient flow by:
- Forcing proper lane selection based on desired speed
- Naturally enforcing "accelerate THEN merge left"
- Naturally enforcing "merge right THEN decelerate"
- Reducing unnecessary lane changes
- Eliminating slow-car-in-fast-lane congestion cascades
""")
