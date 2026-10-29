"""
Baseline ADAS algorithms for comparative evaluation against OverTakeGuard:
1. TTC-Only Warning Baseline (Simple Time-to-Collision threshold)
2. Naive Sensor Fusion Baseline (Simple union/distance heuristic without occlusion or abort awareness)
"""

from __future__ import annotations
from typing import List, Optional
from overtakeguard.models import (
    VehicleState,
    ManeuverPhase,
    AdvisoryLevel,
    AdvisoryOutput,
    DetectedTarget,
    AbortabilityState,
    SensorObservation
)


class TTCOnlyBaseline:
    """
    Industry-standard basic collision warning:
    - Only evaluates raw Time-to-Collision (TTC) to directly detected targets.
    - Blind to occlusion shadows / hidden oncoming cars behind large vehicles.
    - Unaware of abort feasibility or Point of No Return.
    """
    def __init__(self, ttc_threshold_s: float = 6.0):
        self.ttc_threshold = ttc_threshold_s

    def evaluate(
        self,
        timestamp: float,
        ego: VehicleState,
        raw_sensor_targets: List[DetectedTarget],
        current_phase: ManeuverPhase,
    ) -> AdvisoryOutput:
        oncoming = [
            t for t in raw_sensor_targets
            if (1.0 <= t.range_y <= 5.5) and (t.rel_vx < -10.0 or t.abs_vx <= 0.0) and t.range_x > 0
        ]

        min_ttc = 999.0
        crit_target = None
        for t in oncoming:
            closing_v = ego.vx - t.rel_vx
            if closing_v > 0.5:
                ttc = t.range_x / closing_v
                if ttc < min_ttc:
                    min_ttc = ttc
                    crit_target = t

        # Static Abortability (Naive assumption that abort is always feasible)
        dummy_abort = AbortabilityState(
            is_feasible=True,
            required_deceleration=2.0,
            time_to_abort=2.5,
            slot_clearance=15.0,
            point_of_no_return=False,
            reason="TTC-only baseline does not model abortability dynamics.",
        )

        if min_ttc < self.ttc_threshold:
            if current_phase in (ManeuverPhase.PASSING, ManeuverPhase.LANE_CHANGE_OUT):
                level = AdvisoryLevel.ABORT_NOW
                reason = f"TTC Baseline: TTC {min_ttc:.1f}s < threshold {self.ttc_threshold:.1f}s. Recommend abort."
            else:
                level = AdvisoryLevel.WAIT
                reason = f"TTC Baseline: Oncoming target detected with TTC {min_ttc:.1f}s < {self.ttc_threshold:.1f}s. Wait."
        else:
            level = AdvisoryLevel.SAFE_TO_OVERTAKE
            reason = f"TTC Baseline: No target detected below {self.ttc_threshold:.1f}s TTC. Safe."

        return AdvisoryOutput(
            timestamp=timestamp,
            phase=current_phase,
            level=level,
            primary_reason=reason,
            confidence=0.75,
            epistemic_uncertainty=0.0,  # Blind to epistemic uncertainty!
            aleatoric_uncertainty=0.1,
            time_to_complete=6.0,
            available_time_window=min_ttc,
            safety_margin_time=min_ttc - 6.0,
            safety_margin_distance=(min_ttc - 6.0) * ego.vx,
            abort_state=dummy_abort,
            occlusion_zones=[],
            detected_hazards=oncoming,
            sensor_trust_index=0.8,
        )


class NaiveFusionBaseline:
    """
    Naive Sensor Fusion:
    - Unions camera and radar targets without covariance weighting.
    - Simple static distance threshold (e.g., 200m).
    - Ignores occluded blind spots behind trucks.
    - Ignores dynamic abort deceleration limits.
    """
    def __init__(self, critical_distance_m: float = 200.0):
        self.critical_distance_m = critical_distance_m

    def evaluate(
        self,
        timestamp: float,
        ego: VehicleState,
        camera_obs: SensorObservation,
        radar_obs: SensorObservation,
        current_phase: ManeuverPhase,
    ) -> AdvisoryOutput:
        # Simple union without dispute resolution
        all_targets = camera_obs.detected_targets + radar_obs.detected_targets
        oncoming = [
            t for t in all_targets
            if (1.0 <= t.range_y <= 5.5) and (t.rel_vx < -10.0 or t.abs_vx <= 0.0) and t.range_x > 0
        ]

        min_range = 999.0
        crit_target = None
        for t in oncoming:
            if t.range_x < min_range:
                min_range = t.range_x
                crit_target = t

        dummy_abort = AbortabilityState(
            is_feasible=True,
            required_deceleration=2.5,
            time_to_abort=3.0,
            slot_clearance=10.0,
            point_of_no_return=False,
            reason="Naive baseline assumes abort is constantly viable.",
        )

        if min_range < self.critical_distance_m:
            if current_phase in (ManeuverPhase.PASSING, ManeuverPhase.LANE_CHANGE_OUT):
                level = AdvisoryLevel.ABORT_NOW
                reason = f"Naive Fusion: Target detected at {min_range:.1f}m < {self.critical_distance_m:.1f}m. Abort."
            else:
                level = AdvisoryLevel.WAIT
                reason = f"Naive Fusion: Opposing target at {min_range:.1f}m. Wait."
        else:
            level = AdvisoryLevel.SAFE_TO_OVERTAKE
            reason = f"Naive Fusion: Opposing lane appears clear beyond {self.critical_distance_m:.1f}m."

        return AdvisoryOutput(
            timestamp=timestamp,
            phase=current_phase,
            level=level,
            primary_reason=reason,
            confidence=0.8,
            epistemic_uncertainty=0.0,
            aleatoric_uncertainty=0.15,
            time_to_complete=6.0,
            available_time_window=min_range / max(1.0, ego.vx + 25.0),
            safety_margin_time=(min_range / max(1.0, ego.vx + 25.0)) - 6.0,
            safety_margin_distance=min_range - 100.0,
            abort_state=dummy_abort,
            occlusion_zones=[],
            detected_hazards=oncoming,
            sensor_trust_index=0.75,
        )
