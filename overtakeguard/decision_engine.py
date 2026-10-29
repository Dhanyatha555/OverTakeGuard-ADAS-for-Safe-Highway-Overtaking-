"""
Rolling-Horizon Decision Engine for OverTakeGuard.
Continuously re-evaluates overtaking feasibility, incorporates stochastic uncertainty bounds,
checks abortability constraints, and outputs explainable multi-level advisories.
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple
from overtakeguard.models import (
    VehicleState,
    ManeuverPhase,
    AdvisoryLevel,
    AdvisoryOutput,
    DetectedTarget,
    OcclusionZone,
    WeatherCondition
)
from overtakeguard.occlusion import OcclusionEngine
from overtakeguard.abortability import AbortabilityEngine


class RollingHorizonDecisionEngine:
    def __init__(
        self,
        driver_reaction_time_s: float = 1.2,
        minimum_safety_buffer_s: float = 2.5,
        lane_change_duration_s: float = 2.5,
        ego_overtake_accel_ms2: float = 1.8,
        ego_max_overtake_speed_ms: float = 33.33,  # 120 km/h
        opposing_lane_y: float = 3.75,
    ):
        self.reaction_time = driver_reaction_time_s
        self.safety_buffer = minimum_safety_buffer_s
        self.lane_change_time = lane_change_duration_s
        self.overtake_accel = ego_overtake_accel_ms2
        self.max_overtake_speed = ego_max_overtake_speed_ms
        self.opposing_lane_y = opposing_lane_y

        self.occlusion_engine = OcclusionEngine(opposing_lane_y=opposing_lane_y)
        self.abort_engine = AbortabilityEngine()

    def estimate_completion_time(
        self,
        ego: VehicleState,
        lead: Optional[VehicleState],
        phase: ManeuverPhase,
    ) -> float:
        """
        Estimates the time needed [s] from the current state to safely complete
        the overtaking manoeuvre and merge back into the ego lane.
        """
        if lead is None:
            return self.lane_change_time * 2.0 + 3.0

        # Safety re-entry gap ahead of lead vehicle
        reentry_buffer_m = 15.0

        lead_front_x = lead.x + lead.length / 2.0
        ego_rear_x = ego.x - ego.length / 2.0

        # Target point: ego's rear bumper is reentry_buffer_m ahead of lead's front bumper
        target_merge_x = lead_front_x + reentry_buffer_m
        remaining_forward_dist = target_merge_x - ego_rear_x

        if remaining_forward_dist <= 2.0:
            # Ego is already well ahead of lead, only needs lane change back
            return self.lane_change_time

        # Kinematics: ego relative to lead vehicle
        # Ego accelerates up to max_overtake_speed
        current_v = ego.vx
        lead_v = lead.vx

        # Numerical integration / kinematics of passing phase
        dt = 0.1
        sim_t = 0.0
        cur_ego_x = ego.x
        cur_lead_x = lead.x
        sim_ego_v = current_v

        while sim_t < 25.0:
            if sim_ego_v < self.max_overtake_speed:
                sim_ego_v = min(self.max_overtake_speed, sim_ego_v + self.overtake_accel * dt)

            cur_ego_x += sim_ego_v * dt
            cur_lead_x += lead_v * dt
            sim_t += dt

            cur_ego_rear = cur_ego_x - ego.length / 2.0
            cur_lead_front = cur_lead_x + lead.length / 2.0

            if cur_ego_rear >= (cur_lead_front + reentry_buffer_m):
                break

        # Add remaining lane changes based on current phase
        if phase == ManeuverPhase.FOLLOWING:
            return sim_t + self.lane_change_time * 2.0
        elif phase == ManeuverPhase.LANE_CHANGE_OUT:
            return sim_t + self.lane_change_time * 1.5
        elif phase == ManeuverPhase.PASSING:
            return sim_t + self.lane_change_time
        elif phase == ManeuverPhase.LANE_CHANGE_IN:
            return min(self.lane_change_time, sim_t)
        return sim_t

    def evaluate(
        self,
        timestamp: float,
        ego: VehicleState,
        lead: Optional[VehicleState],
        all_vehicles: List[VehicleState],
        fused_targets: List[DetectedTarget],
        weather: WeatherCondition = WeatherCondition.CLEAR,
        current_phase: ManeuverPhase = ManeuverPhase.FOLLOWING,
        sensor_trust_index: float = 0.95,
        aleatoric_uncertainty: float = 0.1,
        geometry_sight_limit_m: Optional[float] = None,
    ) -> AdvisoryOutput:
        """
        Main decision-making cycle. Re-evaluates feasibility on a rolling horizon.
        """
        # 1. Compute line-of-sight occlusions
        occlusion_zones, clear_sight_dist = self.occlusion_engine.compute_occlusions(
            ego=ego,
            vehicles=all_vehicles,
            weather=weather,
            geometry_sight_limit_m=geometry_sight_limit_m,
        )

        # 2. Compute time required to complete overtaking forward
        time_to_complete = self.estimate_completion_time(ego, lead, current_phase)
        total_required_time = time_to_complete + self.reaction_time + self.safety_buffer

        # 3. Assess detected hazards in opposing lane
        # Filter genuine oncoming hazards from benign objects (lead vehicle, roadside signs/poles)
        oncoming_hazards: List[DetectedTarget] = []
        benign_objects: List[DetectedTarget] = []
        lead_id = lead.id if lead else None

        for t in fused_targets:
            if t.range_x <= 0:
                continue

            # Benign filter: Lead vehicle ahead in ego lane
            is_lead = (lead_id is not None and t.target_id == lead_id)
            # Benign filter: Roadside shoulder objects outside travel lanes
            is_roadside = (t.range_y < 0.8) or (t.range_y > 5.8)
            # Benign filter: Vehicles traveling in same direction as ego in ego lane
            is_same_direction = (t.abs_vx > 5.0) and (t.range_y < 2.0)

            if is_lead or is_roadside or is_same_direction:
                benign_objects.append(t)
                continue

            # Genuine opposing hazard: In opposing lane corridor and approaching in opposing direction
            in_opposing_corridor = (1.0 <= t.range_y <= 5.5)
            is_oncoming_closing = (t.abs_vx <= 0.0) or (t.rel_vx < -10.0)

            if in_opposing_corridor and is_oncoming_closing:
                oncoming_hazards.append(t)
            else:
                benign_objects.append(t)

        min_detected_ttc = 999.0
        most_critical_target: Optional[DetectedTarget] = None

        for t in oncoming_hazards:
            # Conservative closing velocity using 2-sigma variance
            closing_v = (ego.vx - t.rel_vx) + 2.0 * math.sqrt(max(0.01, t.variance_vx))
            closing_dist = max(0.5, t.range_x - 2.0 * math.sqrt(max(0.01, t.variance_x)))

            if closing_v > 0.5:
                ttc = closing_dist / closing_v
                if ttc < min_detected_ttc:
                    min_detected_ttc = ttc
                    most_critical_target = t

        # 4. Assess occluded latent hazard emergence
        is_latent_critical, latent_emergence_t, latent_reason = self.occlusion_engine.evaluate_latent_hazard_emergence(
            ego=ego,
            occlusion_zones=occlusion_zones,
            required_maneuver_time=time_to_complete,
        )

        # 4b. Check if opposing lane is currently occupied alongside or passing ego
        lane_occupied_alongside = False
        alongside_vehicle_desc = ""
        ego_front = ego.x + ego.length / 2.0
        ego_rear = ego.x - ego.length / 2.0

        for v in all_vehicles:
            if v.id == ego.id:
                continue
            if abs(v.y - self.opposing_lane_y) < 1.5:  # In opposing lane
                v_front = v.x + (v.length / 2.0 if v.vx >= 0 else -v.length / 2.0)
                v_rear = v.x - (v.length / 2.0 if v.vx >= 0 else -v.length / 2.0)
                v_min = min(v_front, v_rear)
                v_max = max(v_front, v_rear)

                # Buffer of 10m around ego body
                if not (v_max < ego_rear - 10.0 or v_min > ego_front + 15.0):
                    lane_occupied_alongside = True
                    alongside_vehicle_desc = f"{v.id} passing alongside at {v.x:.1f}m"
                    break

        # 5. Compute available time window
        available_window = min(min_detected_ttc, latent_emergence_t)
        safety_margin_time = available_window - total_required_time
        safety_margin_dist = safety_margin_time * ego.vx

        # 6. Epistemic uncertainty calculation
        # Fraction of oncoming road occluded within critical horizon (150m)
        total_occ_depth = sum(z.depth for z in occlusion_zones if z.start_x < ego.x + 150.0)
        epistemic_unc = min(0.95, (total_occ_depth / 100.0) * 0.7 + (1.0 - min(1.0, clear_sight_dist / 200.0)) * 0.3)
        epistemic_unc = round(epistemic_unc, 3)

        # Overall composite confidence
        composite_confidence = max(0.05, min(0.99, (1.0 - epistemic_unc * 0.5) * (1.0 - aleatoric_uncertainty * 0.5) * sensor_trust_index))
        composite_confidence = round(composite_confidence, 3)

        # 7. Abortability evaluation
        abort_state = self.abort_engine.evaluate_abort_feasibility(
            ego=ego,
            lead=lead,
            phase=current_phase,
            time_to_complete_forward=time_to_complete,
        )

        # 8. Multi-Level Explainable Advisory Selection
        level = AdvisoryLevel.SAFE_TO_OVERTAKE
        primary_reason = ""

        # --- A. IN-PROGRESS MANEUVER (LANE_CHANGE_OUT, PASSING) ---
        if current_phase in (ManeuverPhase.LANE_CHANGE_OUT, ManeuverPhase.PASSING):
            # Imminent collision or severely depleted margin
            is_imminent_threat = (min_detected_ttc < 5.0) or (safety_margin_time < -1.0) or (is_latent_critical and latent_emergence_t < 3.5)

            if is_imminent_threat:
                if abort_state.is_feasible:
                    level = AdvisoryLevel.ABORT_NOW
                    if most_critical_target is not None:
                        primary_reason = (
                            f"ABORT NOW: Oncoming vehicle {most_critical_target.target_id} at {most_critical_target.range_x:.1f}m "
                            f"(TTC: {min_detected_ttc:.1f}s < required {total_required_time:.1f}s). "
                            f"Safe slot open in ego lane, deceleration {abort_state.required_deceleration:.1f} m/s²."
                        )
                    else:
                        primary_reason = (
                            f"ABORT NOW: Latent hazard emergence risk ({latent_emergence_t:.1f}s). "
                            f"Abort is feasible with {abort_state.required_deceleration:.1f} m/s² braking."
                        )
                else:
                    # CRITICAL RULE: "Never recommend an abort manoeuvre without checking whether it is still feasible."
                    level = AdvisoryLevel.CAUTION
                    primary_reason = (
                        f"CRITICAL CAUTION (POINT OF NO RETURN): Approaching hazard (TTC: {min_detected_ttc:.1f}s). "
                        f"DO NOT ABORT: Returning to lane requires unsafe {abort_state.required_deceleration:.1f} m/s². "
                        f"Rapid forward completion is physically safer ({time_to_complete:.1f}s needed)."
                    )
            elif safety_margin_time < 1.5 or aleatoric_uncertainty > 0.45 or epistemic_unc > 0.5:
                level = AdvisoryLevel.CAUTION
                primary_reason = (
                    f"CAUTION: Tight safety margin ({safety_margin_time:.1f}s) or elevated uncertainty "
                    f"(Aleatoric: {aleatoric_uncertainty:.2f}, Epistemic: {epistemic_unc:.2f}). Monitor closing gaps closely."
                )
            else:
                level = AdvisoryLevel.SAFE_TO_OVERTAKE
                primary_reason = (
                    f"MANEUVER PROCEEDING SAFELY: Clear buffer ahead (+{safety_margin_time:.1f}s margin). "
                    f"Completion in {time_to_complete:.1f}s."
                )

        # --- B. PRE-MANEUVER EVALUATION (FOLLOWING) ---
        else:
            # 0. Opposing lane currently occupied alongside ego
            if lane_occupied_alongside:
                level = AdvisoryLevel.WAIT
                primary_reason = f"WAIT: Opposing lane currently occupied ({alongside_vehicle_desc}). Do not steer into oncoming traffic."

            # 1. Direct detected oncoming hazard
            elif most_critical_target is not None and (min_detected_ttc < total_required_time or most_critical_target.range_x < 220.0):
                level = AdvisoryLevel.WAIT
                primary_reason = (
                    f"WAIT: Approaching oncoming hazard ({most_critical_target.target_id}) detected at {most_critical_target.range_x:.1f}m. "
                    f"TTC: {min_detected_ttc:.1f}s < required {total_required_time:.1f}s (Deficit: {total_required_time - min_detected_ttc:.1f}s)."
                )

            # 2. Hidden latent hazard due to occlusion
            elif is_latent_critical:
                level = AdvisoryLevel.WAIT
                primary_reason = latent_reason

            # 3. Weather / Visibility / Sensor Conflict degradation
            elif sensor_trust_index < 0.65 or aleatoric_uncertainty > 0.4:
                level = AdvisoryLevel.CAUTION
                primary_reason = (
                    f"CAUTION (UNRELIABLE SENSORS): Degraded sensor trust ({sensor_trust_index:.2f}) or sensor contradiction. "
                    f"Weather ({weather.value}) limits radar/camera concordance. Do not initiate overtake until verified."
                )

            # 4. Insufficient clear sight distance
            elif clear_sight_dist < (ego.vx * total_required_time * 0.7):
                level = AdvisoryLevel.WAIT
                primary_reason = (
                    f"WAIT: Sight distance limited to {clear_sight_dist:.1f}m. "
                    f"Safe overtake requires at least {(ego.vx * total_required_time * 0.7):.1f}m of unobstructed visibility."
                )

            # 5. All safety criteria satisfied
            else:
                level = AdvisoryLevel.SAFE_TO_OVERTAKE
                primary_reason = (
                    f"SAFE TO OVERTAKE: Clear road confirmed for {clear_sight_dist:.1f}m. "
                    f"Available time window {available_window:.1f}s exceeds required {total_required_time:.1f}s "
                    f"(Buffer: +{safety_margin_time:.1f}s). Abort lane slot open."
                )

        return AdvisoryOutput(
            timestamp=timestamp,
            phase=current_phase,
            level=level,
            primary_reason=primary_reason,
            confidence=composite_confidence,
            epistemic_uncertainty=epistemic_unc,
            aleatoric_uncertainty=aleatoric_uncertainty,
            time_to_complete=round(time_to_complete, 2),
            available_time_window=round(available_window, 2),
            safety_margin_time=round(safety_margin_time, 2),
            safety_margin_distance=round(safety_margin_dist, 1),
            abort_state=abort_state,
            occlusion_zones=occlusion_zones,
            detected_hazards=oncoming_hazards,
            sensor_trust_index=sensor_trust_index,
        )
