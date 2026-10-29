"""
Abortability analysis engine for OverTakeGuard.
Evaluates physical feasibility, deceleration requirements, and the Point of No Return (PoNR)
to ensure safe return to the original travel lane.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
from overtakeguard.models import (
    VehicleState,
    AbortabilityState,
    ManeuverPhase
)


class AbortabilityEngine:
    def __init__(
        self,
        max_comfortable_decel_ms2: float = 3.5,
        max_emergency_decel_ms2: float = 6.0,
        lateral_lane_change_time_s: float = 2.0,
        reaction_time_s: float = 0.8,
        min_rear_safety_margin_m: float = 8.0,
    ):
        self.max_comfortable_decel = max_comfortable_decel_ms2
        self.max_emergency_decel = max_emergency_decel_ms2
        self.lateral_return_time = lateral_lane_change_time_s
        self.reaction_time = reaction_time_s
        self.min_rear_safety_margin = min_rear_safety_margin_m

    def evaluate_abort_feasibility(
        self,
        ego: VehicleState,
        lead: Optional[VehicleState],
        phase: ManeuverPhase,
        time_to_complete_forward: float,
    ) -> AbortabilityState:
        """
        Evaluates whether an abort manoeuvre returning to the original lane is physically feasible.
        """
        # If ego is still following, abort is trivially feasible (ego is already in lane)
        if phase == ManeuverPhase.FOLLOWING:
            return AbortabilityState(
                is_feasible=True,
                required_deceleration=0.0,
                time_to_abort=0.0,
                slot_clearance=50.0,
                point_of_no_return=False,
                reason="Ego vehicle is in original travel lane; abort is trivial (maintain following).",
            )

        if lead is None:
            # No lead vehicle to squeeze behind; return to lane is completely clear
            return AbortabilityState(
                is_feasible=True,
                required_deceleration=1.0,
                time_to_abort=self.reaction_time + self.lateral_return_time,
                slot_clearance=100.0,
                point_of_no_return=False,
                reason="Original lane is unobstructed; return to lane is unconstrained.",
            )

        # Target slot in original lane is behind lead's rear bumper:
        lead_rear_x = lead.x - lead.length / 2.0
        target_slot_front_x = lead_rear_x - self.min_rear_safety_margin
        ego_front_x = ego.x + ego.length / 2.0
        ego_rear_x = ego.x - ego.length / 2.0

        # Distance ego must drop back relative to lead vehicle:
        # If ego is at x=50, lead at x=40, ego is 10m ahead. To get behind lead_rear (32m),
        # ego must fall back by (50 - 32) = 18m.
        longitudinal_drop_needed = ego_rear_x - target_slot_front_x

        # If ego is already longitudinally behind lead's rear bumper
        if longitudinal_drop_needed <= 0:
            time_to_abort = self.reaction_time + self.lateral_return_time
            return AbortabilityState(
                is_feasible=True,
                required_deceleration=0.5,
                time_to_abort=round(time_to_abort, 2),
                slot_clearance=abs(longitudinal_drop_needed) + 15.0,
                point_of_no_return=False,
                reason="Ego is behind lead rear bumper; gentle return to lane is immediately feasible.",
            )

        # Ego is alongside or ahead of lead vehicle.
        # Check relative velocity: if ego is faster than lead, ego must brake below lead's speed
        rel_v = ego.vx - lead.vx  # [m/s] (positive means ego is pulling away)

        # Dynamic calculation: To drop back distance S relative to lead while moving at rel_v:
        # Time required to shed rel_v with deceleration a_rel: t_shed = rel_v / a_rel
        # Distance gained forward during shedding: S_gain = 0.5 * rel_v * t_shed
        # Total relative distance to recover: S_total = longitudinal_drop_needed + S_gain
        # Once at lead speed, ego must brake by delta_v_back (e.g. 3 m/s below lead) to slip back.
        # Approximate required braking deceleration:
        estimated_abort_time = max(
            2.5,
            self.reaction_time + self.lateral_return_time + math.sqrt(max(1.0, 2.0 * longitudinal_drop_needed / 2.5))
        )
        
        # Required deceleration to execute this relative rollback
        # S_rel = 0.5 * a_decel * t^2 => a_decel = 2 * S_rel / t^2
        req_decel = (2.0 * longitudinal_drop_needed) / max(1.0, (estimated_abort_time - self.reaction_time) ** 2)
        req_decel = max(0.8, req_decel)

        # POINT OF NO RETURN (PoNR) Evaluation:
        # 1. Required deceleration exceeds emergency threshold (> 6 m/s^2)
        # 2. Ego is already ahead of lead vehicle front bumper (ego_rear_x > lead.x + lead.length/2)
        # 3. Completing the overtake forward takes significantly less time than aborting
        is_past_front_bumper = ego_rear_x > (lead.x + lead.length / 4.0)
        is_faster_to_complete = time_to_complete_forward < (estimated_abort_time * 0.7)
        exceeds_emergency_braking = req_decel > self.max_emergency_decel

        point_of_no_return = (is_past_front_bumper and is_faster_to_complete) or exceeds_emergency_braking

        if point_of_no_return:
            return AbortabilityState(
                is_feasible=False,
                required_deceleration=round(req_decel, 2),
                time_to_abort=round(estimated_abort_time, 2),
                slot_clearance=0.0,
                point_of_no_return=True,
                reason=(
                    f"POINT OF NO RETURN REACHED: Abort requires extreme braking ({req_decel:.1f} m/s²). "
                    f"Ego is past lead vehicle. Forward completion ({time_to_complete_forward:.1f}s) "
                    f"is physically safer than aborting ({estimated_abort_time:.1f}s)."
                ),
            )

        if req_decel > self.max_comfortable_decel:
            return AbortabilityState(
                is_feasible=True,
                required_deceleration=round(req_decel, 2),
                time_to_abort=round(estimated_abort_time, 2),
                slot_clearance=round(max(2.0, 20.0 - longitudinal_drop_needed), 1),
                point_of_no_return=False,
                reason=(
                    f"Abort feasible via HARD BRAKING ({req_decel:.1f} m/s² > comfort {self.max_comfortable_decel:.1f} m/s²). "
                    f"Requires {estimated_abort_time:.1f}s to fall back into lane."
                ),
            )

        return AbortabilityState(
            is_feasible=True,
            required_deceleration=round(req_decel, 2),
            time_to_abort=round(estimated_abort_time, 2),
            slot_clearance=round(max(5.0, 25.0 - longitudinal_drop_needed), 1),
            point_of_no_return=False,
            reason=f"Abort fully feasible with moderate braking ({req_decel:.1f} m/s²). Time to safe slot: {estimated_abort_time:.1f}s.",
        )
