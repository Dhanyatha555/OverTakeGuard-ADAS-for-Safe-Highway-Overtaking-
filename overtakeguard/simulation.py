"""
2D Highway Simulation Engine for OverTakeGuard.
Simulates kinematics of ego, lead, oncoming, and occluding vehicles,
executes sensor pipelines, and tracks maneuver progression.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from overtakeguard.models import (
    VehicleState,
    VehicleType,
    ManeuverPhase,
    WeatherCondition,
    AdvisoryOutput,
    AdvisoryLevel
)
from overtakeguard.sensors import CameraSensor, RadarSensor
from overtakeguard.fusion import SensorFusionEngine
from overtakeguard.decision_engine import RollingHorizonDecisionEngine
from overtakeguard.baselines import TTCOnlyBaseline, NaiveFusionBaseline


class HighwaySimulation:
    def __init__(
        self,
        ego: VehicleState,
        lead: Optional[VehicleState] = None,
        oncoming_vehicles: Optional[List[VehicleState]] = None,
        weather: WeatherCondition = WeatherCondition.CLEAR,
        dt: float = 0.1,
        auto_pilot_advisory_compliance: bool = False,
        seed: Optional[int] = 42,
        geometry_sight_limit_m: Optional[float] = None,
    ):
        self.dt = dt
        self.time = 0.0
        self.weather = weather
        self.auto_pilot = auto_pilot_advisory_compliance
        self.rng = np.random.default_rng(seed)
        self.geometry_sight_limit_m = geometry_sight_limit_m

        # Vehicle states
        self.ego = ego.copy()
        self.lead = lead.copy() if lead else None
        self.oncoming_vehicles = [v.copy() for v in (oncoming_vehicles or [])]

        # Maneuver state machine
        self.phase = ManeuverPhase.FOLLOWING
        self.driver_overtake_intent = False
        self.is_aborting = False

        # Subsystems
        self.camera = CameraSensor()
        self.radar = RadarSensor()
        self.fusion_engine = SensorFusionEngine()
        self.decision_engine = RollingHorizonDecisionEngine()

        # Baselines
        self.ttc_baseline = TTCOnlyBaseline()
        self.naive_fusion_baseline = NaiveFusionBaseline()

        # Telemetry History
        self.history: List[Dict[str, Any]] = []
        self.is_collision = False
        self.collision_target_id: Optional[str] = None
        self.min_ground_truth_clearance_m = 999.0

    @property
    def all_vehicles(self) -> List[VehicleState]:
        vehs = [self.ego]
        if self.lead:
            vehs.append(self.lead)
        vehs.extend(self.oncoming_vehicles)
        return vehs

    def trigger_overtake_initiation(self):
        """Driver signals intent to begin overtaking."""
        if self.phase == ManeuverPhase.FOLLOWING and not self.is_aborting:
            self.driver_overtake_intent = True
            self.phase = ManeuverPhase.LANE_CHANGE_OUT

    def trigger_driver_abort(self):
        """Driver triggers manual or advisory-prompted abort."""
        if self.phase in (ManeuverPhase.LANE_CHANGE_OUT, ManeuverPhase.PASSING):
            self.is_aborting = True
            self.phase = ManeuverPhase.ABORTING

    def step(
        self,
        force_camera_miss: bool = False,
        force_radar_miss: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes one simulation step of duration self.dt.
        """
        # 1. Occlusion engine pre-calculation for sensor line-of-sight
        pre_occ_zones, _ = self.decision_engine.occlusion_engine.compute_occlusions(
            self.ego, self.all_vehicles, self.weather, geometry_sight_limit_m=self.geometry_sight_limit_m
        )

        # 2. Sensor sensing
        cam_obs = self.camera.sense(
            self.ego, self.all_vehicles, pre_occ_zones, self.weather, self.rng, force_miss=force_camera_miss
        )
        rad_obs = self.radar.sense(
            self.ego, self.all_vehicles, pre_occ_zones, self.weather, self.rng, force_miss=force_radar_miss
        )

        # 3. Sensor Fusion & Uncertainty Analysis
        fused_targets, sensor_trust, aleatoric_unc, conflict_score = self.fusion_engine.fuse(
            cam_obs, rad_obs, self.weather
        )

        # 4. OverTakeGuard Decision Engine
        advisory = self.decision_engine.evaluate(
            timestamp=self.time,
            ego=self.ego,
            lead=self.lead,
            all_vehicles=self.all_vehicles,
            fused_targets=fused_targets,
            weather=self.weather,
            current_phase=self.phase,
            sensor_trust_index=sensor_trust,
            aleatoric_uncertainty=aleatoric_unc,
            geometry_sight_limit_m=self.geometry_sight_limit_m,
        )

        # 5. Baseline Evaluations
        raw_radar_targets = rad_obs.detected_targets
        ttc_advisory = self.ttc_baseline.evaluate(
            self.time, self.ego, raw_radar_targets, self.phase
        )
        naive_advisory = self.naive_fusion_baseline.evaluate(
            self.time, self.ego, cam_obs, rad_obs, self.phase
        )

        # 6. Auto-Pilot Advisory Compliance (if enabled, driver obeys ABORT NOW immediately)
        if self.auto_pilot:
            if advisory.level == AdvisoryLevel.ABORT_NOW and self.phase in (ManeuverPhase.LANE_CHANGE_OUT, ManeuverPhase.PASSING):
                self.trigger_driver_abort()

        # 7. Physical Kinematic Updates
        self._update_physics()

        # 8. Check Ground Truth Collisions & Margins
        self._check_ground_truth_hazards()

        # 9. Record Step State Snapshot
        snapshot = {
            "time": round(self.time, 2),
            "ego_x": round(self.ego.x, 2),
            "ego_y": round(self.ego.y, 2),
            "ego_vx": round(self.ego.vx, 2),
            "ego_speed_kph": round(self.ego.speed_kph, 1),
            "phase": self.phase.value,
            "lead_x": round(self.lead.x, 2) if self.lead else None,
            "oncoming": [{"id": v.id, "x": round(v.x, 2), "y": round(v.y, 2), "vx": round(v.vx, 2)} for v in self.oncoming_vehicles],
            "advisory": advisory,
            "ttc_baseline_level": ttc_advisory.level.value,
            "naive_baseline_level": naive_advisory.level.value,
            "sensor_trust": sensor_trust,
            "aleatoric_uncertainty": aleatoric_unc,
            "camera_targets_count": len(cam_obs.detected_targets),
            "radar_targets_count": len(rad_obs.detected_targets),
            "fused_targets_count": len(fused_targets),
            "is_collision": self.is_collision,
            "min_clearance_m": round(self.min_ground_truth_clearance_m, 1),
        }
        self.history.append(snapshot)
        self.time += self.dt

        return snapshot

    def _update_physics(self):
        """Simulates vehicle dynamics and lane changing kinematics."""
        # A. Lead Vehicle Dynamics (cruising smoothly)
        if self.lead:
            self.lead.x += self.lead.vx * self.dt

        # B. Oncoming Vehicles Dynamics (moving in opposing direction, vx < 0)
        for v in self.oncoming_vehicles:
            v.x += v.vx * self.dt

        # C. Ego Vehicle Maneuver State Machine Dynamics
        opposing_lane_center_y = 3.75
        ego_lane_center_y = 0.0

        if self.phase == ManeuverPhase.FOLLOWING:
            # Cruising at comfortable following distance behind lead
            if self.lead:
                gap = (self.lead.x - self.lead.length / 2.0) - (self.ego.x + self.ego.length / 2.0)
                desired_gap = 25.0
                if gap < desired_gap:
                    # Decelerate to match lead
                    self.ego.ax = max(-2.0, (gap - desired_gap) * 0.2)
                else:
                    # Accelerate gently to cruise
                    self.ego.ax = min(1.0, (self.lead.vx - self.ego.vx) * 0.5)
            else:
                self.ego.ax = 0.0

            self.ego.vx = max(5.0, min(28.0, self.ego.vx + self.ego.ax * self.dt))
            self.ego.x += self.ego.vx * self.dt
            self.ego.y = ego_lane_center_y

        elif self.phase == ManeuverPhase.LANE_CHANGE_OUT:
            # Steering into oncoming lane while accelerating
            lateral_speed = 3.75 / 2.5  # complete in 2.5 seconds (~1.5 m/s)
            self.ego.y = min(opposing_lane_center_y, self.ego.y + lateral_speed * self.dt)
            self.ego.ax = 1.8
            self.ego.vx = min(33.33, self.ego.vx + self.ego.ax * self.dt)
            self.ego.x += self.ego.vx * self.dt

            if self.ego.y >= opposing_lane_center_y - 0.05:
                self.ego.y = opposing_lane_center_y
                self.phase = ManeuverPhase.PASSING

        elif self.phase == ManeuverPhase.PASSING:
            # Accelerating in oncoming lane alongside lead vehicle
            self.ego.y = opposing_lane_center_y
            self.ego.ax = 1.6
            self.ego.vx = min(33.33, self.ego.vx + self.ego.ax * self.dt)
            self.ego.x += self.ego.vx * self.dt

            # Check if ego has safely cleared the lead vehicle front bumper
            if self.lead:
                lead_front = self.lead.x + self.lead.length / 2.0
                ego_rear = self.ego.x - self.ego.length / 2.0
                if ego_rear >= (lead_front + 15.0):
                    self.phase = ManeuverPhase.LANE_CHANGE_IN

        elif self.phase == ManeuverPhase.LANE_CHANGE_IN:
            # Returning to original lane ahead of lead vehicle
            lateral_speed = 3.75 / 2.0
            self.ego.y = max(ego_lane_center_y, self.ego.y - lateral_speed * self.dt)
            self.ego.ax = -0.5  # ease off accelerator
            self.ego.vx = max(22.0, self.ego.vx + self.ego.ax * self.dt)
            self.ego.x += self.ego.vx * self.dt

            if self.ego.y <= ego_lane_center_y + 0.05:
                self.ego.y = ego_lane_center_y
                self.phase = ManeuverPhase.COMPLETED

        elif self.phase == ManeuverPhase.ABORTING:
            # Decelerating and dropping back into original lane behind lead
            # 1. Hard braking
            self.ego.ax = -3.8
            self.ego.vx = max(12.0, self.ego.vx + self.ego.ax * self.dt)
            self.ego.x += self.ego.vx * self.dt

            # 2. Lateral merge back into ego lane once longitudinal slot is open
            lead_rear = (self.lead.x - self.lead.length / 2.0) if self.lead else 0.0
            ego_front = self.ego.x + self.ego.length / 2.0

            # If ego is behind or pulling behind lead rear bumper, steer back
            if not self.lead or ego_front <= (lead_rear + 2.0):
                lateral_speed = 3.75 / 1.8
                self.ego.y = max(ego_lane_center_y, self.ego.y - lateral_speed * self.dt)
                if self.ego.y <= ego_lane_center_y + 0.05:
                    self.ego.y = ego_lane_center_y
                    self.phase = ManeuverPhase.FOLLOWING
                    self.is_aborting = False

        elif self.phase == ManeuverPhase.COMPLETED:
            # Cruising safely ahead of lead
            self.ego.ax = 0.0
            self.ego.vx = 27.78
            self.ego.x += self.ego.vx * self.dt
            self.ego.y = ego_lane_center_y

    def _check_ground_truth_hazards(self):
        """Evaluates ground truth proximity and detects collisions."""
        ego_front = self.ego.x + self.ego.length / 2.0
        ego_rear = self.ego.x - self.ego.length / 2.0
        ego_left = self.ego.y + self.ego.width / 2.0
        ego_right = self.ego.y - self.ego.width / 2.0

        for v in self.all_vehicles:
            if v.id == self.ego.id:
                continue

            v_front = v.x + (v.length / 2.0 if v.vx >= 0 else -v.length / 2.0)
            v_rear = v.x - (v.length / 2.0 if v.vx >= 0 else -v.length / 2.0)
            v_min_x = min(v_front, v_rear)
            v_max_x = max(v_front, v_rear)

            v_left = v.y + v.width / 2.0
            v_right = v.y - v.width / 2.0

            # Longitudinal & lateral clearance
            dx = max(0.0, max(ego_rear - v_max_x, v_min_x - ego_front))
            dy = max(0.0, max(ego_right - v_left, v_right - ego_left))
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < self.min_ground_truth_clearance_m:
                self.min_ground_truth_clearance_m = dist

            # Collision check: bounding box intersection
            overlap_x = (ego_front >= v_min_x) and (ego_rear <= v_max_x)
            overlap_y = (ego_left >= v_right) and (ego_right <= v_left)

            if overlap_x and overlap_y:
                self.is_collision = True
                self.collision_target_id = v.id
