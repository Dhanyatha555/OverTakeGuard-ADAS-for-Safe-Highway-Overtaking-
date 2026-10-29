from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any


class VehicleType(Enum):
    EGO = "Ego Vehicle"
    LEAD = "Lead Vehicle"
    ONCOMING = "Oncoming Vehicle"
    FOLLOWING = "Following Vehicle"
    OBSTACLE = "Obstacle"


class ManeuverPhase(Enum):
    FOLLOWING = "Following Lead"
    LANE_CHANGE_OUT = "Lane Change Out (Into Opposing)"
    PASSING = "Passing in Opposing Lane"
    LANE_CHANGE_IN = "Lane Change In (Return to Lane)"
    ABORTING = "Aborting Maneuver (Merging Back)"
    COMPLETED = "Overtake Completed"


class AdvisoryLevel(Enum):
    SAFE_TO_OVERTAKE = "SAFE TO OVERTAKE"
    WAIT = "WAIT"
    CAUTION = "CAUTION"
    ABORT_NOW = "ABORT NOW"


class WeatherCondition(Enum):
    CLEAR = "Clear / Sunny"
    LIGHT_RAIN = "Light Rain"
    HEAVY_RAIN = "Heavy Rain"
    DENSE_FOG = "Dense Fog"
    NIGHT_GLARE = "Night / Headlight Glare"


@dataclass
class VehicleState:
    id: str
    vehicle_type: VehicleType
    x: float               # Longitudinal position [m] along the highway
    y: float               # Lateral position [m] (0.0 = ego lane center, 3.75 = opposing lane center)
    vx: float              # Longitudinal velocity [m/s] (positive in ego direction, negative for oncoming)
    vy: float = 0.0        # Lateral velocity [m/s]
    ax: float = 0.0        # Longitudinal acceleration [m/s^2]
    length: float = 4.5    # Length of vehicle [m] (cars ~4.5m, trucks ~14-16m)
    width: float = 1.8     # Width of vehicle [m] (cars ~1.8m, trucks ~2.5m)
    is_occluding: bool = False  # Large vehicle causing line-of-sight blockage
    visible_to_driver: bool = True  # True ground truth visibility flag

    @property
    def speed_kph(self) -> float:
        return abs(self.vx) * 3.6

    @property
    def front_bumper_x(self) -> float:
        return self.x + (self.length / 2.0 if self.vx >= 0 else -self.length / 2.0)

    @property
    def rear_bumper_x(self) -> float:
        return self.x - (self.length / 2.0 if self.vx >= 0 else -self.length / 2.0)

    def copy(self) -> VehicleState:
        return VehicleState(
            id=self.id,
            vehicle_type=self.vehicle_type,
            x=self.x,
            y=self.y,
            vx=self.vx,
            vy=self.vy,
            ax=self.ax,
            length=self.length,
            width=self.width,
            is_occluding=self.is_occluding,
            visible_to_driver=self.visible_to_driver
        )


@dataclass
class DetectedTarget:
    target_id: str
    range_x: float            # Relative longitudinal distance from ego [m]
    range_y: float            # Relative lateral distance from ego [m]
    rel_vx: float             # Relative longitudinal velocity [m/s]
    abs_vx: float             # Estimated absolute longitudinal velocity [m/s]
    confidence: float         # Detection confidence [0.0 - 1.0]
    sensor_source: str        # 'camera', 'radar', or 'fused'
    variance_x: float = 1.0   # Positional variance [m^2]
    variance_vx: float = 0.5  # Velocity variance [(m/s)^2]
    is_conflict: bool = False # Flag indicating cross-sensor contradiction


@dataclass
class SensorObservation:
    sensor_name: str
    timestamp: float
    detected_targets: List[DetectedTarget]
    sensor_health: float      # Reliability factor [0.0 - 1.0]
    field_of_view_deg: float
    max_range_m: float
    weather_impact_factor: float


@dataclass
class OcclusionZone:
    start_x: float            # Start of occluded highway segment [m]
    end_x: float              # End of occluded highway segment [m]
    lane_id: int              # Target lane occluded (1 = opposing lane)
    depth: float              # Length of occluded segment [m]
    occluding_vehicle_id: str
    latent_hazard_risk: float # Estimated probability/severity of a concealed oncoming hazard [0.0 - 1.0]
    polygon_vertices: List[Tuple[float, float]] = field(default_factory=list) # Visual polygon (x, y)


@dataclass
class AbortabilityState:
    is_feasible: bool
    required_deceleration: float  # [m/s^2] (positive magnitude of braking required)
    time_to_abort: float          # [s] time required to drop back and merge
    slot_clearance: float         # [m] longitudinal clearance available in original lane
    point_of_no_return: bool      # True if aborting is no longer feasible or exceeds safety limits
    reason: str


@dataclass
class AdvisoryOutput:
    timestamp: float
    phase: ManeuverPhase
    level: AdvisoryLevel
    primary_reason: str
    confidence: float             # [0.0 - 1.0] composite confidence
    epistemic_uncertainty: float  # [0.0 - 1.0] uncertainty due to occlusion / lack of visibility
    aleatoric_uncertainty: float  # [0.0 - 1.0] uncertainty due to sensor noise / disagreement
    time_to_complete: float       # [s] estimated time needed to finish overtaking
    available_time_window: float  # [s] time before closest hazard or occluded emergence
    safety_margin_time: float     # [s] available_time - time_needed
    safety_margin_distance: float # [m] distance buffer at completion point
    abort_state: AbortabilityState
    occlusion_zones: List[OcclusionZone]
    detected_hazards: List[DetectedTarget]
    sensor_trust_index: float     # [0.0 - 1.0] weighted sensor concordance score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 2),
            "phase": self.phase.value,
            "advisory_level": self.level.value,
            "primary_reason": self.primary_reason,
            "confidence": round(self.confidence, 3),
            "epistemic_uncertainty": round(self.epistemic_uncertainty, 3),
            "aleatoric_uncertainty": round(self.aleatoric_uncertainty, 3),
            "time_to_complete_s": round(self.time_to_complete, 2),
            "available_time_window_s": round(self.available_time_window, 2),
            "safety_margin_time_s": round(self.safety_margin_time, 2),
            "safety_margin_distance_m": round(self.safety_margin_distance, 1),
            "abort_feasible": self.abort_state.is_feasible,
            "abort_req_decel_ms2": round(self.abort_state.required_deceleration, 2),
            "point_of_no_return": self.abort_state.point_of_no_return,
            "sensor_trust_index": round(self.sensor_trust_index, 3),
        }
