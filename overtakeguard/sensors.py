"""
Simulated sensor models (Camera & Radar) for OverTakeGuard.
Generates realistic noisy measurements, handles environmental degradation,
field of view limits, and occlusion line-of-sight clipping.
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple, Optional
from overtakeguard.models import (
    VehicleState,
    DetectedTarget,
    SensorObservation,
    WeatherCondition,
    OcclusionZone
)


class BaseSensor:
    def __init__(self, name: str, max_range_m: float, fov_deg: float):
        self.name = name
        self.max_range_m = max_range_m
        self.fov_deg = fov_deg
        self.fov_rad = math.radians(fov_deg)

    def is_in_fov(self, ego: VehicleState, target: VehicleState) -> Tuple[bool, float, float]:
        """Calculates relative distance and angle; checks field-of-view cone."""
        dx = target.x - ego.x
        dy = target.y - ego.y
        dist = math.sqrt(dx * dx + dy * dy)
        angle = math.atan2(dy, dx)

        if dist > self.max_range_m:
            return False, dist, angle
        if dx < 0:  # Sensor faces forward
            return False, dist, angle
        if abs(angle) > (self.fov_rad / 2.0):
            return False, dist, angle

        return True, dist, angle


class CameraSensor(BaseSensor):
    """
    Monocular/Stereo Vision Camera:
    - High classification confidence and lateral resolution.
    - Moderate longitudinal range and velocity precision.
    - Highly vulnerable to heavy rain, dense fog, and glare/lighting changes.
    """
    def __init__(
        self,
        max_range_m: float = 180.0,
        fov_deg: float = 65.0,
        range_noise_std: float = 2.0,
        vel_noise_std: float = 1.2,
    ):
        super().__init__("Camera", max_range_m, fov_deg)
        self.range_noise_std = range_noise_std
        self.vel_noise_std = vel_noise_std

    def compute_weather_impact(self, weather: WeatherCondition) -> Tuple[float, float, float]:
        """
        Returns:
            (range_multiplier, detection_prob, noise_multiplier)
        """
        if weather == WeatherCondition.CLEAR:
            return 1.0, 0.99, 1.0
        elif weather == WeatherCondition.LIGHT_RAIN:
            return 0.85, 0.92, 1.3
        elif weather == WeatherCondition.HEAVY_RAIN:
            return 0.50, 0.65, 2.5
        elif weather == WeatherCondition.DENSE_FOG:
            return 0.30, 0.40, 3.5
        elif weather == WeatherCondition.NIGHT_GLARE:
            return 0.65, 0.70, 2.0
        return 1.0, 0.95, 1.0

    def sense(
        self,
        ego: VehicleState,
        all_vehicles: List[VehicleState],
        occlusion_zones: List[OcclusionZone],
        weather: WeatherCondition = WeatherCondition.CLEAR,
        rng: Optional[np.random.Generator] = None,
        force_miss: bool = False,
    ) -> SensorObservation:
        if rng is None:
            rng = np.random.default_rng()

        range_mult, det_prob, noise_mult = self.compute_weather_impact(weather)
        effective_range = self.max_range_m * range_mult
        health = max(0.1, min(1.0, det_prob / noise_mult))

        detected: List[DetectedTarget] = []

        if not force_miss:
            for v in all_vehicles:
                if v.id == ego.id:
                    continue

                in_fov, dist, _ = self.is_in_fov(ego, v)
                if not in_fov or dist > effective_range:
                    continue

                # Check if target is inside an occlusion zone
                is_occluded = any(
                    z.start_x <= v.x <= z.end_x for z in occlusion_zones
                )
                if is_occluded:
                    continue  # Camera cannot see through lead truck!

                # Probabilistic detection drop
                if rng.random() > det_prob:
                    continue

                # Add noise
                sigma_x = self.range_noise_std * noise_mult * (1.0 + (dist / 100.0) ** 1.5)
                sigma_vx = self.vel_noise_std * noise_mult

                measured_dx = (v.x - ego.x) + rng.normal(0, sigma_x)
                measured_dy = (v.y - ego.y) + rng.normal(0, 0.15)
                measured_rel_vx = (v.vx - ego.vx) + rng.normal(0, sigma_vx)
                estimated_abs_vx = ego.vx + measured_rel_vx

                conf = max(0.2, min(0.99, (det_prob - 0.2 * (dist / effective_range)) / noise_mult))

                detected.append(
                    DetectedTarget(
                        target_id=v.id,
                        range_x=round(float(measured_dx), 2),
                        range_y=round(float(measured_dy), 2),
                        rel_vx=round(float(measured_rel_vx), 2),
                        abs_vx=round(float(estimated_abs_vx), 2),
                        confidence=round(float(conf), 3),
                        sensor_source="camera",
                        variance_x=round(float(sigma_x ** 2), 3),
                        variance_vx=round(float(sigma_vx ** 2), 3),
                    )
                )

        return SensorObservation(
            sensor_name="Camera",
            timestamp=0.0,
            detected_targets=detected,
            sensor_health=round(health, 3),
            field_of_view_deg=self.fov_deg,
            max_range_m=round(effective_range, 1),
            weather_impact_factor=round(noise_mult, 2),
        )


class RadarSensor(BaseSensor):
    """
    Long-Range Millimeter-Wave Radar:
    - Excellent range accuracy and direct Doppler velocity measurement.
    - Operates well through fog and night.
    - Attenuated by heavy rain (backscatter attenuation).
    - Blocked by metal vehicle hulls (lead truck).
    """
    def __init__(
        self,
        max_range_m: float = 250.0,
        fov_deg: float = 35.0,
        range_noise_std: float = 0.6,
        vel_noise_std: float = 0.35,
    ):
        super().__init__("Radar", max_range_m, fov_deg)
        self.range_noise_std = range_noise_std
        self.vel_noise_std = vel_noise_std

    def compute_weather_impact(self, weather: WeatherCondition) -> Tuple[float, float, float]:
        """
        Radar is much more robust in fog and night, but heavy rain causes microwave attenuation.
        """
        if weather == WeatherCondition.CLEAR:
            return 1.0, 0.99, 1.0
        elif weather == WeatherCondition.LIGHT_RAIN:
            return 0.92, 0.95, 1.15
        elif weather == WeatherCondition.HEAVY_RAIN:
            # Significant rain attenuation and clutter
            return 0.65, 0.72, 2.0
        elif weather == WeatherCondition.DENSE_FOG:
            return 0.90, 0.95, 1.1
        elif weather == WeatherCondition.NIGHT_GLARE:
            return 1.0, 0.99, 1.0  # Radar is unaffected by optical glare
        return 1.0, 0.98, 1.0

    def sense(
        self,
        ego: VehicleState,
        all_vehicles: List[VehicleState],
        occlusion_zones: List[OcclusionZone],
        weather: WeatherCondition = WeatherCondition.CLEAR,
        rng: Optional[np.random.Generator] = None,
        force_miss: bool = False,
    ) -> SensorObservation:
        if rng is None:
            rng = np.random.default_rng()

        range_mult, det_prob, noise_mult = self.compute_weather_impact(weather)
        effective_range = self.max_range_m * range_mult
        health = max(0.1, min(1.0, det_prob / noise_mult))

        detected: List[DetectedTarget] = []

        if not force_miss:
            for v in all_vehicles:
                if v.id == ego.id:
                    continue

                in_fov, dist, _ = self.is_in_fov(ego, v)
                if not in_fov or dist > effective_range:
                    continue

                # Check if target is inside an occlusion zone
                is_occluded = any(
                    z.start_x <= v.x <= z.end_x for z in occlusion_zones
                )
                if is_occluded:
                    continue

                if rng.random() > det_prob:
                    continue

                # Radar has very accurate range and doppler velocity
                sigma_x = self.range_noise_std * noise_mult
                sigma_vx = self.vel_noise_std * noise_mult

                measured_dx = (v.x - ego.x) + rng.normal(0, sigma_x)
                measured_dy = (v.y - ego.y) + rng.normal(0, 0.35)
                measured_rel_vx = (v.vx - ego.vx) + rng.normal(0, sigma_vx)
                estimated_abs_vx = ego.vx + measured_rel_vx

                conf = max(0.3, min(0.99, (det_prob - 0.1 * (dist / effective_range)) / (noise_mult ** 0.5)))

                detected.append(
                    DetectedTarget(
                        target_id=v.id,
                        range_x=round(float(measured_dx), 2),
                        range_y=round(float(measured_dy), 2),
                        rel_vx=round(float(measured_rel_vx), 2),
                        abs_vx=round(float(estimated_abs_vx), 2),
                        confidence=round(float(conf), 3),
                        sensor_source="radar",
                        variance_x=round(float(sigma_x ** 2), 3),
                        variance_vx=round(float(sigma_vx ** 2), 3),
                    )
                )

        return SensorObservation(
            sensor_name="Radar",
            timestamp=0.0,
            detected_targets=detected,
            sensor_health=round(health, 3),
            field_of_view_deg=self.fov_deg,
            max_range_m=round(effective_range, 1),
            weather_impact_factor=round(noise_mult, 2),
        )
