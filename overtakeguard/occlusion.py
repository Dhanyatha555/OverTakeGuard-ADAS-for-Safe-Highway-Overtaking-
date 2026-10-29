"""
Occlusion awareness and line-of-sight modeling for OverTakeGuard.
Calculates visual shadows cast by lead/occluding vehicles and computes
worst-case latent hazard emergence times.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional
from overtakeguard.models import VehicleState, OcclusionZone, WeatherCondition


class OcclusionEngine:
    def __init__(
        self,
        opposing_lane_y: float = 3.75,
        max_horizon_m: float = 350.0,
        default_hypothetical_oncoming_speed: float = 27.78,  # 100 km/h in m/s
    ):
        """
        :param opposing_lane_y: Lateral centerline coordinate of oncoming lane [m].
        :param max_horizon_m: Maximum longitudinal visibility limit under ideal conditions [m].
        :param default_hypothetical_oncoming_speed: Assumed speed of unseen hazard [m/s].
        """
        self.opposing_lane_y = opposing_lane_y
        self.max_horizon_m = max_horizon_m
        self.default_hypothetical_oncoming_speed = default_hypothetical_oncoming_speed

    def compute_weather_sight_limit(self, weather: WeatherCondition) -> float:
        """Computes environmental sight range horizon based on weather."""
        if weather == WeatherCondition.CLEAR:
            return self.max_horizon_m
        elif weather == WeatherCondition.LIGHT_RAIN:
            return self.max_horizon_m * 0.85
        elif weather == WeatherCondition.HEAVY_RAIN:
            return self.max_horizon_m * 0.50
        elif weather == WeatherCondition.DENSE_FOG:
            return self.max_horizon_m * 0.30
        elif weather == WeatherCondition.NIGHT_GLARE:
            return self.max_horizon_m * 0.65
        return self.max_horizon_m

    def compute_occlusions(
        self,
        ego: VehicleState,
        vehicles: List[VehicleState],
        weather: WeatherCondition = WeatherCondition.CLEAR,
        geometry_sight_limit_m: Optional[float] = None,
    ) -> Tuple[List[OcclusionZone], float]:
        """
        Computes all occluded road zones in the opposing lane cast by vehicles ahead of ego
        and road geometry (vertical crests / horizontal curves).
        Returns:
            (List of OcclusionZone, effective_clear_sight_distance)
        """
        weather_horizon = self.compute_weather_sight_limit(weather)
        effective_horizon = weather_horizon
        if geometry_sight_limit_m is not None:
            effective_horizon = min(weather_horizon, geometry_sight_limit_m)

        max_sight_x = ego.x + effective_horizon

        # Ego sensor origin (front center or driver position)
        sensor_x = ego.x + ego.length / 2.0
        sensor_y = ego.y

        occlusion_zones: List[OcclusionZone] = []
        closest_obstruction_sight_x = max_sight_x

        # If road geometry (e.g. crest curve) restricts sightline ahead of ego
        if geometry_sight_limit_m is not None and geometry_sight_limit_m < weather_horizon:
            geom_occ_start = ego.x + geometry_sight_limit_m
            geom_occ_end = geom_occ_start + 120.0
            occlusion_zones.append(
                OcclusionZone(
                    start_x=round(geom_occ_start, 1),
                    end_x=round(geom_occ_end, 1),
                    lane_id=1,
                    depth=round(geom_occ_end - geom_occ_start, 1),
                    occluding_vehicle_id="ROAD_GEOMETRY_CURVE_CREST",
                    latent_hazard_risk=0.85,
                    polygon_vertices=[
                        (geom_occ_start, self.opposing_lane_y + 1.2),
                        (geom_occ_end, self.opposing_lane_y + 1.2),
                        (geom_occ_end, self.opposing_lane_y - 1.2),
                        (geom_occ_start, self.opposing_lane_y - 1.2),
                    ],
                )
            )
            closest_obstruction_sight_x = min(closest_obstruction_sight_x, geom_occ_start)

        # Find all potential occluding vehicles in front of ego
        for v in vehicles:
            if v.id == ego.id:
                continue

            # We are primarily concerned with large vehicles ahead of ego that block line-of-sight
            if v.x <= sensor_x:
                continue

            # Standard passenger cars do not block highway sightlines; large trucks/buses do
            if not v.is_occluding and (v.length < 8.0 and v.width < 2.3):
                continue

            # Check if this vehicle blocks line-of-sight to the opposing lane
            # Vehicle bounding box corners
            rear_x = v.x - v.length / 2.0
            front_x = v.x + v.length / 2.0
            left_y = v.y + v.width / 2.0
            right_y = v.y - v.width / 2.0

            # Ray 1: From ego sensor through vehicle's rear-left corner
            dx_rl = rear_x - sensor_x
            dy_rl = left_y - sensor_y

            # Ray 2: From ego sensor through vehicle's front-left corner
            dx_fl = front_x - sensor_x
            dy_fl = left_y - sensor_y

            # Determine shadow cast on the centerline of the opposing lane (y = opposing_lane_y)
            target_dy = self.opposing_lane_y - sensor_y

            if dy_rl > 0.05 and dx_rl > 0:
                # Intersect ray with y = opposing_lane_y
                shadow_start_x = sensor_x + dx_rl * (target_dy / dy_rl)
            else:
                shadow_start_x = rear_x

            if dy_fl > 0.05 and dx_fl > 0:
                shadow_end_ray_x = sensor_x + dx_fl * (target_dy / dy_fl)
            else:
                shadow_end_ray_x = front_x

            # In practice, the occlusion begins just ahead of where the truck obstructs
            # the line of sight and extends up to the shadow cast by the front-left corner
            # or up to where sightline is restored.
            occ_start = max(rear_x, shadow_start_x)
            occ_end = max(occ_start + 10.0, shadow_end_ray_x + (front_x - rear_x) * 1.5)
            occ_end = min(occ_end, max_sight_x)

            if occ_end > occ_start and occ_start < max_sight_x:
                # Calculate risk: larger trucks closer to ego create larger blind spots
                gap = rear_x - sensor_x
                size_factor = min(1.0, (v.length * v.width) / 30.0)
                proximity_factor = max(0.2, min(1.0, 30.0 / max(5.0, gap)))
                risk = round(size_factor * proximity_factor * (1.2 if v.is_occluding else 0.7), 2)
                risk = min(1.0, max(0.1, risk))

                # Build polygon for visual rendering in dashboard
                poly = [
                    (rear_x, left_y),
                    (occ_start, self.opposing_lane_y + 1.2),
                    (occ_end, self.opposing_lane_y + 1.2),
                    (occ_end, self.opposing_lane_y - 1.2),
                    (occ_start, self.opposing_lane_y - 1.2),
                    (front_x, left_y)
                ]

                zone = OcclusionZone(
                    start_x=round(occ_start, 1),
                    end_x=round(occ_end, 1),
                    lane_id=1,
                    depth=round(occ_end - occ_start, 1),
                    occluding_vehicle_id=v.id,
                    latent_hazard_risk=risk,
                    polygon_vertices=poly,
                )
                occlusion_zones.append(zone)

                if occ_start < closest_obstruction_sight_x:
                    closest_obstruction_sight_x = occ_start

        effective_sight_distance = max(0.0, closest_obstruction_sight_x - sensor_x)
        return occlusion_zones, effective_sight_distance

    def evaluate_latent_hazard_emergence(
        self,
        ego: VehicleState,
        occlusion_zones: List[OcclusionZone],
        required_maneuver_time: float,
    ) -> Tuple[bool, float, str]:
        """
        Evaluates whether a hypothetical oncoming vehicle emerging from the nearest occluded zone
        could collide with ego before the maneuver completes.

        Returns:
            (is_latent_hazard_critical, emergence_time_s, explanation_string)
        """
        if not occlusion_zones:
            return False, 999.0, "No occluded zones detected in opposing lane."

        # Find the most dangerous (closest) occlusion zone
        closest_zone = min(occlusion_zones, key=lambda z: z.start_x)
        relative_distance = closest_zone.start_x - (ego.x + ego.length / 2.0)

        if relative_distance <= 0:
            return True, 0.0, "Ego is already within occluded blind corridor!"

        closing_speed = ego.vx + self.default_hypothetical_oncoming_speed
        if closing_speed <= 1.0:
            closing_speed = 20.0

        emergence_time = relative_distance / closing_speed

        # Safety buffer required: reaction time (1.5s) + safety margin (2.0s)
        safety_buffer = 3.5
        required_clear_time = required_maneuver_time + safety_buffer

        if emergence_time < required_clear_time:
            time_deficit = required_clear_time - emergence_time
            explanation = (
                f"LATENT HAZARD RISK: Blind spot starts {relative_distance:.1f}m ahead behind "
                f"{closest_zone.occluding_vehicle_id}. Unseen vehicle could emerge in "
                f"{emergence_time:.1f}s (Needed: {required_clear_time:.1f}s, Deficit: {time_deficit:.1f}s)."
            )
            return True, emergence_time, explanation

        return False, emergence_time, f"Occluded zone at {relative_distance:.1f}m allows {emergence_time:.1f}s buffer (Safe)."
