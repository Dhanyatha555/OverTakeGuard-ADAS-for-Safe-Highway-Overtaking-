"""
Sensor Fusion and Uncertainty Estimation for OverTakeGuard.
Fuses Camera and Radar observations, detects cross-sensor contradictions,
calculates aleatoric/epistemic uncertainty, and applies fail-safe hypotheses.
"""

from __future__ import annotations
import math
from typing import List, Dict, Tuple, Optional
from overtakeguard.models import (
    SensorObservation,
    DetectedTarget,
    WeatherCondition
)


class SensorFusionEngine:
    def __init__(
        self,
        association_gate_dist_m: float = 12.0,
        vel_discrepancy_threshold_ms: float = 4.0,
        max_stale_frames: int = 3,
    ):
        self.association_gate_dist_m = association_gate_dist_m
        self.vel_discrepancy_threshold_ms = vel_discrepancy_threshold_ms
        self.max_stale_frames = max_stale_frames
        self.track_history: Dict[str, Dict[str, Any]] = {}

    def fuse(
        self,
        camera_obs: SensorObservation,
        radar_obs: SensorObservation,
        weather: WeatherCondition = WeatherCondition.CLEAR,
    ) -> Tuple[List[DetectedTarget], float, float, float]:
        """
        Fuses camera and radar observations.
        Returns:
            (fused_targets, composite_sensor_trust, aleatoric_uncertainty, conflict_score)
        """
        fused_targets: List[DetectedTarget] = []
        cam_targets = {t.target_id: t for t in camera_obs.detected_targets}
        rad_targets = {t.target_id: t for t in radar_obs.detected_targets}

        all_target_ids = set(cam_targets.keys()).union(set(rad_targets.keys()))
        total_conflicts = 0
        total_associations = 0

        # Base sensor weights from operational health
        w_cam = max(0.05, camera_obs.sensor_health)
        w_rad = max(0.05, radar_obs.sensor_health)

        for tid in all_target_ids:
            has_cam = tid in cam_targets
            has_rad = tid in rad_targets

            if has_cam and has_rad:
                total_associations += 1
                c_tgt = cam_targets[tid]
                r_tgt = rad_targets[tid]

                # Check for physical discrepancies
                dist_diff = abs(c_tgt.range_x - r_tgt.range_x)
                vel_diff = abs(c_tgt.rel_vx - r_tgt.rel_vx)

                is_conflict = (dist_diff > self.association_gate_dist_m) or (vel_diff > self.vel_discrepancy_threshold_ms)
                if is_conflict:
                    total_conflicts += 1

                # Inverse variance weighting (Optimal Kalman/Bayesian Fusion)
                inv_var_cam_x = 1.0 / max(0.01, c_tgt.variance_x)
                inv_var_rad_x = 1.0 / max(0.01, r_tgt.variance_x)
                fused_var_x = 1.0 / (inv_var_cam_x + inv_var_rad_x)
                fused_range_x = fused_var_x * (c_tgt.range_x * inv_var_cam_x + r_tgt.range_x * inv_var_rad_x)

                inv_var_cam_vx = 1.0 / max(0.01, c_tgt.variance_vx)
                inv_var_rad_vx = 1.0 / max(0.01, r_tgt.variance_vx)
                fused_var_vx = 1.0 / (inv_var_cam_vx + inv_var_rad_vx)
                fused_rel_vx = fused_var_vx * (c_tgt.rel_vx * inv_var_cam_vx + r_tgt.rel_vx * inv_var_rad_vx)

                # Lateral position: Camera has much sharper angular resolution
                fused_range_y = 0.8 * c_tgt.range_y + 0.2 * r_tgt.range_y

                # Under safety-critical ADAS: If conflict exists, take conservative estimate
                if is_conflict:
                    # Conservative: pick closer range and more negative closing velocity
                    fused_range_x = min(c_tgt.range_x, r_tgt.range_x)
                    fused_rel_vx = min(c_tgt.rel_vx, r_tgt.rel_vx)
                    fused_conf = 0.5 * min(c_tgt.confidence, r_tgt.confidence)
                else:
                    # Concordant observations bolster confidence
                    fused_conf = min(0.99, max(c_tgt.confidence, r_tgt.confidence) + 0.1)

                fused_targets.append(
                    DetectedTarget(
                        target_id=tid,
                        range_x=round(float(fused_range_x), 2),
                        range_y=round(float(fused_range_y), 2),
                        rel_vx=round(float(fused_rel_vx), 2),
                        abs_vx=round(float(c_tgt.abs_vx * 0.3 + r_tgt.abs_vx * 0.7), 2),
                        confidence=round(float(fused_conf), 3),
                        sensor_source="fused (cam+rad)",
                        variance_x=round(float(fused_var_x), 3),
                        variance_vx=round(float(fused_var_vx), 3),
                        is_conflict=is_conflict,
                    )
                )

            elif has_cam and not has_rad:
                # Camera only detection: Could be radar rain attenuation or optical false alarm
                c_tgt = cam_targets[tid]
                # If radar health is low (e.g. heavy rain), trust camera more
                is_suspicious = (radar_obs.sensor_health > 0.7) and (c_tgt.range_x < radar_obs.max_range_m * 0.8)
                if is_suspicious:
                    total_conflicts += 1

                penalized_conf = c_tgt.confidence * (0.65 if is_suspicious else 0.85)
                fused_targets.append(
                    DetectedTarget(
                        target_id=tid,
                        range_x=c_tgt.range_x,
                        range_y=c_tgt.range_y,
                        rel_vx=c_tgt.rel_vx,
                        abs_vx=c_tgt.abs_vx,
                        confidence=round(float(penalized_conf), 3),
                        sensor_source="camera-only",
                        variance_x=c_tgt.variance_x * 1.5,
                        variance_vx=c_tgt.variance_vx * 1.5,
                        is_conflict=is_suspicious,
                    )
                )

            else:  # has_rad and not has_cam
                # Radar only detection: Could be camera glare/fog blindness or radar clutter
                r_tgt = rad_targets[tid]
                is_suspicious = (camera_obs.sensor_health > 0.7) and (r_tgt.range_x < camera_obs.max_range_m * 0.8)
                if is_suspicious:
                    total_conflicts += 1

                penalized_conf = r_tgt.confidence * (0.70 if is_suspicious else 0.88)
                fused_targets.append(
                    DetectedTarget(
                        target_id=tid,
                        range_x=r_tgt.range_x,
                        range_y=r_tgt.range_y,
                        rel_vx=r_tgt.rel_vx,
                        abs_vx=r_tgt.abs_vx,
                        confidence=round(float(penalized_conf), 3),
                        sensor_source="radar-only",
                        variance_x=r_tgt.variance_x * 1.4,
                        variance_vx=r_tgt.variance_vx * 1.2,
                        is_conflict=is_suspicious,
                    )
                )

        # Check for missing/stale tracks from track history
        current_seen_ids = {t.target_id for t in fused_targets}
        stale_tracks_count = 0
        new_track_history: Dict[str, Dict[str, Any]] = {}

        for t in fused_targets:
            new_track_history[t.target_id] = {
                "target": t,
                "stale_frames": 0,
            }

        # Check previously tracked targets that disappeared abruptly
        for prev_id, prev_info in self.track_history.items():
            if prev_id not in current_seen_ids:
                stale_cnt = prev_info["stale_frames"] + 1
                if stale_cnt <= self.max_stale_frames:
                    prev_t: DetectedTarget = prev_info["target"]
                    extrapolated_range_x = max(1.0, prev_t.range_x + prev_t.rel_vx * 0.1)
                    stale_conf = max(0.15, prev_t.confidence * 0.70)
                    stale_target = DetectedTarget(
                        target_id=prev_id,
                        range_x=round(float(extrapolated_range_x), 2),
                        range_y=prev_t.range_y,
                        rel_vx=prev_t.rel_vx,
                        abs_vx=prev_t.abs_vx,
                        confidence=round(float(stale_conf), 3),
                        sensor_source="stale_track_memory",
                        variance_x=round(float(prev_t.variance_x * 1.5), 3),
                        variance_vx=round(float(prev_t.variance_vx * 1.3), 3),
                        is_conflict=True,
                    )
                    fused_targets.append(stale_target)
                    stale_tracks_count += 1
                    new_track_history[prev_id] = {
                        "target": stale_target,
                        "stale_frames": stale_cnt,
                    }

        self.track_history = new_track_history

        # Compute composite metrics
        avg_health = 0.5 * (camera_obs.sensor_health + radar_obs.sensor_health)
        conflict_rate = (total_conflicts / max(1, len(all_target_ids))) if all_target_ids else 0.0

        sensor_trust = max(0.1, min(1.0, avg_health * (1.0 - 0.45 * conflict_rate)))
        # Aleatoric uncertainty stems from sensor noise, degraded weather, conflicts, and stale tracks
        stale_penalty = min(0.3, stale_tracks_count * 0.15)
        aleatoric_unc = max(0.05, min(0.95, (1.0 - avg_health) * 0.6 + conflict_rate * 0.4 + stale_penalty))

        return fused_targets, round(sensor_trust, 3), round(aleatoric_unc, 3), round(conflict_rate, 3)
