import pytest
import numpy as np
from overtakeguard.models import (
    VehicleState,
    VehicleType,
    WeatherCondition,
    DetectedTarget,
    SensorObservation
)
from overtakeguard.sensors import CameraSensor, RadarSensor
from overtakeguard.fusion import SensorFusionEngine


def test_sensor_fusion_concordance():
    fusion = SensorFusionEngine()

    cam_target = DetectedTarget(
        target_id="ONCOMING_1",
        range_x=120.0,
        range_y=3.7,
        rel_vx=-45.0,
        abs_vx=-23.0,
        confidence=0.85,
        sensor_source="camera",
        variance_x=2.5,
        variance_vx=1.0,
    )
    cam_obs = SensorObservation(
        sensor_name="Camera",
        timestamp=0.0,
        detected_targets=[cam_target],
        sensor_health=0.9,
        field_of_view_deg=65.0,
        max_range_m=180.0,
        weather_impact_factor=1.0,
    )

    rad_target = DetectedTarget(
        target_id="ONCOMING_1",
        range_x=121.0,
        range_y=3.8,
        rel_vx=-45.5,
        abs_vx=-23.5,
        confidence=0.92,
        sensor_source="radar",
        variance_x=0.4,
        variance_vx=0.2,
    )
    rad_obs = SensorObservation(
        sensor_name="Radar",
        timestamp=0.0,
        detected_targets=[rad_target],
        sensor_health=0.95,
        field_of_view_deg=35.0,
        max_range_m=250.0,
        weather_impact_factor=1.0,
    )

    fused, trust, aleatoric_unc, conflict = fusion.fuse(cam_obs, rad_obs)
    assert len(fused) == 1
    f_tgt = fused[0]
    assert f_tgt.target_id == "ONCOMING_1"
    assert f_tgt.is_conflict is False
    assert trust > 0.8
    assert aleatoric_unc < 0.25
    # Fused range should be between cam and rad range
    assert 119.5 <= f_tgt.range_x <= 121.5


def test_sensor_conflict_detection():
    fusion = SensorFusionEngine(association_gate_dist_m=10.0)

    # Disagreeing distances: Camera sees at 150m, Radar sees at 90m
    cam_target = DetectedTarget(
        target_id="CAR_X",
        range_x=150.0,
        range_y=3.7,
        rel_vx=-40.0,
        abs_vx=-18.0,
        confidence=0.8,
        sensor_source="camera",
        variance_x=4.0,
        variance_vx=1.5,
    )
    cam_obs = SensorObservation(
        sensor_name="Camera",
        timestamp=0.0,
        detected_targets=[cam_target],
        sensor_health=0.85,
        field_of_view_deg=65.0,
        max_range_m=180.0,
        weather_impact_factor=1.0,
    )

    rad_target = DetectedTarget(
        target_id="CAR_X",
        range_x=90.0,
        range_y=3.7,
        rel_vx=-40.0,
        abs_vx=-18.0,
        confidence=0.85,
        sensor_source="radar",
        variance_x=0.5,
        variance_vx=0.3,
    )
    rad_obs = SensorObservation(
        sensor_name="Radar",
        timestamp=0.0,
        detected_targets=[rad_target],
        sensor_health=0.9,
        field_of_view_deg=35.0,
        max_range_m=250.0,
        weather_impact_factor=1.0,
    )

    fused, trust, aleatoric_unc, conflict = fusion.fuse(cam_obs, rad_obs)
    assert len(fused) == 1
    assert fused[0].is_conflict is True
    # In safety critical ADAS, conservative range is taken (the closer one: 90m)
    assert fused[0].range_x == 90.0
    assert aleatoric_unc > 0.35  # elevated uncertainty due to conflict


def test_stale_track_memory_and_extrapolation():
    fusion = SensorFusionEngine(max_stale_frames=3)

    # Frame 1: Target is clearly visible to radar
    target = DetectedTarget(
        target_id="ONCOMING_CAR",
        range_x=100.0,
        range_y=3.75,
        rel_vx=-40.0,
        abs_vx=-18.0,
        confidence=0.90,
        sensor_source="radar",
        variance_x=0.5,
        variance_vx=0.2,
    )
    rad_obs1 = SensorObservation(
        sensor_name="Radar",
        timestamp=0.0,
        detected_targets=[target],
        sensor_health=0.95,
        field_of_view_deg=35.0,
        max_range_m=250.0,
        weather_impact_factor=1.0,
    )
    cam_empty = SensorObservation(
        sensor_name="Camera",
        timestamp=0.0,
        detected_targets=[],
        sensor_health=0.90,
        field_of_view_deg=65.0,
        max_range_m=180.0,
        weather_impact_factor=1.0,
    )

    fused1, _, _, _ = fusion.fuse(cam_empty, rad_obs1)
    assert len(fused1) == 1
    assert fused1[0].target_id == "ONCOMING_CAR"

    # Frame 2: Target is abruptly dropped by both sensors (e.g. spray / glitch)
    rad_empty = SensorObservation(
        sensor_name="Radar",
        timestamp=0.1,
        detected_targets=[],
        sensor_health=0.95,
        field_of_view_deg=35.0,
        max_range_m=250.0,
        weather_impact_factor=1.0,
    )

    fused2, trust2, aleatoric_unc2, _ = fusion.fuse(cam_empty, rad_empty)
    # Target should be preserved via stale track memory
    assert len(fused2) == 1
    stale_tgt = fused2[0]
    assert stale_tgt.target_id == "ONCOMING_CAR"
    assert stale_tgt.sensor_source == "stale_track_memory"
    assert stale_tgt.is_conflict is True
    # Range should be extrapolated forward: 100.0 + (-40.0 * 0.1) = 96.0m
    assert stale_tgt.range_x < 100.0
    assert stale_tgt.confidence < target.confidence  # Confidence penalized
    assert stale_tgt.variance_x > target.variance_x  # Variance elevated


def test_low_confidence_and_noisy_fusion():
    fusion = SensorFusionEngine()

    # Camera with low confidence / high variance (e.g. rain blur)
    cam_noisy = DetectedTarget(
        target_id="CAR_RAIN",
        range_x=120.0,
        range_y=3.7,
        rel_vx=-30.0,
        abs_vx=-10.0,
        confidence=0.40,
        sensor_source="camera",
        variance_x=20.0,  # very high variance
        variance_vx=8.0,
    )
    cam_obs = SensorObservation(
        sensor_name="Camera",
        timestamp=0.0,
        detected_targets=[cam_noisy],
        sensor_health=0.50,
        field_of_view_deg=65.0,
        max_range_m=120.0,
        weather_impact_factor=2.0,
    )

    # Radar with high confidence / low variance
    rad_sharp = DetectedTarget(
        target_id="CAR_RAIN",
        range_x=105.0,
        range_y=3.75,
        rel_vx=-32.0,
        abs_vx=-12.0,
        confidence=0.90,
        sensor_source="radar",
        variance_x=0.5,  # low variance
        variance_vx=0.2,
    )
    rad_obs = SensorObservation(
        sensor_name="Radar",
        timestamp=0.0,
        detected_targets=[rad_sharp],
        sensor_health=0.85,
        field_of_view_deg=35.0,
        max_range_m=200.0,
        weather_impact_factor=1.2,
    )

    fused, trust, aleatoric_unc, _ = fusion.fuse(cam_obs, rad_obs)
    assert len(fused) == 1
    # Inverse variance weighting should weight radar heavily (near 105m, far from 120m)
    assert abs(fused[0].range_x - 105.0) < 2.0
    assert fused[0].variance_x < rad_sharp.variance_x

