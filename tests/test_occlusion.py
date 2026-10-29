import pytest
from overtakeguard.models import VehicleState, VehicleType, WeatherCondition
from overtakeguard.occlusion import OcclusionEngine


def test_occlusion_generation_by_truck():
    engine = OcclusionEngine(opposing_lane_y=3.75, max_horizon_m=350.0)
    ego = VehicleState(
        id="EGO",
        vehicle_type=VehicleType.EGO,
        x=10.0,
        y=0.0,
        vx=20.0,
        length=4.5,
        width=1.8,
    )
    truck = VehicleState(
        id="TRUCK",
        vehicle_type=VehicleType.LEAD,
        x=35.0,
        y=0.0,
        vx=15.0,
        length=16.0,
        width=2.5,
        is_occluding=True,
    )

    zones, sight_dist = engine.compute_occlusions(ego, [ego, truck], WeatherCondition.CLEAR)
    assert len(zones) >= 1
    zone = zones[0]
    assert zone.occluding_vehicle_id == "TRUCK"
    assert zone.depth > 10.0
    assert zone.latent_hazard_risk > 0.3
    assert sight_dist < 350.0


def test_latent_hazard_emergence_evaluation():
    engine = OcclusionEngine(opposing_lane_y=3.75, default_hypothetical_oncoming_speed=27.78)
    ego = VehicleState(
        id="EGO",
        vehicle_type=VehicleType.EGO,
        x=10.0,
        y=0.0,
        vx=22.0,
        length=4.5,
        width=1.8,
    )
    truck = VehicleState(
        id="TRUCK",
        vehicle_type=VehicleType.LEAD,
        x=40.0,
        y=0.0,
        vx=15.0,
        length=15.0,
        width=2.5,
        is_occluding=True,
    )

    zones, _ = engine.compute_occlusions(ego, [ego, truck], WeatherCondition.CLEAR)
    is_critical, emergence_t, explanation = engine.evaluate_latent_hazard_emergence(
        ego=ego,
        occlusion_zones=zones,
        required_maneuver_time=6.5,
    )
    assert is_critical is True
    assert emergence_t < (6.5 + 3.5)
    assert "LATENT HAZARD RISK" in explanation


def test_road_geometry_crest_occlusion():
    engine = OcclusionEngine(opposing_lane_y=3.75, max_horizon_m=350.0)
    ego = VehicleState(
        id="EGO",
        vehicle_type=VehicleType.EGO,
        x=10.0,
        y=0.0,
        vx=22.0,
        length=4.5,
        width=1.8,
    )

    # Road geometry crest limits line-of-sight to 80m ahead of ego
    zones, sight_dist = engine.compute_occlusions(
        ego=ego,
        vehicles=[ego],
        weather=WeatherCondition.CLEAR,
        geometry_sight_limit_m=80.0,
    )

    assert sight_dist <= 80.0
    assert len(zones) >= 1
    geom_zone = next(z for z in zones if z.occluding_vehicle_id == "ROAD_GEOMETRY_CURVE_CREST")
    assert geom_zone.latent_hazard_risk >= 0.8
    assert geom_zone.depth > 0.0

    is_critical, emergence_t, explanation = engine.evaluate_latent_hazard_emergence(
        ego=ego,
        occlusion_zones=zones,
        required_maneuver_time=7.0,
    )
    assert is_critical is True
    assert "LATENT HAZARD RISK" in explanation

