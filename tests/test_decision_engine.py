import pytest
from overtakeguard.models import (
    VehicleState,
    VehicleType,
    ManeuverPhase,
    AdvisoryLevel,
    DetectedTarget,
    WeatherCondition
)
from overtakeguard.decision_engine import RollingHorizonDecisionEngine


def test_decision_engine_safe_clear_road():
    engine = RollingHorizonDecisionEngine()
    ego = VehicleState(
        id="EGO",
        vehicle_type=VehicleType.EGO,
        x=15.0,
        y=0.0,
        vx=22.0,
        length=4.5,
    )
    lead = VehicleState(
        id="LEAD",
        vehicle_type=VehicleType.LEAD,
        x=50.0,
        y=0.0,
        vx=17.0,
        length=4.5,
        is_occluding=False,
    )

    advisory = engine.evaluate(
        timestamp=0.0,
        ego=ego,
        lead=lead,
        all_vehicles=[ego, lead],
        fused_targets=[],
        weather=WeatherCondition.CLEAR,
        current_phase=ManeuverPhase.FOLLOWING,
        sensor_trust_index=0.95,
        aleatoric_uncertainty=0.05,
    )

    assert advisory.level == AdvisoryLevel.SAFE_TO_OVERTAKE
    assert advisory.safety_margin_time > 0.0
    assert "SAFE TO OVERTAKE" in advisory.primary_reason


def test_decision_engine_wait_approaching_hazard():
    engine = RollingHorizonDecisionEngine()
    ego = VehicleState(
        id="EGO",
        vehicle_type=VehicleType.EGO,
        x=15.0,
        y=0.0,
        vx=22.0,
        length=4.5,
    )
    lead = VehicleState(
        id="LEAD",
        vehicle_type=VehicleType.LEAD,
        x=50.0,
        y=0.0,
        vx=17.0,
        length=4.5,
    )
    oncoming_tgt = DetectedTarget(
        target_id="ONCOMING_CAR",
        range_x=130.0,
        range_y=3.75,
        rel_vx=-48.0,
        abs_vx=-26.0,
        confidence=0.95,
        sensor_source="fused",
        variance_x=0.5,
        variance_vx=0.2,
    )

    advisory = engine.evaluate(
        timestamp=0.0,
        ego=ego,
        lead=lead,
        all_vehicles=[ego, lead],
        fused_targets=[oncoming_tgt],
        weather=WeatherCondition.CLEAR,
        current_phase=ManeuverPhase.FOLLOWING,
        sensor_trust_index=0.92,
        aleatoric_uncertainty=0.08,
    )

    assert advisory.level == AdvisoryLevel.WAIT
    assert "Approaching oncoming hazard" in advisory.primary_reason


def test_decision_engine_abort_now_in_passing_phase():
    engine = RollingHorizonDecisionEngine()
    ego = VehicleState(
        id="EGO",
        vehicle_type=VehicleType.EGO,
        x=45.0,
        y=3.75,
        vx=26.0,
        length=4.5,
    )
    lead = VehicleState(
        id="LEAD",
        vehicle_type=VehicleType.LEAD,
        x=50.0,
        y=0.0,
        vx=17.0,
        length=4.5,
    )
    # Sudden close oncoming vehicle
    hazard = DetectedTarget(
        target_id="SUDDEN_HAZARD",
        range_x=80.0,
        range_y=3.75,
        rel_vx=-52.0,
        abs_vx=-26.0,
        confidence=0.9,
        sensor_source="fused",
        variance_x=0.4,
        variance_vx=0.2,
    )

    advisory = engine.evaluate(
        timestamp=2.0,
        ego=ego,
        lead=lead,
        all_vehicles=[ego, lead],
        fused_targets=[hazard],
        weather=WeatherCondition.CLEAR,
        current_phase=ManeuverPhase.PASSING,
        sensor_trust_index=0.9,
        aleatoric_uncertainty=0.1,
    )

    # When abort is feasible, it should advise ABORT NOW
    assert advisory.level == AdvisoryLevel.ABORT_NOW
    assert "ABORT NOW" in advisory.primary_reason
    assert advisory.abort_state.is_feasible is True


def test_benign_objects_filtering():
    engine = RollingHorizonDecisionEngine()
    ego = VehicleState(
        id="EGO",
        vehicle_type=VehicleType.EGO,
        x=15.0,
        y=0.0,
        vx=22.0,
        length=4.5,
    )
    lead = VehicleState(
        id="LEAD",
        vehicle_type=VehicleType.LEAD,
        x=50.0,
        y=0.0,
        vx=18.0,
        length=4.5,
        is_occluding=False,
    )

    # 1. Lead car detection ahead in same lane
    target_lead = DetectedTarget(
        target_id="LEAD",
        range_x=35.0,
        range_y=0.0,
        rel_vx=-4.0,
        abs_vx=18.0,
        confidence=0.95,
        sensor_source="fused",
    )
    # 2. Roadside shoulder sign / pole (y = 6.2m, outside travel lanes)
    target_pole = DetectedTarget(
        target_id="ROAD_SIGN_POLE",
        range_x=45.0,
        range_y=6.2,
        rel_vx=-22.0,
        abs_vx=0.0,
        confidence=0.90,
        sensor_source="fused",
    )
    # 3. Same-direction car cruising ahead in ego lane
    target_same_dir = DetectedTarget(
        target_id="CAR_AHEAD",
        range_x=80.0,
        range_y=0.1,
        rel_vx=-2.0,
        abs_vx=20.0,
        confidence=0.92,
        sensor_source="fused",
    )

    advisory = engine.evaluate(
        timestamp=0.0,
        ego=ego,
        lead=lead,
        all_vehicles=[ego, lead],
        fused_targets=[target_lead, target_pole, target_same_dir],
        weather=WeatherCondition.CLEAR,
        current_phase=ManeuverPhase.FOLLOWING,
        sensor_trust_index=0.95,
        aleatoric_uncertainty=0.05,
    )

    # Benign objects must NOT trigger critical warnings (WAIT / ABORT)
    assert len(advisory.detected_hazards) == 0
    assert advisory.level == AdvisoryLevel.SAFE_TO_OVERTAKE
    assert "SAFE TO OVERTAKE" in advisory.primary_reason


def test_road_geometry_sight_limit_advisory():
    engine = RollingHorizonDecisionEngine()
    ego = VehicleState(
        id="EGO",
        vehicle_type=VehicleType.EGO,
        x=15.0,
        y=0.0,
        vx=22.0,
        length=4.5,
    )
    lead = VehicleState(
        id="LEAD",
        vehicle_type=VehicleType.LEAD,
        x=50.0,
        y=0.0,
        vx=17.0,
        length=4.5,
        is_occluding=False,
    )

    # Road crest limits forward sight distance to 60m
    advisory = engine.evaluate(
        timestamp=0.0,
        ego=ego,
        lead=lead,
        all_vehicles=[ego, lead],
        fused_targets=[],
        weather=WeatherCondition.CLEAR,
        current_phase=ManeuverPhase.FOLLOWING,
        sensor_trust_index=0.95,
        aleatoric_uncertainty=0.05,
        geometry_sight_limit_m=60.0,
    )

    assert advisory.level == AdvisoryLevel.WAIT
    assert "Sight distance limited" in advisory.primary_reason or "LATENT HAZARD RISK" in advisory.primary_reason


def test_ponr_prevents_abort_advisory():
    engine = RollingHorizonDecisionEngine()
    # Ego has passed lead vehicle: ego at x=65, lead at x=45 (12m truck)
    ego = VehicleState(
        id="EGO",
        vehicle_type=VehicleType.EGO,
        x=65.0,
        y=3.75,
        vx=30.0,
        length=4.5,
    )
    lead = VehicleState(
        id="LEAD_TRUCK",
        vehicle_type=VehicleType.LEAD,
        x=45.0,
        y=0.0,
        vx=18.0,
        length=12.0,
    )
    # Sudden oncoming hazard detected
    hazard = DetectedTarget(
        target_id="ONCOMING_HAZARD",
        range_x=90.0,
        range_y=3.75,
        rel_vx=-50.0,
        abs_vx=-20.0,
        confidence=0.92,
        sensor_source="fused",
    )

    advisory = engine.evaluate(
        timestamp=3.0,
        ego=ego,
        lead=lead,
        all_vehicles=[ego, lead],
        fused_targets=[hazard],
        weather=WeatherCondition.CLEAR,
        current_phase=ManeuverPhase.PASSING,
        sensor_trust_index=0.9,
        aleatoric_uncertainty=0.1,
    )

    # Returning to lane is infeasible (Point of No Return), so ABORT NOW must NOT be issued
    assert advisory.abort_state.is_feasible is False
    assert advisory.abort_state.point_of_no_return is True
    assert advisory.level != AdvisoryLevel.ABORT_NOW
    assert advisory.level == AdvisoryLevel.CAUTION
    assert "POINT OF NO RETURN" in advisory.primary_reason

