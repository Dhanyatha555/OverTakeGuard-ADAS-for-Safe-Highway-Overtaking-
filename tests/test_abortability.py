import pytest
from overtakeguard.models import VehicleState, VehicleType, ManeuverPhase
from overtakeguard.abortability import AbortabilityEngine


def test_abort_feasible_when_behind_lead():
    engine = AbortabilityEngine()
    ego = VehicleState(
        id="EGO",
        vehicle_type=VehicleType.EGO,
        x=20.0,
        y=1.5,
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
    )

    state = engine.evaluate_abort_feasibility(
        ego=ego,
        lead=lead,
        phase=ManeuverPhase.LANE_CHANGE_OUT,
        time_to_complete_forward=7.0,
    )
    assert state.is_feasible is True
    assert state.point_of_no_return is False
    assert state.required_deceleration < engine.max_comfortable_decel


def test_abort_point_of_no_return():
    engine = AbortabilityEngine(max_emergency_decel_ms2=6.0)
    # Ego is well ahead of lead vehicle
    ego = VehicleState(
        id="EGO",
        vehicle_type=VehicleType.EGO,
        x=65.0,  # Ego is 15m ahead of lead vehicle center
        y=3.75,
        vx=30.0,
        length=4.5,
    )
    lead = VehicleState(
        id="LEAD",
        vehicle_type=VehicleType.LEAD,
        x=45.0,
        y=0.0,
        vx=18.0,
        length=12.0,  # 12m truck
    )

    # Completing forward is quick (1.2s), while aborting backwards requires massive braking
    state = engine.evaluate_abort_feasibility(
        ego=ego,
        lead=lead,
        phase=ManeuverPhase.PASSING,
        time_to_complete_forward=1.2,
    )
    assert state.point_of_no_return is True
    assert state.is_feasible is False
    assert "POINT OF NO RETURN" in state.reason
