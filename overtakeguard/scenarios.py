"""
Curated reproducible benchmark scenarios for OverTakeGuard.
Includes all 6 required benchmark cases plus full parameter configuration.
"""

from __future__ import annotations
from typing import Dict, Any, Callable
from overtakeguard.models import (
    VehicleState,
    VehicleType,
    WeatherCondition
)
from overtakeguard.simulation import HighwaySimulation


def scenario_1_clear_road() -> HighwaySimulation:
    """Scenario 1: Clear road and safe overtake under ideal daylight conditions."""
    ego = VehicleState(
        id="EGO",
        vehicle_type=VehicleType.EGO,
        x=15.0,
        y=0.0,
        vx=22.2,  # 80 km/h
        ax=0.0,
        length=4.6,
        width=1.8,
    )
    lead = VehicleState(
        id="LEAD_CAR",
        vehicle_type=VehicleType.LEAD,
        x=50.0,
        y=0.0,
        vx=17.0,  # 61 km/h (slow car)
        ax=0.0,
        length=4.5,
        width=1.8,
        is_occluding=False,
    )
    # Empty oncoming lane
    return HighwaySimulation(
        ego=ego,
        lead=lead,
        oncoming_vehicles=[],
        weather=WeatherCondition.CLEAR,
        seed=101,
    )


def scenario_2_approaching_oncoming() -> HighwaySimulation:
    """Scenario 2: Clear oncoming vehicle approaching rapidly in opposing lane."""
    ego = VehicleState(
        id="EGO",
        vehicle_type=VehicleType.EGO,
        x=15.0,
        y=0.0,
        vx=22.2,
        ax=0.0,
        length=4.6,
        width=1.8,
    )
    lead = VehicleState(
        id="LEAD_CAR",
        vehicle_type=VehicleType.LEAD,
        x=50.0,
        y=0.0,
        vx=18.0,
        ax=0.0,
        length=4.5,
        width=1.8,
        is_occluding=False,
    )
    oncoming = VehicleState(
        id="ONCOMING_SEDAN",
        vehicle_type=VehicleType.ONCOMING,
        x=220.0,
        y=3.75,
        vx=-25.0,  # 90 km/h towards ego
        ax=0.0,
        length=4.8,
        width=1.8,
    )
    return HighwaySimulation(
        ego=ego,
        lead=lead,
        oncoming_vehicles=[oncoming],
        weather=WeatherCondition.CLEAR,
        seed=100,
    )


def scenario_3_hidden_truck_occlusion() -> HighwaySimulation:
    """
    Scenario 3: Large lead semi-trailer truck occludes an oncoming high-speed car.
    Crucial test for Occlusion-Awareness: Naive baseline says 'safe', OverTakeGuard says 'WAIT'.
    """
    ego = VehicleState(
        id="EGO",
        vehicle_type=VehicleType.EGO,
        x=15.0,
        y=0.0,
        vx=22.0,
        ax=0.0,
        length=4.6,
        width=1.8,
    )
    lead_truck = VehicleState(
        id="SEMI_TRUCK",
        vehicle_type=VehicleType.LEAD,
        x=45.0,
        y=0.0,
        vx=17.5,
        ax=0.0,
        length=16.0,  # 16-meter long commercial truck
        width=2.5,   # Wide box trailer
        is_occluding=True,
    )
    hidden_oncoming = VehicleState(
        id="HIDDEN_CAR",
        vehicle_type=VehicleType.ONCOMING,
        x=185.0,     # Deep inside the truck's blind spot
        y=3.75,
        vx=-27.78,  # 100 km/h fast vehicle
        ax=0.0,
        length=4.5,
        width=1.8,
        visible_to_driver=False,
    )
    return HighwaySimulation(
        ego=ego,
        lead=lead_truck,
        oncoming_vehicles=[hidden_oncoming],
        weather=WeatherCondition.CLEAR,
        seed=103,
    )


def scenario_4_emerging_hazard_abort() -> HighwaySimulation:
    """
    Scenario 4: Oncoming vehicle suddenly emerges while ego is in opposing lane.
    Crucial test for Abortability Engine: verifies safe deceleration and Point of No Return.
    """
    ego = VehicleState(
        id="EGO",
        vehicle_type=VehicleType.EGO,
        x=30.0,
        y=1.8,      # Already transitioned halfway into passing lane
        vx=25.0,
        ax=1.5,
        length=4.6,
        width=1.8,
    )
    lead = VehicleState(
        id="LEAD_CAR",
        vehicle_type=VehicleType.LEAD,
        x=55.0,
        y=0.0,
        vx=19.0,
        ax=0.0,
        length=4.5,
        width=1.8,
    )
    # Emerges at 165m closing rapidly
    oncoming = VehicleState(
        id="SURPRISE_HAZARD",
        vehicle_type=VehicleType.ONCOMING,
        x=165.0,
        y=3.75,
        vx=-28.0,
        ax=0.0,
        length=4.5,
        width=1.8,
    )
    sim = HighwaySimulation(
        ego=ego,
        lead=lead,
        oncoming_vehicles=[oncoming],
        weather=WeatherCondition.CLEAR,
        auto_pilot_advisory_compliance=True,
        seed=104,
    )
    sim.phase = sim.phase.__class__.LANE_CHANGE_OUT
    return sim


def scenario_5_radar_rain_degradation() -> HighwaySimulation:
    """Scenario 5: Heavy rain causes severe radar attenuation, backscatter clutter, and camera blur."""
    ego = VehicleState(
        id="EGO",
        vehicle_type=VehicleType.EGO,
        x=15.0,
        y=0.0,
        vx=20.0,
        ax=0.0,
        length=4.6,
        width=1.8,
    )
    lead = VehicleState(
        id="LEAD_VAN",
        vehicle_type=VehicleType.LEAD,
        x=48.0,
        y=0.0,
        vx=16.0,
        ax=0.0,
        length=6.0,
        width=2.0,
    )
    # Distant oncoming vehicle in rain
    oncoming = VehicleState(
        id="RAIN_GHOST_HAZARD",
        vehicle_type=VehicleType.ONCOMING,
        x=190.0,
        y=3.75,
        vx=-24.0,
        ax=0.0,
        length=4.6,
        width=1.8,
    )
    return HighwaySimulation(
        ego=ego,
        lead=lead,
        oncoming_vehicles=[oncoming],
        weather=WeatherCondition.HEAVY_RAIN,
        seed=105,
    )


def scenario_6_conflicting_sensors() -> HighwaySimulation:
    """
    Scenario 6: Conflicting observations:
    Direct high glare / optical reflection blinds camera, but radar detects an approaching target.
    Tests discordance handling and conservative fail-safe logic.
    """
    ego = VehicleState(
        id="EGO",
        vehicle_type=VehicleType.EGO,
        x=15.0,
        y=0.0,
        vx=22.0,
        ax=0.0,
        length=4.6,
        width=1.8,
    )
    lead = VehicleState(
        id="LEAD_CAR",
        vehicle_type=VehicleType.LEAD,
        x=52.0,
        y=0.0,
        vx=17.5,
        ax=0.0,
        length=4.5,
        width=1.8,
    )
    oncoming = VehicleState(
        id="GLARE_VEHICLE",
        vehicle_type=VehicleType.ONCOMING,
        x=175.0,
        y=3.75,
        vx=-26.0,
        ax=0.0,
        length=4.6,
        width=1.8,
    )
    return HighwaySimulation(
        ego=ego,
        lead=lead,
        oncoming_vehicles=[oncoming],
        weather=WeatherCondition.NIGHT_GLARE,
        seed=106,
    )


SCENARIO_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Scenario 1: Clear Road & Safe Overtake": {
        "builder": scenario_1_clear_road,
        "description": "Ideal visibility, clear opposing lane. Demonstrates baseline safe feasibility advice.",
        "expected_advisory": "SAFE TO OVERTAKE",
        "steps": 60,
    },
    "Scenario 2: Approaching Oncoming Vehicle": {
        "builder": scenario_2_approaching_oncoming,
        "description": "Direct line-of-sight hazard approaching at 90 km/h. Verifies timely WAIT recommendation.",
        "expected_advisory": "WAIT",
        "steps": 40,
    },
    "Scenario 3: Hidden Oncoming Vehicle Behind Truck": {
        "builder": scenario_3_hidden_truck_occlusion,
        "description": "16m box trailer blocks oncoming sightline. Baseline fails blindly; OverTakeGuard detects occlusion.",
        "expected_advisory": "WAIT",
        "steps": 45,
    },
    "Scenario 4: Emerging Hazard During Overtake (Abort)": {
        "builder": scenario_4_emerging_hazard_abort,
        "description": "Hazard suddenly appears during passing phase. Evaluates dynamic deceleration and ABORT NOW.",
        "expected_advisory": "ABORT NOW",
        "steps": 50,
    },
    "Scenario 5: Radar Degradation in Heavy Rain": {
        "builder": scenario_5_radar_rain_degradation,
        "description": "Heavy precipitation attenuates radar and obscures camera. Tests aleatoric uncertainty elevation.",
        "expected_advisory": "CAUTION",
        "steps": 40,
    },
    "Scenario 6: Conflicting Camera & Radar (Glare)": {
        "builder": scenario_6_conflicting_sensors,
        "description": "Camera blinded by glare misses hazard; Radar detects it. Tests conservative fail-safe fusion.",
        "expected_advisory": "WAIT",
        "steps": 40,
    },
}
