import pytest
from overtakeguard.scenarios import SCENARIO_REGISTRY
from overtakeguard.evaluation import ADASBenchmarkSuite
from overtakeguard.models import AdvisoryLevel


def test_scenario_1_safe():
    info = SCENARIO_REGISTRY["Scenario 1: Clear Road & Safe Overtake"]
    sim = info["builder"]()
    for _ in range(info["steps"]):
        sim.step()

    suite = ADASBenchmarkSuite()
    metrics = suite.evaluate_simulation_run(sim.history, system_type="overtakeguard")
    assert metrics["unsafe_advice_rate_pct"] == 0.0
    assert metrics["collision_occurred"] is False


def test_scenario_2_approaching():
    info = SCENARIO_REGISTRY["Scenario 2: Approaching Oncoming Vehicle"]
    sim = info["builder"]()
    for _ in range(info["steps"]):
        sim.step()

    suite = ADASBenchmarkSuite()
    metrics = suite.evaluate_simulation_run(sim.history, system_type="overtakeguard")
    assert metrics["unsafe_advice_rate_pct"] == 0.0
    assert metrics["hazard_warning_recall_pct"] > 90.0


def test_scenario_3_occlusion_awareness():
    info = SCENARIO_REGISTRY["Scenario 3: Hidden Oncoming Vehicle Behind Truck"]
    sim = info["builder"]()
    for _ in range(info["steps"]):
        sim.step()

    suite = ADASBenchmarkSuite()
    # OverTakeGuard should have 0% unsafe advice because it recognizes truck occlusion
    otg_metrics = suite.evaluate_simulation_run(sim.history, system_type="overtakeguard")
    assert otg_metrics["unsafe_advice_rate_pct"] == 0.0

    # In contrast, the TTC-only baseline fails because it is blind to occlusions
    ttc_metrics = suite.evaluate_simulation_run(sim.history, system_type="ttc_baseline")
    assert ttc_metrics["unsafe_advice_rate_pct"] > 0.0


def test_scenario_4_abort_action():
    info = SCENARIO_REGISTRY["Scenario 4: Emerging Hazard During Overtake (Abort)"]
    sim = info["builder"]()
    for _ in range(info["steps"]):
        sim.step()

    suite = ADASBenchmarkSuite()
    metrics = suite.evaluate_simulation_run(sim.history, system_type="overtakeguard")
    assert metrics["abort_triggered"] is True
    assert metrics["abort_success"] is True
    assert metrics["collision_occurred"] is False


def test_scenario_5_rain_degraded_sensors():
    info = SCENARIO_REGISTRY["Scenario 5: Radar Degradation in Heavy Rain"]
    sim = info["builder"]()
    for _ in range(info["steps"]):
        sim.step()

    suite = ADASBenchmarkSuite()
    metrics = suite.evaluate_simulation_run(sim.history, system_type="overtakeguard")
    assert metrics["unsafe_advice_rate_pct"] == 0.0


def test_scenario_6_sensor_conflict():
    info = SCENARIO_REGISTRY["Scenario 6: Conflicting Camera & Radar (Glare)"]
    sim = info["builder"]()
    for _ in range(info["steps"]):
        sim.step()

    suite = ADASBenchmarkSuite()
    metrics = suite.evaluate_simulation_run(sim.history, system_type="overtakeguard")
    assert metrics["unsafe_advice_rate_pct"] == 0.0
