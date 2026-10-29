"""
Evaluation and benchmarking framework for OverTakeGuard.
Computes ADAS safety metrics: Unsafe-Advice Rate, Hazard Recall, False Warnings,
Warning Lead Time, Abort Success Rate, and Safety Margins.
Exports structured benchmark reports to JSON and CSV.
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from overtakeguard.models import AdvisoryLevel, ManeuverPhase
from overtakeguard.scenarios import SCENARIO_REGISTRY


class ADASBenchmarkSuite:
    def __init__(self, ground_truth_hazard_ttc_threshold: float = 6.5):
        self.hazard_ttc_thresh = ground_truth_hazard_ttc_threshold

    def evaluate_simulation_run(
        self,
        history: List[Dict[str, Any]],
        system_type: str = "overtakeguard",  # "overtakeguard", "ttc_baseline", "naive_baseline"
    ) -> Dict[str, Any]:
        """
        Computes safety metrics for a single simulation trajectory.
        """
        total_steps = len(history)
        if total_steps == 0:
            return {}

        true_hazard_steps = 0
        true_safe_steps = 0

        # Confusion matrix counters
        tp = 0  # Hazard present & system gave warning (WAIT / CAUTION / ABORT)
        fn = 0  # Hazard present but system gave SAFE (UNSAFE ADVICE!)
        fp = 0  # Safe road but system gave WAIT/ABORT
        tn = 0  # Safe road & system gave SAFE

        warning_timestamps: List[float] = []
        abort_triggered = False
        abort_successful = False

        min_clearance = 999.0
        min_time_margin = 999.0

        time_at_cpa = 0.0

        for step in history:
            t = step["time"]
            ego_x = step["ego_x"]
            ego_vx = step["ego_vx"]
            phase = step["phase"]

            # Ground truth hazard determination
            # A step is hazardous if any oncoming vehicle exists with true closing TTC < threshold
            # or if collision occurred, or if vehicle is passing alongside.
            is_gt_hazard = False
            gt_min_ttc = 999.0

            for on in step["oncoming"]:
                dx = on["x"] - ego_x
                closing_v = ego_vx - on["vx"]  # on["vx"] is negative
                if dx > 0 and closing_v > 0.5:
                    ttc = dx / closing_v
                    if ttc < gt_min_ttc:
                        gt_min_ttc = ttc

            is_alongside = any(abs(on["x"] - ego_x) < 15.0 for on in step["oncoming"])
            if (gt_min_ttc < self.hazard_ttc_thresh) or is_alongside or step["is_collision"]:
                is_gt_hazard = True

            # Advisory from the chosen system
            if system_type == "overtakeguard":
                adv = step["advisory"].level.value
                t_margin = step["advisory"].safety_margin_time
            elif system_type == "ttc_baseline":
                adv = step["ttc_baseline_level"]
                t_margin = (gt_min_ttc - 6.0)
            else:  # naive_baseline
                adv = step["naive_baseline_level"]
                t_margin = (gt_min_ttc - 6.0)

            is_system_warning = adv in (AdvisoryLevel.WAIT.value, AdvisoryLevel.CAUTION.value, AdvisoryLevel.ABORT_NOW.value)
            is_system_safe = (adv == AdvisoryLevel.SAFE_TO_OVERTAKE.value)

            if is_gt_hazard:
                true_hazard_steps += 1
                if is_system_warning:
                    tp += 1
                    warning_timestamps.append(t)
                else:
                    fn += 1  # Catastrophic: Said SAFE during genuine hazard
            else:
                true_safe_steps += 1
                if is_system_warning and adv != AdvisoryLevel.CAUTION.value:
                    fp += 1
                else:
                    tn += 1

            if adv == AdvisoryLevel.ABORT_NOW.value:
                abort_triggered = True

            if step["min_clearance_m"] < min_clearance:
                min_clearance = step["min_clearance_m"]
                time_at_cpa = t
            if t_margin < min_time_margin:
                min_time_margin = t_margin

        # Check if abort was completed safely
        last_step = history[-1]
        if abort_triggered:
            abort_successful = (not last_step["is_collision"]) and (last_step["ego_y"] <= 0.2)

        # Metric Calculations
        unsafe_advice_rate = (fn / true_hazard_steps * 100.0) if true_hazard_steps > 0 else 0.0
        hazard_recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 100.0
        false_warning_rate = (fp / true_safe_steps * 100.0) if true_safe_steps > 0 else 0.0
        warning_lead_time_s = max(0.0, time_at_cpa - warning_timestamps[0]) if warning_timestamps else 0.0

        return {
            "system": system_type,
            "total_steps": total_steps,
            "true_hazard_steps": true_hazard_steps,
            "true_safe_steps": true_safe_steps,
            "tp_warnings": tp,
            "fp_warnings": fp,
            "tn_safe": tn,
            "fn_unsafe_advice": fn,
            "unsafe_advice_rate_pct": round(unsafe_advice_rate, 2),
            "hazard_warning_recall_pct": round(hazard_recall, 2),
            "false_warning_rate_pct": round(false_warning_rate, 2),
            "warning_lead_time_s": round(warning_lead_time_s, 2),
            "min_ground_truth_clearance_m": round(min_clearance, 2),
            "min_time_margin_s": round(min_time_margin, 2),
            "abort_triggered": abort_triggered,
            "abort_success": abort_successful,
            "collision_occurred": last_step["is_collision"],
        }

    def run_full_benchmark(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes all 6 standard scenarios for OverTakeGuard, TTC-only, and Naive-Fusion.
        Returns a comparative summary DataFrame and detailed metric dict.
        """
        results_list: List[Dict[str, Any]] = []

        systems = ["overtakeguard", "ttc_baseline", "naive_baseline"]

        for sc_name, sc_info in SCENARIO_REGISTRY.items():
            builder = sc_info["builder"]
            steps = sc_info["steps"]

            # Run once to generate synchronized step history
            sim = builder()
            for _ in range(steps):
                sim.step()

            history = sim.history

            for sys in systems:
                metrics = self.evaluate_simulation_run(history, system_type=sys)
                metrics["scenario"] = sc_name
                results_list.append(metrics)

        df = pd.DataFrame(results_list)
        return df, {"benchmarks": results_list}

    def export_benchmark_reports(
        self,
        output_dir: str = "reports"
    ) -> Tuple[str, str]:
        """Runs full benchmark and exports CSV and JSON summaries."""
        os.makedirs(output_dir, exist_ok=True)
        df, report_dict = self.run_full_benchmark()

        csv_path = os.path.join(output_dir, "adas_benchmark_comparison.csv")
        json_path = os.path.join(output_dir, "adas_benchmark_comparison.json")

        df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

        return csv_path, json_path
