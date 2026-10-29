"""
OverTakeGuard: Interactive ADAS Highway Simulation Dashboard.
A Rolling-Horizon, Uncertainty-Aware Overtaking Feasibility and Abortability Assistant
for Restricted-Visibility Highways.
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from overtakeguard.models import (
    VehicleState,
    VehicleType,
    ManeuverPhase,
    AdvisoryLevel,
    WeatherCondition
)
from overtakeguard.scenarios import SCENARIO_REGISTRY
from overtakeguard.evaluation import ADASBenchmarkSuite
from overtakeguard.simulation import HighwaySimulation


# Configure Streamlit page layout and theme
st.set_page_config(
    page_title="OverTakeGuard | ADAS Highway Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark ADAS Telemetry theme)
st.markdown(
    """
    <style>
    .main {
        background-color: #0d1117;
        color: #e6edf3;
    }
    .advisory-card {
        padding: 18px 24px;
        border-radius: 12px;
        margin-bottom: 16px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }
    .advisory-safe {
        background: linear-gradient(135deg, #1b4d3e, #0f2e24);
        border: 2px solid #2ea043;
        color: #7ee787;
    }
    .advisory-wait {
        background: linear-gradient(135deg, #1f3a5f, #0d213a);
        border: 2px solid #388bfd;
        color: #79c0ff;
    }
    .advisory-caution {
        background: linear-gradient(135deg, #594511, #382b08);
        border: 2px solid #d29922;
        color: #f2cc60;
    }
    .advisory-abort {
        background: linear-gradient(135deg, #6e1c24, #3d0e14);
        border: 2px solid #f85149;
        color: #ff7b72;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(248, 81, 73, 0.7); }
        70% { box-shadow: 0 0 0 14px rgba(248, 81, 73, 0); }
        100% { box-shadow: 0 0 0 0 rgba(248, 81, 73, 0); }
    }
    .metric-badge {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f0f6fc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_highway_canvas(snapshot: dict, occlusion_zones: list, detected_targets: list) -> plt.Figure:
    """Renders 2D Bird's-Eye Highway Canvas."""
    fig, ax = plt.subplots(figsize=(14, 3.4), dpi=140)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    ego_x = snapshot["ego_x"]
    # Dynamic camera window centered slightly ahead of ego
    view_min_x = max(0.0, ego_x - 30.0)
    view_max_x = view_min_x + 240.0

    ax.set_xlim(view_min_x, view_max_x)
    ax.set_ylim(-3.0, 7.0)

    # Highway Surface
    ax.axhspan(-1.875, 5.625, color='#1f242c', zorder=1)

    # Road Borders (Solid White Lines)
    ax.axhline(-1.875, color='#e6edf3', linewidth=2.5, zorder=2)
    ax.axhline(5.625, color='#e6edf3', linewidth=2.5, zorder=2)

    # Lane Divider (Dashed Yellow in center, y = 1.875)
    ax.axhline(1.875, color='#d29922', linestyle='--', linewidth=2.0, dashes=(8, 6), zorder=2)

    # Lane Identifiers
    ax.text(view_min_x + 6, -0.9, "TRAVEL LANE (EGO)", color='#8b949e', fontsize=8, fontweight='bold')
    ax.text(view_min_x + 6, 4.6, "OPPOSING LANE (ONCOMING / OVERTAKE)", color='#8b949e', fontsize=8, fontweight='bold')

    # Draw Occlusion Zones
    for zone in occlusion_zones:
        start_x = max(view_min_x, zone.start_x)
        end_x = min(view_max_x, zone.end_x)
        if end_x > start_x:
            # Occluded corridor in opposing lane
            rect = patches.Rectangle(
                (start_x, 1.875),
                end_x - start_x,
                3.75,
                linewidth=1.5,
                edgecolor='#f85149',
                facecolor='#f85149',
                alpha=0.22,
                hatch='///',
                zorder=3,
            )
            ax.add_patch(rect)
            ax.text(
                (start_x + end_x) / 2.0,
                3.75,
                f"OCCLUSION SHADOW ({zone.occluding_vehicle_id})\nLATENT HAZARD RISK {int(zone.latent_hazard_risk * 100)}%",
                color='#ff7b72',
                fontsize=7.5,
                fontweight='bold',
                ha='center',
                va='center',
                zorder=4,
            )

    # Draw Lead Vehicle
    if snapshot["lead_x"] is not None:
        lead_x = snapshot["lead_x"]
        is_truck = "TRUCK" in snapshot.get("scenario_name", "") or "SEMI" in snapshot.get("scenario_name", "") or snapshot.get("lead_length", 4.5) > 8.0
        length = snapshot.get("lead_length", 16.0 if is_truck else 4.5)
        width = 2.4 if is_truck else 1.8
        color = '#d97706' if is_truck else '#0284c7'

        lead_rect = patches.Rectangle(
            (lead_x - length / 2.0, 0.0 - width / 2.0),
            length,
            width,
            linewidth=1.8,
            edgecolor='#fde047',
            facecolor=color,
            alpha=0.9,
            zorder=5,
        )
        ax.add_patch(lead_rect)
        ax.text(
            lead_x,
            0.0,
            f"LEAD {'TRUCK' if is_truck else 'CAR'}",
            color='#ffffff',
            fontsize=8,
            fontweight='bold',
            ha='center',
            va='center',
            zorder=6,
        )

    # Draw Oncoming Vehicles
    for on in snapshot["oncoming"]:
        on_x = on["x"]
        on_y = on["y"]
        if view_min_x - 10 <= on_x <= view_max_x + 10:
            on_rect = patches.Rectangle(
                (on_x - 2.3, on_y - 0.9),
                4.6,
                1.8,
                linewidth=1.8,
                edgecolor='#f87171',
                facecolor='#dc2626',
                alpha=0.9,
                zorder=5,
            )
            ax.add_patch(on_rect)
            ax.text(
                on_x,
                on_y,
                f"ONCOMING\n{abs(on['vx']) * 3.6:.0f} km/h",
                color='#ffffff',
                fontsize=7,
                fontweight='bold',
                ha='center',
                va='center',
                zorder=6,
            )

    # Draw Ego Vehicle
    ego_y = snapshot["ego_y"]
    ego_rect = patches.Rectangle(
        (ego_x - 2.3, ego_y - 0.9),
        4.6,
        1.8,
        linewidth=2.2,
        edgecolor='#38bdf8',
        facecolor='#2563eb',
        alpha=0.95,
        zorder=7,
    )
    ax.add_patch(ego_rect)
    ax.text(
        ego_x,
        ego_y,
        f"EGO\n{snapshot['ego_speed_kph']:.0f} km/h",
        color='#ffffff',
        fontsize=7.5,
        fontweight='bold',
        ha='center',
        va='center',
        zorder=8,
    )

    # Sensor FoV Cone (Wedge from ego front)
    sensor_x = ego_x + 2.3
    wedge = patches.Wedge(
        (sensor_x, ego_y),
        r=120.0,
        theta1=-25,
        theta2=25,
        color='#38bdf8',
        alpha=0.08,
        zorder=2,
    )
    ax.add_patch(wedge)

    # Draw Detected Target Markers
    for tgt in detected_targets:
        abs_x = ego_x + tgt.range_x
        abs_y = ego_y + tgt.range_y
        if view_min_x <= abs_x <= view_max_x:
            ax.plot(abs_x, abs_y, marker='x', markersize=9, color='#38bdf8', markeredgewidth=2.2, zorder=9)
            ax.text(
                abs_x,
                abs_y + 1.2,
                f"TRACK: {tgt.target_id}\nTTC: {tgt.range_x / max(0.5, abs(tgt.rel_vx)):.1f}s",
                color='#38bdf8',
                fontsize=7,
                fontweight='bold',
                ha='center',
                zorder=9,
            )

    ax.set_xlabel("Longitudinal Highway Coordinate [m]", color='#8b949e', fontsize=9)
    ax.set_yticks([])
    ax.tick_params(axis='x', colors='#8b949e', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#30363d')

    plt.tight_layout()
    return fig


def main():
    st.sidebar.title("🛡️ OverTakeGuard")
    st.sidebar.caption("Rolling-Horizon Uncertainty-Aware ADAS Assistant")

    # Sidebar Scenario Controls
    scenario_names = list(SCENARIO_REGISTRY.keys())
    selected_sc_name = st.sidebar.selectbox("Select Benchmark Scenario:", scenario_names)
    sc_info = SCENARIO_REGISTRY[selected_sc_name]

    st.sidebar.info(sc_info["description"])

    # Auto Pilot & Intent Toggles
    auto_pilot = st.sidebar.checkbox("Autonomous Advisory Compliance (Auto-Abort)", value=True)
    driver_force_overtake = st.sidebar.button("Initiate Overtake Maneuver")
    driver_force_abort = st.sidebar.button("Driver Manual Abort")

    # Simulation playback state in session
    if "current_scenario" not in st.session_state or st.session_state.current_scenario != selected_sc_name:
        st.session_state.current_scenario = selected_sc_name
        st.session_state.sim = sc_info["builder"]()
        st.session_state.sim.auto_pilot = auto_pilot
        st.session_state.step_idx = 0
        st.session_state.history_cache = []

    sim: HighwaySimulation = st.session_state.sim
    sim.auto_pilot = auto_pilot

    if driver_force_overtake:
        sim.trigger_overtake_initiation()
    if driver_force_abort:
        sim.trigger_driver_abort()

    # Pre-run or Step Controls
    col_c1, col_c2, col_c3, col_c4 = st.sidebar.columns(4)
    with col_c1:
        step_btn = st.button("▶️ Step")
    with col_c2:
        run_all_btn = st.button("⏩ Run All")
    with col_c3:
        reset_btn = st.button("🔄 Reset")

    if reset_btn:
        st.session_state.sim = sc_info["builder"]()
        st.session_state.sim.auto_pilot = auto_pilot
        st.session_state.step_idx = 0
        st.session_state.history_cache = []
        st.rerun()

    if step_btn:
        if len(sim.history) < sc_info["steps"]:
            sim.step()
            st.session_state.step_idx = len(sim.history) - 1

    if run_all_btn:
        while len(sim.history) < sc_info["steps"]:
            sim.step()
        st.session_state.step_idx = len(sim.history) - 1

    # Ensure at least 1 step
    if not sim.history:
        sim.step()
        st.session_state.step_idx = 0

    # Current Timeline Scrubber
    total_steps_run = len(sim.history)
    if total_steps_run > 1:
        selected_step = st.slider(
            "Simulation Timeline [Step / Time]",
            min_value=0,
            max_value=total_steps_run - 1,
            value=min(st.session_state.step_idx, total_steps_run - 1),
            step=1,
        )
    else:
        selected_step = 0
        st.caption("Simulation ready at initial step (T = 0.0s). Click '▶️ Step' or '⏩ Run All' to progress.")
    st.session_state.step_idx = selected_step

    cur_snap = sim.history[selected_step]
    cur_snap["scenario_name"] = selected_sc_name
    cur_snap["lead_length"] = sim.lead.length if sim.lead else 4.5
    advisory = cur_snap["advisory"]

    # ------------------ TOP ADVISORY BANNER ------------------
    adv_level = advisory.level.value
    if adv_level == AdvisoryLevel.SAFE_TO_OVERTAKE.value:
        card_class = "advisory-safe"
        badge_icon = "🟢"
    elif adv_level == AdvisoryLevel.WAIT.value:
        card_class = "advisory-wait"
        badge_icon = "🔵"
    elif adv_level == AdvisoryLevel.CAUTION.value:
        card_class = "advisory-caution"
        badge_icon = "🟡"
    else:
        card_class = "advisory-abort"
        badge_icon = "🔴"

    st.markdown(
        f"""
        <div class="advisory-card {card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.8rem; font-weight: 800; letter-spacing: 0.05em;">
                    {badge_icon} {adv_level}
                </span>
                <span style="font-size: 0.95rem; opacity: 0.9; font-weight: 600;">
                    Phase: {cur_snap['phase']} | T = {cur_snap['time']:.1f}s
                </span>
            </div>
            <div style="font-size: 1.1rem; margin-top: 10px; font-weight: 500; line-height: 1.4;">
                {advisory.primary_reason}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------ TELEMETRY KPI METRICS ------------------
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(
            f"""<div class="metric-badge">
                <div class="metric-label">Ego Speed</div>
                <div class="metric-val">{cur_snap['ego_speed_kph']:.1f} <span style="font-size:0.8rem; color:#8b949e;">km/h</span></div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m2:
        t_comp_str = f"{advisory.time_to_complete:.1f} s" if advisory.time_to_complete < 99.0 else "N/A"
        st.markdown(
            f"""<div class="metric-badge">
                <div class="metric-label">Time To Complete</div>
                <div class="metric-val">{t_comp_str}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m3:
        t_avail_str = f"{advisory.available_time_window:.1f} s" if advisory.available_time_window < 900.0 else ">15 s"
        st.markdown(
            f"""<div class="metric-badge">
                <div class="metric-label">Available Window</div>
                <div class="metric-val">{t_avail_str}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m4:
        margin_color = "#7ee787" if advisory.safety_margin_time > 1.5 else ("#f2cc60" if advisory.safety_margin_time > 0 else "#ff7b72")
        margin_str = f"{advisory.safety_margin_time:+.1f} s" if abs(advisory.safety_margin_time) < 800.0 else "Safe"
        st.markdown(
            f"""<div class="metric-badge">
                <div class="metric-label">Safety Time Margin</div>
                <div class="metric-val" style="color: {margin_color};">{margin_str}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m5:
        abort_color = "#7ee787" if advisory.abort_state.is_feasible else "#ff7b72"
        st.markdown(
            f"""<div class="metric-badge">
                <div class="metric-label">Abort Feasibility</div>
                <div class="metric-val" style="color: {abort_color}; font-size:1.1rem;">
                    {'FEASIBLE' if advisory.abort_state.is_feasible else 'PO-NO-RETURN'}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m6:
        st.markdown(
            f"""<div class="metric-badge">
                <div class="metric-label">Sensor Trust</div>
                <div class="metric-val">{int(advisory.sensor_trust_index * 100)}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.write("")

    # ------------------ HIGHWAY VISUALIZER ------------------
    st.subheader("🛣️ Bird's-Eye Highway Canvas & Occlusion Field")
    fig = render_highway_canvas(cur_snap, advisory.occlusion_zones, advisory.detected_hazards)
    st.pyplot(fig)

    # ------------------ TABBED DEEP DIVES ------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔬 Sensor Fusion & Uncertainty",
        "📐 Occlusion & Latent Emergence",
        "🛑 Abortability Dynamics",
        "📊 Benchmark & Baseline Comparison",
    ])

    with tab1:
        st.markdown("### Camera & Radar Fusion Analysis")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Aleatoric Uncertainty (Noise / Contradiction)", f"{advisory.aleatoric_uncertainty:.2f}")
            st.caption("Captures sensor measurement variance, precipitation attenuation, and cross-sensor contradiction.")
        with col_s2:
            st.metric("Epistemic Uncertainty (Occluded Space)", f"{advisory.epistemic_uncertainty:.2f}")
            st.caption("Reflects unobserved road ahead and missing visual coverage behind large vehicles.")
        with col_s3:
            st.metric("Composite ADAS Confidence", f"{advisory.confidence * 100:.1f}%")
            st.caption("Reliability index weighted by sensor concordance and environmental clarity.")

        st.markdown("#### Detected Sensor Targets")
        if advisory.detected_hazards:
            target_data = [
                {
                    "Target ID": t.target_id,
                    "Range X (m)": t.range_x,
                    "Range Y (m)": t.range_y,
                    "Rel Vx (m/s)": t.rel_vx,
                    "Abs Speed (km/h)": abs(t.abs_vx) * 3.6,
                    "Confidence": f"{t.confidence * 100:.0f}%",
                    "Sensor Source": t.sensor_source,
                    "Cross-Sensor Contradiction": "⚠️ CONFLICT" if t.is_conflict else "CONCORDANT",
                }
                for t in advisory.detected_hazards
            ]
            st.dataframe(pd.DataFrame(target_data), width="stretch")
        else:
            st.info("No oncoming vehicles detected within forward sensor line-of-sight.")

    with tab2:
        st.markdown("### Line-of-Sight Occlusion & Latent Hazard Modeling")
        if advisory.occlusion_zones:
            for idx, zone in enumerate(advisory.occlusion_zones):
                st.warning(
                    f"**Occlusion Zone #{idx + 1}**: Cast by `{zone.occluding_vehicle_id}` along Opposing Lane. "
                    f"Spanning X = {zone.start_x:.1f}m to {zone.end_x:.1f}m (Length: {zone.depth:.1f}m). "
                    f"Latent Hazard Risk Score: **{int(zone.latent_hazard_risk * 100)}%**"
                )
        else:
            st.success("No critical visual shadows detected. Line of sight along opposing lane is unobstructed.")

    with tab3:
        st.markdown("### Abortability Analysis & Point of No Return (PoNR)")
        astate = advisory.abort_state
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.metric("Required Deceleration for Abort", f"{astate.required_deceleration:.2f} m/s²")
            st.metric("Time Required to Regain Lane", f"{astate.time_to_abort:.2f} s")
        with col_a2:
            st.metric("Original Lane Slot Clearance", f"{astate.slot_clearance:.1f} m")
            st.metric("Point of No Return Reached?", "YES - DO NOT ABORT" if astate.point_of_no_return else "NO - ABORT SAFE")

        st.info(f"**Abort Engine Rationale**: {astate.reason}")

    with tab4:
        st.markdown("### Comparative Baseline Evaluation")
        st.markdown(
            "Comparing **OverTakeGuard** against a standard **TTC-only Warning Baseline** and a **Naive Sensor Fusion Baseline** across all 6 scenarios."
        )

        suite = ADASBenchmarkSuite()
        benchmark_df, _ = suite.run_full_benchmark()

        # Display Pivot / Comparison Table
        st.dataframe(
            benchmark_df[[
                "scenario", "system", "unsafe_advice_rate_pct", "hazard_warning_recall_pct",
                "false_warning_rate_pct", "warning_lead_time_s", "min_ground_truth_clearance_m", "collision_occurred"
            ]],
            width="stretch"
        )

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            csv_path, json_path = suite.export_benchmark_reports()
            with open(csv_path, "rb") as f:
                st.download_button("📥 Download Benchmark CSV", f, file_name="adas_benchmark_comparison.csv", mime="text/csv")
        with col_d2:
            with open(json_path, "rb") as f:
                st.download_button("📥 Download Benchmark JSON", f, file_name="adas_benchmark_comparison.json", mime="application/json")


if __name__ == "__main__":
    main()
