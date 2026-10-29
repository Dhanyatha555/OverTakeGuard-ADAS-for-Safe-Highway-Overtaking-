# OverTakeGuard: A Rolling-Horizon, Uncertainty-Aware Overtaking Feasibility and Abortability Assistant

**Project Title:** OverTakeGuard: Real-Time ADAS for Safe Highway Overtaking Under Restricted Visibility  
**Problem Statement:** Problem Statement 3: Real-Time ADAS for Safe Highway Overtaking Under Restricted Visibility  
**Implementation:** Python 3.12+, Streamlit, NumPy, Pandas, Matplotlib, Pytest  

---

## 📌 Executive Summary
Overtaking on two-lane bidirectional highways is one of the highest-risk driving maneuvers, responsible for a disproportionate share of severe and fatal head-on collisions. When the driver's forward field of view is occluded by large commercial vehicles (semi-trailer trucks, buses) or compromised by adverse weather (heavy rain, fog) and night glare, human judgment becomes vulnerable to cognitive blind spots and optimistic misestimations.

**OverTakeGuard** is a simulation-based ADAS proof-of-concept that continuously evaluates overtaking feasibility and abortability. Rather than naively assuming that absence of detected oncoming traffic implies clear road, OverTakeGuard models **line-of-sight occlusion shadows**, quantifies **epistemic and aleatoric uncertainties**, evaluates **dynamic abort deceleration profiles**, and enforces the safety rule: *Never recommend an abort maneuver without verifying whether it is physically feasible (Point of No Return analysis)*.

---

## 🏗️ System Architecture

The OverTakeGuard software architecture is structured into modular, decoupled subsystems:

```
ADAS PS3/
├── overtakeguard/
│   ├── __init__.py           # Package definition
│   ├── models.py             # Data classes (VehicleState, SensorObservation, OcclusionZone, AdvisoryOutput)
│   ├── simulation.py         # 2D Highway kinematic simulator with multi-vehicle dynamics
│   ├── sensors.py            # Simulated Camera & Radar sensors with noise, FoV, and weather attenuation
│   ├── occlusion.py          # Line-of-sight ray tracing and latent hazard emergence engine
│   ├── fusion.py             # Inverse-variance Bayesian sensor fusion with conflict detection
│   ├── abortability.py       # Deceleration constraints, slot availability, and Point of No Return (PoNR)
│   ├── decision_engine.py    # Rolling-horizon feasibility engine and explainable advisory generator
│   ├── baselines.py          # Comparative baselines: TTC-only warning & Naive Sensor Fusion
│   ├── scenarios.py          # 6 curated reproducible benchmark scenarios
│   └── evaluation.py         # ADAS metrics computation (unsafe advice rate, recall, lead time, margins)
├── tests/
│   ├── test_occlusion.py     # Tests for shadow polygons and emergence time
│   ├── test_sensor_fusion.py # Tests for covariance weighting and sensor contradiction
│   ├── test_abortability.py   # Tests for deceleration limits and Point of No Return
│   ├── test_decision_engine.py# Tests for multi-level explainable advisories
│   └── test_scenarios.py     # End-to-end regression tests across all 6 benchmark cases
├── reports/
│   ├── adas_benchmark_comparison.csv   # Exported benchmark CSV
│   └── adas_benchmark_comparison.json  # Exported benchmark JSON
├── app.py                    # Streamlit interactive dashboard with 2D visualizer
├── requirements.txt          # Python dependencies
└── README.md                 # Complete system documentation
```

---

## 🔬 Core Methodologies

### 1. Occlusion Shadow & Latent Hazard Modeling (`occlusion.py`)
- When trailing a large vehicle (such as a 16-meter box truck), the truck's bulk creates an optical blind cone in the opposing lane.
- Using ray-tracing from the ego vehicle's sensor origin through the truck's bounding corners, OverTakeGuard determines the occluded road interval $[x_{occ\_start}, x_{occ\_end}]$ along the oncoming lane.
- **Worst-Case Latent Hazard Hypothesis:** The system assumes a hypothetical oncoming vehicle traveling at maximum highway speed ($v_{oncoming}^{max} = 100\text{ km/h}$) could be concealed inside the shadow.
- The emergence time $T_{emergence} = \frac{x_{occ\_start} - x_{ego}}{v_{ego} + v_{oncoming}^{max}}$ is compared against the required overtaking completion time. If $T_{emergence} < T_{complete} + t_{buffer}$, the road is declared blind and the system prompts `WAIT`, even if no physical target has been detected yet!

### 2. Sensor Fusion & Uncertainty Representation (`sensors.py`, `fusion.py`)
- **Camera Sensor:** Provides high classification fidelity and sharp lateral resolution, but degrades sharply in heavy rain and night glare ($\sigma_x \propto r^{1.5}$, weather noise multiplier up to $2.5\times$).
- **Radar Sensor:** Provides precise Doppler range-rate and longitudinal distance, robust in night/fog, but experiences attenuation and clutter in heavy precipitation.
- **Inverse-Variance Weighting:** Fuses matched target estimates using covariance intersection:
  $$\mu_{fused} = \frac{\frac{\mu_{cam}}{\sigma_{cam}^2} + \frac{\mu_{rad}}{\sigma_{rad}^2}}{\frac{1}{\sigma_{cam}^2} + \frac{1}{\sigma_{rad}^2}}$$
- **Conflict Handling (Fail-Safe Assumption):** When camera and radar disagree (e.g. radar detects a target during optical glare while camera claims clear road), the system flags `is_conflict = True`, elevates aleatoric uncertainty, adopts the conservative hazard estimate, and issues `WAIT` or `CAUTION`.

### 3. Abortability Analysis & Point of No Return (`abortability.py`)
- A critical limitation of legacy ADAS is blindly advising the driver to "abort" mid-maneuver even when the vehicle has already passed the lead truck!
- OverTakeGuard computes the required braking deceleration $a_{req}$ to slip back behind the lead vehicle's rear bumper into the original lane:
  $$a_{req} = \frac{2 \cdot \Delta x_{drop\_needed}}{(T_{abort} - t_{reaction})^2}$$
- **Point of No Return (PoNR):** If $a_{req} > a_{emergency\_max}$ ($6.0\text{ m/s}^2$) or if ego is ahead of the lead vehicle and forward completion requires less time than aborting ($T_{complete} < 0.7 \cdot T_{abort}$), aborting is declared **INFEASIBLE**.
- Under this condition, the system *never recommends abort*; instead, it signals `CRITICAL CAUTION` and instructs forward completion or shoulder evasion.

### 4. Explainable Multi-Level Advisories (`decision_engine.py`)
The system outputs 4 distinct, color-coded actionable advisories with plain-language physical rationales:
- 🟢 **`SAFE TO OVERTAKE`**: Clear road confirmed ahead, positive safety time margin ($> 2.5$s), low uncertainty, open abort slot.
- 🔵 **`WAIT`**: Approaching oncoming vehicle detected, latent hazard emergence risk inside truck blind spot, or opposing lane occupied alongside ego.
- 🟡 **`CAUTION`**: Tight safety margin, elevated sensor uncertainty (rain/glare), or Point of No Return reached while approaching hazard.
- 🔴 **`ABORT NOW`**: Maneuver underway in opposing lane, critical oncoming hazard detected, and abort into original lane is physically feasible.

---

## 📊 Benchmark Scenarios & Evaluation Results

OverTakeGuard was benchmarked across 6 challenging highway scenarios against two baselines:
1. **TTC-Only Baseline:** Standard collision warning threshold based on detected target TTC ($\le 6.0$s). Blind to occlusions and abort feasibility.
2. **Naive Sensor Fusion Baseline:** Static distance threshold ($\le 200$m) using target union. Ignores occlusion shadows and vehicle dynamics.

### Quantitative Comparison Table

| Scenario Name | Metric | OverTakeGuard (Proposed) | TTC-Only Baseline | Naive Fusion Baseline |
| :--- | :--- | :---: | :---: | :---: |
| **Scenario 1: Clear Road & Safe Overtake** | Unsafe Advice Rate | **0.0%** | 0.0% | 0.0% |
| | Hazard Warning Recall | **100.0%** | 100.0% | 100.0% |
| | Min Clearance | **20.2 m** | 20.2 m | 20.2 m |
| **Scenario 2: Approaching Oncoming Vehicle** | Unsafe Advice Rate | **0.0%** | 0.0% | **5.0%** |
| | Hazard Warning Recall | **100.0%** | 100.0% | **95.0%** |
| | Min Clearance | **21.4 m** | 21.4 m | 21.4 m |
| **Scenario 3: Hidden Oncoming Car Behind Truck** | **Unsafe Advice Rate** | **0.0%** | **58.97%** *(Failure)* | **56.41%** *(Failure)* |
| | **Hazard Warning Recall** | **100.0%** | **41.03%** | **43.59%** |
| | Safety Diagnosis | **Occlusion detected, WAIT issued** | Blind to truck shadow | Blind to truck shadow |
| **Scenario 4: Emerging Hazard During Overtake** | Unsafe Advice Rate | **0.0%** | 16.67% | 10.0% |
| | Abort Triggered & Succeeded | **YES (100%)** | YES | YES |
| | Collision Avoidance | **No Collision** | No Collision | No Collision |
| **Scenario 5: Radar Degradation in Heavy Rain** | Unsafe Advice Rate | **0.0%** | **50.0%** *(Missed det.)* | **42.5%** |
| | Hazard Warning Recall | **100.0%** | 50.0% | 57.5% |
| **Scenario 6: Conflicting Camera & Radar (Glare)** | Unsafe Advice Rate | **0.0%** | **13.16%** | **13.16%** |
| | Hazard Warning Recall | **100.0%** | 86.84% | 86.84% |

### Key Findings
- **Zero Unsafe Advice:** Across all 6 reproducible benchmark scenarios, OverTakeGuard achieved **0.0% unsafe advice rate**, compared to **59.0%** for TTC-only and **56.4%** for Naive Fusion in the truck occlusion scenario.
- **Occlusion Protection:** In Scenario 3, legacy baselines gave `SAFE TO OVERTAKE` right into an impending head-on crash because the oncoming car was obscured by the lead truck. OverTakeGuard's latent hazard emergence engine successfully prevented initiation.
- **Robustness Under Degradation:** In heavy rain and night glare, OverTakeGuard elevates aleatoric uncertainty and penalizes sensor trust, avoiding false clears and maintaining a 100% hazard recall.

---

## 🚀 Installation & How to Run

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12.8)
- Windows / Linux / macOS

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m pytest tests/ -v
```
All 21 automated unit and regression tests will execute and pass.

### 4. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501` to interact with the real-time simulation, step through scenarios, inspect sensor fusion cones, and download benchmark data.

### 5. Generate Benchmark Reports (CLI)
```bash
python -c "from overtakeguard.evaluation import ADASBenchmarkSuite; suite = ADASBenchmarkSuite(); csv_p, json_p = suite.export_benchmark_reports(); print('Exported:', csv_p, json_p)"
```

---

## ⚠️ Assumptions & Limitations
1. **Simulation Scope:** This project is a proof-of-concept software simulation evaluated on kinematic vehicle models. It does **not** claim real-vehicle hardware-in-the-loop validation or ISO 26262 functional safety certification.
2. **Road Topology:** Road geometry is currently modeled as a 2-lane planar highway with configurable sight horizons. Advanced 3D vertical crests and continuous non-planar curvature can be incorporated in future iterations.
3. **Driver Agency:** OverTakeGuard is an *advisory assistant* designed to support and inform driver decision-making, not an autonomous replacement for human vigilance.
