# 🛰️ Foveated 2.5D LiDAR & Camera Perception System
### Real-Time Semantic Elevation Mapping, 3D World Simulation & Multi-Object Visual Telemetry

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6%2B-yellow.svg)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![Three.js](https://img.shields.io/badge/Three.js-WebGL-black.svg)](https://threejs.org/)

A high-throughput, multi-sensor perception pipeline that combines a **3D WebGL Simulator (Three.js)** with a **Python-based Semantic Perception Backend (YOLO/OpenCV)** in real-time.

Inspired by **human foveated vision**, the system allocates maximum compute and resolution to the immediate driving corridor while progressively downsampling distant space — slashing memory consumption and inference overhead. It includes a rich **WebGL 3D World Simulator** featuring autonomous pure-pursuit driving, 2nd-order suspension physics with dynamic body roll, continuous traffic loops, and a dual-sensor teleoperation dashboard.

---

## 📸 System Visual Gallery & Simulation Showcase

| **1. 3D Virtual World Simulation (Pure-Pursuit Autopilot & 2nd-Order Body Roll)** |
|:---:|
| ![3D Virtual World Simulation](docs/images/sim_3d_world_live.png) |
| *Real-time WebGL 3D simulation: Autonomous vehicle with Newtonian spring-damper suspension physics and dynamic body roll, navigating dark asphalt highway lanes with 3 concentric holographic LiDAR rings (Near 10m, Mid 30m, Far 70m), continuous cruising traffic, and sidewalk pedestrians.* |

| **2. Dual-Sensor Teleoperation Split View** | **3. Multi-Ring LiDAR Bird's-Eye View (BEV)** |
|:---:|:---:|
| ![Dual-Sensor Teleop Split View](docs/images/dual_split_live.png) | ![LiDAR BEV Grid](docs/images/lidar_bev_grid_live.png) |
| *Dual-Split teleoperation: Synchronized 3-ring LiDAR BEV grid (left) alongside live camera foveation with near/mid/far depth perspective bands and sky/hood spatial masking (right).* | *Full 2.5D BEV elevation grid: 3 concentric resolution tiers (Near 5cm, Mid 15cm, Far 50cm), bounding box classification, distance badges, and negative obstacle (pothole) markers.* |

| **4. Camera Foveation with Active ROI & Depth Bands** | **5. Interactive Teleoperation Landing Interface** |
|:---:|:---:|
| ![Camera Foveation](docs/images/camera_foveation_live.png) | ![Landing Hero Interface](docs/images/landing_hero.png) |
| *Spatial horizon gating: Discards static non-road pixels (sky 35% and hood 15%) while applying optical-flow motion gating and depth-scaled detection boxes to dynamic obstacles.* | *Interactive browser HUD interface featuring instant view mode switching, live camera feed selection, LiDAR point cloud controls, and real-time compute savings telemetry.* |

---

## 💡 What Is This & What Does It Do?

Autonomous vehicles and mobile robots need to perceive their immediate environment at 30+ frames per second. Running heavy deep neural networks over raw point clouds causes high latency on edge compute hardware.

This software solves that bottleneck via **Sensor Fusion & Foveation**:
1. **Adaptive 2.5D Ego-Grid**: Uses three concentric resolution rings (Near, Mid, Far) to perfectly map the environment without wasting compute on empty distant space.
2. **Aligns Camera Vision via Dual Foveation**: Eliminates static non-road pixels (sky and vehicle hood), extracting features and tracking objects (cars, pedestrians, cyclists) only where obstacles can realistically appear.
3. **Active Safety System (TTC)**: Real-time calculation of Metric Relative Velocity to warn the driver of impending collisions.
4. **Simulates Full Autonomous Environments**: Built-in Three.js 3D simulation with dynamic vehicle physics, pure-pursuit path tracking, sidewalk pedestrians, and continuous traffic loops.

---

## 🏗️ System Architecture

The architecture consists of a lightweight WebGL Frontend and a Python Vision Backend communicating over high-speed WebSockets.

```
                              [ WebGL Browser Dashboard ]
                      (teleop_dashboard.js / three_simulator.js)
                      ┌───────────────────┴───────────────────┐
                      ▼                                       ▼
          [ 3D Virtual City Engine ]              [ 2.5D Sensor Fusion Engine ]
           Three.js Autonomous Sim                   Multi-Ring LiDAR BEV
                      │                                       │
                      └───────────────────┬───────────────────┘
                                          ▼
                               [ WebSocket Bridge ]
                             JSON Telemetry & Base64
                                          │
                                          ▼
                            [ Python Perception Server ]
                              (yolo_vision_server.py)
                      ┌───────────────────┴───────────────────┐
                      ▼                                       ▼
          [ YOLO Semantic Detector ]              [ Kalman Object Tracker ]
            Ultralytics / YOLOv8n                     Stable Track IDs
```

---

## 🏎️ Autonomous Vehicle Physics & Simulation Engine

The included 3D world simulator runs a realistic vehicle model with continuous physics:

* **Pure Pursuit Lane Following**: Monotonically advancing lookahead waypoint pursuit tracking the center of the right driving lane.
* **2nd-Order Spring-Damper Body Roll**: Realistic chassis lean outward during cornering.
* **Pitch Dive & Squat**: Chassis pitches forward during braking and squats during acceleration.
* **Continuous Traffic Loop**: Cruising sedans and SUVs loop continuously around a 500-meter closed circuit.
* **Dynamic Weather Degradation**: Toggle severe fog that mathematically degrades the simulated YOLO camera confidence while proving LiDAR's robustness.

---

## 🎮 Interactive Controls & Cheatsheet

When running the interactive teleoperation interface, the vehicle and cameras can be controlled via keyboard:

| Control | Key | Action |
|:---|:---:|:---|
| **Throttle / Accelerate** | `W` or `↑` | Accelerate ego vehicle forward |
| **Reverse** | `S` or `↓` | Drive ego vehicle in reverse |
| **Steer Left** | `A` or `←` | Steer wheels left |
| **Steer Right** | `D` or `→` | Steer wheels right |
| **Toggle Autopilot** | `P` | Switch between Autonomous Lane-Following & Manual Control |
| **Toggle Weather** | `R` | Toggle severe fog and sensor degradation simulation |
| **Cycle Cameras** | `C` | Switch camera between **Chase** (3rd), **Cockpit** (1st), and **Overhead** (Top) |

---

## ⚙️ Complete Setup & Installation Guide

### Prerequisites
* **Python**: Python 3.10+ (with `pip` and virtual environment support).
* **Node.js**: v18+ (optional, for the static server).

### Step 1: Clone the Repository
```bash
git clone https://github.com/Kkushak16/Foveated-2.5D-Semantic-Elevation-Mapping.git
cd Foveated-2.5D-Semantic-Elevation-Mapping
```

### Step 2: Python Environment Setup
Install the required machine learning dependencies for the backend.

**Windows (PowerShell):**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Run the Dashboard
Launch the unified server, which starts both the static UI host and the Python WebSocket vision backend.

```bash
python app.py
```

Then open your browser and navigate to:
👉 **`http://localhost:8080`**
