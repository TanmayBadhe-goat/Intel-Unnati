# Edge-Ready Smart Digital Twin for Predictive Maintenance of Industrial Motor Drive Systems  
## using Reduced-Order Modelling and AI

**Intel Unnati – Problem Statement 5**  
**Software-Only Digital Twin Implementation**

---

## Project Overview

In this project, I focused on keeping the model simple and explainable so that it can realistically run on Intel edge devices.

This project implements an **edge-ready smart digital twin** of an industrial DC motor drive system using a hybrid **physics-based and AI-driven** approach.

The solution includes:

- A physics-based **Full Order Model (FOM)** of the electro-mechanical motor system  
- A control-oriented **Reduced Order Model (ROM)** suitable for real-time edge execution  
- **Virtual sensors** with realistic noise, bias, and drift  
- Multiple **fault scenarios** (healthy / degraded / fault) for predictive maintenance  
- **AI-based health classification** using Random Forest and Logistic Regression  
- A complete **digital twin pipeline** aligned with Intel Edge AI and Industry 4.0 concepts  

The implementation is fully software-based and developed in **Python** using a **Jupyter-style notebook script**.

This project directly addresses **Intel Unnati PS-5** requirements:

- Application of **mechatronics principles** (electrical, mechanical, thermal dynamics)  
- Clear **Reduced Order Modelling** from FOM → ROM  
- Use of **AI/ML techniques** for predictive maintenance  
- Simulation-based validation and result analysis  
- Professional documentation suitable for evaluation and interviews  

A visual representation of the digital twin workflow is provided in  
**`digital_twin_architecture.png`**.

---

### Why not LSTM or CNN?

Classical machine learning models were intentionally chosen due to their
**explainability**, **lower computational cost**, and **suitability for real-time
Intel edge deployment**.

A one-page executive summary is provided in **`PROJECT_SUMMARY.md`**.

---

## Repository Structure

- `edge_smart_digital_twin_dc_motor.py`  
  Main notebook-style script containing:
  - Full Order Model (FOM) and Reduced Order Model (ROM)
  - Motor simulations under healthy and faulty conditions
  - Sensor modelling and fault scenario generation
  - Dataset creation and feature extraction
  - AI models (Random Forest, Logistic Regression, optional Isolation Forest)
  - Digital twin architecture and maintenance decision logic

- `Technical_Report_Edge_Ready_Smart_Digital_Twin_DC_Motor.md`  
  Evaluation-ready technical report aligned with Intel Unnati PS-5 structure.

- `PROJECT_SUMMARY.md`  
  One-page overview including results, relevance, and limitations.

- `requirements.txt`  
  Python dependencies required to run the project.

---

## Running the Project

### 1. Create and activate a virtual environment (recommended)

python -m venv venv
venv\Scripts\activate   # On Windows

### 2. Install dependencies

pip install -r requirements.txt
pip install jupyter

#### Option A – Run as a script

python edge_smart_digital_twin_dc_motor.py

#### Option B – Run using Jupyter / VS Code (Recommended)

-Open the project folder in VS Code

-Open edge_smart_digital_twin_dc_motor.py

-Use Run Cell or Run All on the # %% notebook cells

## Expected Outputs

### Plots
-FOM motor response (voltage, current, speed, temperature)
-FOM vs ROM comparison plots
-Simulated sensor signals under fault conditions
-Confusion matrices for Random Forest and Logistic Regression
-Random Forest feature importance
-Health-state prediction timeline

### Printed Metrics
-FOM vs ROM simulation time and speedup factor
-RMSE between FOM and ROM (current and speed)
-Classification reports (precision, recall, F1-score)
-Example maintenance decisions based on predicted probabilities

### Intel Edge AI Alignment
-Reduced Order Model enables low-latency edge execution
-Feature extraction uses simple statistical operations
-AI models are compact and explainable
-The pipeline can be optimized for Intel hardware using Edge AI toolchains
such as OpenVINO in real deployments
