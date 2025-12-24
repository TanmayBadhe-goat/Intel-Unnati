# Project Summary  
## Edge-Ready Smart Digital Twin for Predictive Maintenance of Industrial Motor Drive Systems

---

## Problem Statement

Industrial motor drive systems are critical assets in smart manufacturing environments such as semiconductor fabs, conveyor systems, and pumping units. Unexpected failures in these systems lead to downtime, productivity loss, and maintenance costs. Intel Unnati Problem Statement 5 focuses on developing software-based digital twin and AI solutions for predictive maintenance that are efficient, explainable, and suitable for edge deployment.

---

## Solution Overview

This project implements a **software-only, edge-ready smart digital twin** of an industrial DC motor drive system using:

- A physics-based **Full Order Model (FOM)** capturing electrical, mechanical, and thermal dynamics.
- A **Reduced Order Model (ROM)** derived from the FOM to enable low-latency execution on Intel edge devices.
- **Virtual industrial sensors** with noise, bias, and drift.
- **Fault simulation** for healthy, degraded, and fault operating conditions.
- **AI-based predictive maintenance** using Random Forest and Logistic Regression.

The complete pipeline demonstrates how physics-based modeling and AI can be combined to enable reliable predictive maintenance in Industry 4.0 environments.

---

## Key Results

- The FOM accurately reproduces realistic DC motor behavior, including inrush current, speed ramp-up, and thermal rise.
- The ROM achieves significantly lower computation time while maintaining low RMSE compared to the FOM.
- Random Forest classification reliably distinguishes healthy, degraded, and fault conditions.
- The overall solution is lightweight and suitable for real-time execution on Intel-based edge platforms.

---

## Intel Edge AI & Industrial Relevance

- The ROM is computationally efficient and suitable for embedded CPUs.
- Feature extraction uses simple statistical operations.
- AI models are compact, explainable, and low-latency.
- The approach aligns with Intel Edge AI, predictive maintenance, and smart manufacturing initiatives.

This architecture can be extended to real industrial deployments using Intel edge hardware and optimization toolchains such as OpenVINO.

---

## Limitations & Assumptions

- The solution is **validated using software-based simulations only**; no real hardware or plant data is used.
- Thermal and vibration effects are **modeled using simplified first-order approximations**, not detailed multiphysics models.
- Classical machine learning models are used instead of deep learning to prioritize **explainability and edge efficiency**.
- Fault scenarios are generated through parameter variations and may not cover all real-world failure modes.

---

## Conclusion

This project demonstrates a complete, edge-ready digital twin and predictive maintenance pipeline that combines mechatronics, reduced-order modelling, and AI. The solution is technically strong, computationally efficient, and aligned with Intel Unnati evaluation criteria, making it suitable for academic assessment, technical interviews, and further industrial extension.
