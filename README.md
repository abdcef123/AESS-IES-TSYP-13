# AESS-IES-TSYP-13: Autonomous FDIR System via FPGA-Accelerated XGBoost

This project presents an embedded **Fault Detection, Isolation, and Recovery (FDIR)** system for a 3U CubeSat. Our solution ensures **electronic speed** and **space resilience** by accelerating an Artificial Intelligence model (XGBoost) directly onto the reconfigurable logic of a space-grade FPGA (Rad-Tolerant). This design eliminates the need for immediate intervention from the On-Board Computer (OBC) or ground control, guaranteeing critical autonomy.

### Key Objectives
 * **Speed:** Total diagnostic latency of **5–10 µs** achieved through parallel RTL processing.
 * **Reliability:** Implementation of **Triple Modular Redundancy (TMR)** for radiation resilience.
 * **Autonomy:** Deterministic and immediate recovery from critical faults (ADCS/EPS).


---

## Architecture, Technologies, and Workflow

The architecture is based on a high-speed, closed-loop control system (see diagrams in `5_DOCUMENTATION/`). The data flow is direct: **ADCS/EPS Sensors** $\to$ **ADC** $\to$ **FPGA** $\to$ **Actuators/Drivers**.

The diagnostic core is the **XGBoost (eXtreme Gradient Boosting)** algorithm, chosen for its superior accuracy in **Fault Detection and Isolation** on tabular sensor data. The model is executed on the **FPGA** which, beyond providing **electronic speed**, implements **TMR** at the RTL level to ensure diagnostic reliability. The proof-of-concept relies on **MATLAB/Simulink Simulation**, used to model orbital dynamics and generate realistic training datasets, ensuring the algorithm's relevance.

![System Overview Diagram](Overview_diagram.jpeg)
---

##  Repository Structure

The structure reflects the system's engineering workflow:

 * **`1_DATA_SIMULATION/`** contains MATLAB/Simulink scripts and raw training data.
 * **`2_MODEL_DEVELOPMENT/`** houses the initial Python code for XGBoost and the model quantization scripts (4-8 bits).
 * **`3_HARDWARE_RTL/`** holds the HLS C/C++ code and the final Verilog/VHDL files (including TMR logic).
 * **`4_VERIFICATION/`** contains the Testbench files for RTL verification and simulation logs.
 * **`5_DOCUMENTATION/`** groups the final report, synoptic diagram, and flow chart.

---

##  Key Results and Proof of Success

The success of the design is confirmed by achieving critical performance indicators:

| Metric | Result Achieved | Proof of Success |
| :--- | :--- | :--- |
| **Inference Latency** | **5−10 µs** | Strict adherence to the **Electronic Speed** criterion for ADCS. |
| **Isolation Accuracy** | **> 96.6%** | High diagnostic reliability enabling autonomous recovery. |
| **Resilience** | **Active TMR** | Design validated for the space environment (SEU Mitigation). |

The XGBoost-on-FPGA FDIR system delivers a **precise, rapid, and radiation-resilient** solution for the critical operations of a 3U CubeSat.



### References

· A Robust Indoor Positioning System Based on the Procrustes Analysis and Weighted Extreme Learning Machine
    IEEE International Conference on Wireless Information Technology and Systems
    Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5873878


· arXiv preprint
    Link: https://arxiv.org/pdf/2407.04730

    · Use of Triple Modular Redundancy (TMR) technology
    Google Scholar
    Link: https://share.google/6FnwssehwG4plA4TN
· A Review of Anomaly Detection in Spacecraft
    Google Scholar
    Link: https://share.google/vFmIXTs4Wr9U3WG9I
· Anomaly Detection Using Deep Learning
    Google Scholar
    Link: https://share.google/n2GDDwi76o6G2yRyf
· Performance Evaluation of Machine Learning Methods for Anomaly Detection in CubeSat Solar Panels
    Link: https://share.google/Y9ijk3H75KiGc6eww
· Use of FPGA in Real-Time Control Systems for Aerospace
    Google Scholar
    Link: https://share.google/CekPDDCho796YxmzE
    · IEEE Aerospace Conference Papers
    IEEE Xplore
    Link: https://ieeexplore.ieee.org/
· CubeSat Reliability and Testing Standards
    CubeSat Program
    Link: https://www.cubesat.org/

