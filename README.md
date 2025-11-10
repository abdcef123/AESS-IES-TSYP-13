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



 References

· A Robust Indoor Positioning System Based on the Procrustes Analysis and Weighted Extreme Learning Machine
    IEEE International Conference on Wireless Information Technology and Systems
    Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5873878


· arXiv preprint
    Link: https://arxiv.org/pdf/2407.04730
