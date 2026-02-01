# Federated Learning–Based Intrusion Detection System (IDS)

## Project Overview
This project explores the use of **Federated Learning (FL)** to build a privacy-preserving **Intrusion Detection System (IDS)**.  
Instead of centralizing sensitive network traffic data, each client trains a local model and only shares model updates with a central server, enabling collaborative learning while preserving data privacy.

The project is divided into **two academic phases**:
- **Capstone 1 (Phase 1):** Proof of Concept, system design, and preliminary evaluation.
- **Capstone 2 (Phase 2):** Full implementation, experimentation, and performance evaluation.

---

## Project Objectives
- Design a distributed IDS using federated learning.
- Preserve data privacy by keeping datasets local to each client.
- Evaluate whether federated learning can achieve competitive performance compared to centralized models.
- Study the impact of non-IID data and client heterogeneity on model performance.
- Provide a scalable and modular research-ready codebase.

---

## Scope of Each Phase

### Phase 1 – Capstone 1 (Proof of Concept)
- Literature review and background study.
- System architecture and component design.
- Dataset understanding and preprocessing strategy.
- Implementation of a **proof-of-concept FL pipeline**.
- Preliminary evaluation using selected clients.
- Report writing and documentation.

> **Note:**  
> Code in this phase is intended for **concept validation only**, not full-scale experimentation.

---

### Phase 2 – Capstone 2 (Full Implementation)
- Finalization of system architecture.
- Full federated learning implementation.
- Client isolation using containers.
- Dataset preparation and distribution per client.
- Federated training and evaluation experiments.
- Baseline comparison with centralized learning.
- Results visualization and analysis.
- Final report and project presentation.

---

## System Architecture (High-Level)

- **Server (Global Model Aggregator)**
  - Receives model updates from clients.
  - Aggregates updates (e.g., FedAvg).
  - Produces and distributes the global model.

- **Clients (Local Training Nodes)**
  - Each client holds a private dataset.
  - Performs local training only.
  - Sends model parameters (not raw data) to the server.

- **Containerized Environment**
  - Each client and server runs in isolated containers.
  - Simulates a real distributed multi-organization setup.

---

## Project Deliverables

### Capstone 1 Deliverables
- Project proposal and problem definition.
- Literature review and related work analysis.
- System architecture and design diagrams.
- Proof-of-concept federated learning code.
- Preliminary evaluation results.
- Capstone 1 report.

### Capstone 2 Deliverables
- Fully implemented federated IDS.
- Containerized client–server setup.
- Experimental evaluation results.
- Performance comparison with centralized models.
- Visualized results and analysis.
- Final project report and presentation.

---

## Repository Structure (Recommended)

project-root/
│
├── server/
│ ├── server.py # Federated aggregation logic
│ ├── model.py # Global model definition
│ └── config.yaml
│
├── clients/
│ ├── client_1/
│ │ └── client.py
│ ├── client_2/
│ │ └── client.py
│ └── client_n/
│
├── data/
│ ├── client_1/
│ ├── client_2/
│ └── client_n/
│
├── utils/
│ ├── data_utils.py # Train/validation/test split logic
│ ├── metrics.py
│ └── logging.py
│
├── experiments/
│ ├── results/
│ └── plots/
│
├── containers/
│ ├── Dockerfile.client
│ ├── Dockerfile.server
│ └── docker-compose.yml
│
├── docs/
│ ├── architecture.md
│ └── gantt_chart.drawio
│
├── README.md
└── requirements.txt


---

## Dataset Handling Strategy
- Each client uses **its own local dataset**.
- Data is split into:
  - **Train set:** used for local training.
  - **Validation set:** used by the client for self-evaluation.
  - **Test set:** used to evaluate the global model per client.
- No data sharing occurs between clients.
- Randomized splits with fixed seeds ensure reproducibility.

---

## Model Evaluation
- Metrics used:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Global model evaluated on:
  - Each client’s local test set.
- Comparison with centralized baseline (Phase 2).

---

## Technologies Used
- Python
- PyTorch / TensorFlow (depending on implementation)
- Federated Learning concepts (FedAvg)
- Docker & Docker Compose
- NumPy, Pandas, Scikit-learn
- Matplotlib / Seaborn

---

## How to Run (High-Level)

### 1. Install dependencies
```bash
pip install -r requirements.txt
