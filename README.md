# Federated Learning–Based Intrusion Detection System (IDS)

## Project Overview
This project presents a **Federated Learning (FL)–based Intrusion Detection System (IDS)** designed to detect malicious network activities while preserving data privacy. Instead of centralizing sensitive data, each client trains a local model on its own dataset, and only model updates are shared with a central server to build a global model.

The project is developed as part of a **Capstone Project**, divided into two phases:
- **Capstone 1**: Design, literature review, and proof of concept
- **Capstone 2**: Full implementation, experimentation, and evaluation

---

## Motivation
Traditional centralized intrusion detection systems require aggregating large volumes of sensitive network data, which raises privacy, legal, and security concerns. Federated learning addresses these issues by enabling collaborative model training without sharing raw data, making it suitable for distributed and privacy‑sensitive environments such as network security.

---

## Key Features
- Federated learning using a **client–server architecture**
- Local training on **distributed and isolated datasets**
- **No raw data sharing** between clients and server
- Support for **class imbalance handling**:
  - Undersampling
  - Oversampling
  - Weighted loss
- Separation between **training clients** and **test‑only clients**
- Evaluation using standard metrics:
  - Accuracy
  - Precision
  - Recall
  - F1‑score
- Command‑line configuration for experiments

---

## Project Structure
.
├── Client/
│ ├── client.py # Client-side training logic
│ ├── model.py # Logistic model and FedAvg implementation
│ └── init.py
│
├── Server/
│ ├── server.py # Federated server logic
│ └── init.py
│
├── data/
│ ├── client_datasets/ # Datasets for each client
│ └── data_utils.py # Data loading, splitting, and preprocessing
│
├── results/
│ ├── client_metrics/ # Per-client evaluation results
│ └── global_metrics/ # Global model evaluation results
│
├── run_federated.py # Main script to run federated training
├── requirements.txt # Python dependencies
└── README.md # Project documentation




---

## System Architecture
- **Clients**:  
  Each client holds a local dataset and trains a local model independently.
- **Server**:  
  The server aggregates client model updates using **Federated Averaging (FedAvg)** to produce a global model.
- **Isolation**:  
  In Capstone 2, clients and server are deployed on separate devices using containerization to simulate a real distributed environment.

---

## Data Handling
- Each client uses its **own local dataset**
- Data is **not shared** across clients
- Random **train/test split per client**
- **Test data is never rebalanced**
- Class imbalance handling is applied **only to training data**

---

## Running the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
