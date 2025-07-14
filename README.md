# 🧠 SOM + KMeans Clustering API

This project implements a **Self-Organizing Map (SOM)** combined with **KMeans clustering** to group customer segments from the [Mall Customer Segmentation dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python?resource=download). It is built with:

- ⚙️ FastAPI for serving predictions
- 🧪 A modular MLOps pipeline (data ingestion → training → evaluation)
- 🐳 Docker + 🛰️ GitHub Actions CI/CD
- ☁️ Deployed on Google Cloud Run

---

## 🛠️ Top 6 Code Improvements Applied

During the refactor and modularization, the following enhancements were made to improve code quality, reusability, and maintainability:

1. **Encapsulation into a Class-Based Design**  
   Transformed the SOM training logic from a monolithic function into an object-oriented class `SOM`, enabling reusability and state tracking.

2. **Precomputation of Neuron Grid**  
   Grid coordinates for the SOM topology were precomputed in the class constructor, improving performance and reducing redundant calculations in each iteration.

3. **Parameter Flexibility via Constructor**  
   Allowed external configuration of SOM hyperparameters (e.g., sigma, learning rate, number of iterations), supporting experimentation and tuning.

4. **Logging for Training Lifecycle**  
   Integrated logging at the start and end of the SOM training process for improved observability in production and debugging environments.

5. **Separation of Concerns (Modularity)**  
   Broke the overall system into well-organized modules: SOM core, KMeans clustering, feature pipeline, evaluation, and API routing — adhering to clean architecture principles.

5. **CI/CD Deployment for Scalability and Automation**  
   Implemented a GitHub Actions workflow that automates Docker builds and deploys the FastAPI service to Google Cloud Run. This allows for scalable, serverless deployments with automatic updates on code changes — ensuring fast, reliable, and reproducible model serving.



## 📁 Project Structure

```text
som_project/
│
├── src/
│   ├── som/                      # Core SOM logic
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── utils.py
│   │   └── config.py
│   │
│   ├── kmeans/                   # Modular KMeans logic
│   │   ├── __init__.py
│   │   ├── cluster.py            # Cluster SOM weights
│   │   └── evaluator.py          # Silhouette score, elbow, etc.
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── generate_features.py
│   │   ├── loader.py
│   │   └── feature_registry.json
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── feature_engineering.py
│   │   ├── model_trainer.py      # ✅ Includes SOM + KMeans
│   │   ├── model_evaluator.py
│   │   └── pipeline_runner.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   
│   │
│   └── visualizer/
│       ├── __init__.py
│       ├── plots.py
│       
│
├── configs/
│   └── train_config.yaml         # Configs for SOM + KMeans
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── schema.json
│
├── artifacts/
│   ├── models/
│   │   ├── som_model.pkl
│   │   ├── minmax_scaler.pkl
│   │   └── kmeans_model.pkl
│   ├── weights/
│   │   ├── som_weights.npy
│   ├── metrics/
│   │   ├── metrics.json
│   └── plots/
│       ├── som_cluster.png
│       ├── som_grid.png
│
├── notebooks/
├── scripts/
├── .github/
│   └── workflows/
│       └── deploy.yml
├── Dockerfile
├── main.py                        # Entry point for fast api
├── run_train_pipeline.py          # Entry point for training 
├── pyproject.toml
├── uv.lock
└── README.md
```

## 🧱 System Architecture Diagram
![alt text](notebooks/assets/SA-Diagram.png)

## ☁️ Cloud Run API (Prediction Endpoint)

FastAPI app is deployed to Google Cloud Run and can be accessed using a public HTTPS URL.

### 🔗 Base URL

```bash
https://som-api-178855485821.australia-southeast1.run.app
```

## 📬 End Point — `/api/predict` — Predict Cluster for Input Data

### 🔧 Method - POST

### 📤 Request Body (JSON)

```json
{
    "age": 20,
    "income":16,
    "spending_score":6
}
```

### ✅ Example Response

```json
{
    "cluster": [
        2
    ]
}
```

## 🚀 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/OsamaDS/SOM-Network.git
```

### 2. Install dependencies

```bash
uv pip install --system .
```

### 3. Start FastAPI server

```bash
uvicorn main:app --reload --port 9000
```

## 🚀 How to Run using Docker

### 1. Pull the image

```bash
docker pull osamads/som-api
```

### 2. Run the container

```bash
docker run -d -p 9000:9000 osamads/som-api
```

