# End-to-End Amazon Sentiment Analysis Pipeline

![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square&logo=python)
![Spark](https://img.shields.io/badge/Apache%20Spark-3.5-orange?style=flat-square&logo=apachespark)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?style=flat-square&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=flat-square&logo=docker)

## Project Overview
This project implements a scalable **End-to-End (E2E) Machine Learning Pipeline** to analyze sentiment in Amazon reviews. 
The architecture is engineered to handle Big Data ingestion (PySpark), reproducible training and versioning (MLflow), and production-ready inference serving (FastAPI/Docker).

---

## Architecture & Workflow

### Phase 1: ETL & Data Engineering (PySpark)
* **Challenge:** The dataset is massive (+5GB hosted on S3) making it unsuitable for Pandas, and suffers from severe class imbalance (86% positive reviews).
* **Solution:**
    * **Big Data Ingestion:** Leveraged **PySpark** to read and process Parquet files at scale.
    * **Rebalancing Strategy:** Implemented a distributed **Undersampling** technique (using Spark's `sampleBy` and `union`) to neutralize bias.
    * **NLP Preprocessing:** Text cleaning and tokenization at scale.
* **Outcome:** Generated a "Golden Sample" of 1 million rows (50/50 balanced class distribution), exported to Parquet for optimized training.

### Phase 2: Training & MLOps (Scikit-learn & MLflow)
* **Modeling:** Built a Scikit-learn pipeline integrating `TfidfVectorizer` (NLP Feature Extraction) and `LogisticRegression`.
* **Experiment Tracking (MLflow):**
    * **Tracking:** Logged multiple experimental runs, capturing hyperparameters and key metrics (Accuracy, F1-Score).
    * **Artifact Management:** Automatically identified the best-performing run and registered the full pipeline as a model artifact.
    * **Model Registry:** Versioned the final model as `amazon_sentiment_model` to strictly decouple the training environment from the production environment.

### Phase 3: Inference API Deployment (FastAPI & Docker)
* **Microservice:** Developed a RESTful API using **FastAPI** exposing a `/predict` endpoint for real-time inference.
* **Dynamic Loading:** On startup, the API queries the MLflow Model Registry to fetch and load the specific `production` version of the model.
* **Containerization:** * Created a **Dockerfile** to package the API and dependencies into an isolated, reproducible image.
    * **Dependency Management:** Strictly pinned library versions (e.g., Scikit-learn, Python 3.9) to prevent **environment skew** between training and inference.

---
