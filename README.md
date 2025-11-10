# Arquitectura del Pipeline de Sentimiento

Este proyecto implementa un pipeline E2E (End-to-End) para analizar el sentimiento de reviews de Amazon. La arquitectura está diseñada para manejar la ingesta a escala de Big Data (PySpark), el entrenamiento y versionado reproducible (MLflow), y el despliegue como una API de inferencia (FastAPI/Docker).

---

## Fase 1: ETL y Forjado de Datos (PySpark)

**Problema:** El dataset (+5GB en S3) es demasiado grande para Pandas y presenta un desbalanceo de clases severo (86% positivos).

**Solución:**

* Lectura de datos Parquet en un DataFrame PySpark.
* Implementación de una estrategia de **Undersampling** a escala (utilizando `sampleBy` y `union` de Spark) para crear un dataset balanceado.
* Procesamiento y limpieza de texto.

**Resultado:** Creación de un "Golden Sample" de 1 millón de filas (50/50) exportado a Parquet, listo para entrenamiento local.

---

## Fase 2: Entrenamiento y Registro (Scikit-learn & MLflow)

**Modelado:** Se implementó un pipeline de Scikit-learn que consiste en `TfidfVectorizer` (para NLP) seguido de una `LogisticRegression`.

**MLOps:** Se utilizó **MLflow** para:

* **Rastrear** múltiples experimentos (runs), registrando hiperparámetros y métricas (ej. accuracy, f1-score).
* **Identificar** el mejor *run* y registrar el pipeline completo de Sklearn como un artefacto (model artifact).
* **Versionar** el modelo final en el **MLflow Model Registry** bajo el nombre `amazon_sentiment_model` para desacoplar el entrenamiento del despliegue.

---

## Fase 3: Despliegue de API (FastAPI & Docker)

**Servicio:** Una API RESTful construida con **FastAPI** que expone un endpoint `/predict`.

**Carga del Modelo:** Al iniciar, la API consulta el Model Registry de MLflow y carga la versión `production` del modelo `amazon_sentiment_model`.

**Contenerización:** Un **Dockerfile** que empaqueta la API de FastAPI y sus dependencias (incluyendo mlflow y scikit-learn) en una imagen aislada.

> **Nota:** Se gestionó la compatibilidad de versiones de librerías (ej. scikit-learn 1.X.X, Python 3.9) entre el entorno de entrenamiento y el contenedor de producción para evitar *mismatch* de artefactos.
