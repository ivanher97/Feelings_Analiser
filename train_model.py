import pandas as pd
import mlflow
import mlflow.sklearn # Importante para 'log_model'

# Imports de Scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# CONFIGURACIÓN DE MLFLOW 
# Apunta a tu directorio local 'mlruns'
mlflow.set_tracking_uri("file:./mlruns")
# Define un nombre para el experimento. Si no existe, se creará.
mlflow.set_experiment("Amazon Sentiment Sklearn")

# CARGA DE DATOS
print("Cargando datos desde Parquet (Pandas)...")
try:
    # Usamos Pandas, no Spark. Esto carga el 1M de filas en RAM.
    df_train = pd.read_parquet("./data/sklearn_sample_1M.parquet")
except FileNotFoundError:
    print("Error: No se encontró el archivo './data/sklearn_sample_1M.parquet'")
    print("Asegúrate de haber ejecutado el script de Spark (ML_Analisis.py) primero.")
    exit()

# DEFINICIÓN DE FEATURES (X) y TARGET (y) 
X = df_train['review_body']
y = df_train['label']
# Dividimos para tener un set de validación
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# CONSTRUCCIÓN DEL PIPELINE DE SKLEARN
# Esto es conceptualmente idéntico a tu pipeline de Spark ML,
# pero ahora es un pipeline 100% Scikit-learn.
print("Construyendo el pipeline de Scikit-learn...")
# MLflow guardará ESTE pipeline como un único artefacto.
# Incluye el pre-procesamiento (Vectorizers) Y el modelo (Classifier).
sentiment_pipeline = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1, 2), max_features=20000)), # Usamos n-gramas y limitamos features
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(solver='liblinear', random_state=42)) # Un solver rápido para este tamaño
])

# 5. INICIO DEL RUN DE MLFLOW
print("Iniciando 'run' de MLflow...")

with mlflow.start_run(run_name="LogisticRegression_Baseline") as run:
    print("Entrenando el modelo...")
    # Entrenamos el pipeline completo
    sentiment_pipeline.fit(X_train, y_train)
    print("Evaluando el modelo...")
    # Evaluamos en el set de test
    y_pred = sentiment_pipeline.predict(X_test)
    y_pred_proba = sentiment_pipeline.predict_proba(X_test)[:, 1] # Probabilidad de la clase '1'
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    #  REGISTRO EN MLFLOW
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    # Log de Parámetros (del pipeline)
    mlflow.log_params(sentiment_pipeline.get_params(deep=False))
    # Puedes loggear parámetros específicos si quieres
    mlflow.log_param("model_solver", "liblinear")
    mlflow.log_param("vectorizer_max_features", 20000)
    # Log de Métricas
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("auc", auc)

    # LA PARTE MÁS IMPORTANTE) GUARDADO DEL MODELO 
    print("Guardando y registrando el modelo en MLflow...")
    # Esta es la línea clave.
    # 1. 'sentiment_pipeline': El objeto del modelo/pipeline a guardar.
    # 2. 'amazon_sentiment_model': El nombre de la "carpeta" del artefacto.
    # 3. 'registered_model_name': El nombre que usaremos en la API para
    #    llamar al modelo (ej: "models:/amazon_sentiment_model/Production").
    mlflow.sklearn.log_model(
        sk_model=sentiment_pipeline,
        name="amazon_sentiment_model", # Nombre de la carpeta dentro del 'run'
        registered_model_name="amazon_sentiment_model" # Nombre para el Registro de Modelos
    )

    print("\n--- ¡Entrenamiento completado y modelo registrado! ---")
    print(f"ID del Run: {run.info.run_id}")
    print("Puedes ver el 'run' en la UI de MLflow ejecutando: mlflow ui")