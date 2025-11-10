# 1. IMPORTS
# Importa las librerías necesarias
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import warnings

print("MLflow tracking URI configurado para apuntar al host.")

# Ignorar advertencias futuras de sklearn, no es crítico pero limpia la salida
warnings.filterwarnings("ignore")

# Una vez iniciado MlFlow, podemos almacenar la direccion del modelo
MODEL_URI = "./mlruns/615840976000367341/models/m-11c472aee18e495a858c212f392f105d/artifacts"
# MODEL_URI = "./mlruns/0/<RUN_ID>/artifacts/amazon_sentiment_model" # Alternativa si buscas en disco

# DEFINICIÓN DEL SCHEMA (El "Contrato" Pydantic)
# Define la estructura de datos que esperas en la petición ENTRANTE.
# Esto crea tu "contrato".
# Si quisieras más campos, los añadirías aquí (ej: user_id: int)
class ReviewInput(BaseModel):
    review_text: str
    

# INSTANCIACIÓN Y CARGA DEL MODELO
# Debe estar aquí, en el scope global, para que no se recargue con cada petición.
# Esto se ejecuta UNA SOLA VEZ, cuando Uvicorn (el servidor) arranca.
# Tenemos que iniciar desde terminal mlflow ui, para acceder al puerto 5000
# Iniciar servidor uvicorn script(sin.py):nombreFastApi(app) -> uvicorn main:app --reload
print("Iniciando la aplicación y cargando el modelo...")
modelo_cargado = mlflow.pyfunc.load_model(MODEL_URI)

app = FastAPI(
    title="API de Sentimiento de Reviews",
    description="Un servicio para predecir el sentimiento de reviews de Amazon."
)


# 4. ENDPOINT DE PREDICCIÓN
# Define el endpoint que recibirá las peticiones POST.
@app.post("/predict")
def predict_sentiment(item: ReviewInput):
    """
    Recibe un texto de review y devuelve su predicción de sentimiento.
    """
    # 5. LÓGICA DE INFERENCIA (El "Puente")
    # El artefacto 'pyfunc' de MLflow (que contiene tu Pipeline de Sklearn)
    # fue entrenado esperando un pandas.DataFrame.
    # Debemos transformar el string simple (item.review_text) en ese DataFrame.
    # El nombre de la columna debe ser el mismo que usaste en el entrenamiento.
    data_df = pd.DataFrame(
        {"review_text": [item.review_text]}
    )

    # 6. OBTENER LA PREDICCIÓN
    # El método .predict() del modelo 'pyfunc' se encarga de todo:
    # 1. Toma el DataFrame.
    # 2. Aplica el CountVectorizer (que está dentro).
    # 3. Aplica el TfidfTransformer (que está dentro).
    # 4. Aplica el clasificador (LogisticRegression).
    # 5. Devuelve el resultado.
    try:
        prediction = modelo_cargado.predict(data_df)
        
        # El resultado de un pipeline de sklearn suele ser un array de numpy
        # (ej: array(['positive'], dtype=object)).
        # Extraemos el primer (y único) elemento para dar una respuesta limpia.
        # Usamos .item() para convertir tipos de numpy a tipos nativos de Python.
        
        # (Si tu predicción es numérica (ej: 1), .item() la convierte a int)
        # (Si es texto (ej: 'positive'), simplemente accede a ella [0])
        
        # Ajusta esta línea según lo que devuelva tu modelo:
        # Si devuelve números (0, 1, 2...), usa .item()
        # Si devuelve texto ('positive', 'negative'...), usa [0]
        
        # Probemos con [0] primero, es lo más común para etiquetas de texto
        output = prediction[0]
        
        # Si 'output' sigue siendo un tipo de numpy, conviértelo
        if hasattr(output, 'item'):
             output = output.item()

        # 7. DEVOLVER LA RESPUESTA
        # FastAPI convertirá automáticamente este diccionario de Python en un JSON.
        return {
            "review_text_recibido": item.review_text,
            "sentimiento_predicho": output
        }

    except Exception as e:
        # Captura de errores durante la predicción
        return {"error": f"Error durante la predicción: {str(e)}"}

# Define un endpoint "raíz" solo para verificar que la API funciona
@app.get("/")
def read_root():
    return {"status": "OK", "message": "API de Sentimiento de Reviews en funcionamiento."}