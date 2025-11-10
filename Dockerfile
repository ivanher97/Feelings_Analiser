# Fichero: Dockerfile

# 1. Usar una imagen base de Python oficial y ligera
FROM python:3.9-slim

# 2. Establecer un directorio de trabajo dentro del contenedor
# A partir de aquí, todos los comandos se ejecutan en /app
WORKDIR /app

# 3. Copiar PRIMERO el fichero de requisitos
# (Esto optimiza el caché de Docker)
COPY requirements.txt .

# 4. Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el resto del código de la aplicación Y los artefactos
COPY main.py .
COPY mlruns ./mlruns

# 6. Exponer el puerto en el que Uvicorn correrá
EXPOSE 8000

# 7. Definir el comando para ejecutar la aplicación
# Usamos 0.0.0.0 para que sea accesible desde fuera del contenedor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]