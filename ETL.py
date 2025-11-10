from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import BinaryType
from pyspark.sql.functions import desc

# Indicamos la version de Hadoop a usar para la sesion Spark. Configurar HADOOP_HOME
package_hdoop_aws = "org.apache.hadoop:hadoop-aws:3.4.0"
# Es un dataset publico, no necesitamos credenciales
anon_provider = "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider"

print(f"Iniciando SparkSession con JAR: {package_hdoop_aws}")

# Configuramos la sesion de Spark, estableciendo los requisitos para conectar con S3
spark = SparkSession.builder \
    .appName("PySpark S3 Connector") \
    .config("spark.jars.packages", package_hdoop_aws) \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", anon_provider) \
    .config("spark.driver.memory", "24g") \
    .getOrCreate()

print("SparkSession configurada exitosamente.")

# Establecemos el punto de lectura de los datos
bucket = "datasets-documentation"
ruta_del_archivo = "amazon_reviews/"
s3_path = f"s3a://{bucket}/{ruta_del_archivo}"

print(f"Intentando leer desde: {s3_path}")

try:
    # Apuntamos a todos los archivos que se encuentren en esa ruta
    df_raw = spark.read.parquet(s3_path)
    print("\n¡Lectura exitosa! Corrigiendo esquema (Binary -> String)...\n")
    
    # Existen columnas con datos en hexadecimal, vamos a traducirlos a lenguaje natural
    # Identifica todas las columnas que son 'binary', las cuales hay que transcribir
    binary_cols = [f.name for f in df_raw.schema.fields if isinstance(f.dataType, BinaryType)]
    print(f"Columnas binarias detectadas: {binary_cols}")

    # Crea el DataFrame corregido en UN SOLO PASO
    # Las lambdas se empiezan a ejecutar del reves, empieza por las operaciones mas a la derecha y van hacia la izquieda, esta hace primero el for
        # luego el if y finalmente si se cumple el if el col(c).cast
    df_fixed = df_raw.select(
        [col(c).cast("string").alias(c) if c in binary_cols else col(c) for c in df_raw.columns]
    )

    # Nos quedamos solo con las columnas con las que vamos a trabajar
    df_fixed = df_fixed.select('review_body', 'star_rating')
    # Eliminamos las filas con contenido nulo en una de las columnas
    df_fixed = df_fixed.dropna()

    # Guardo en local el parquet con texto legible para su analisis y procesado de datos en otro script
    #df_fixed.write.mode("overwrite").parquet("amazon_reviews_fixed.parquet")

    # Mostrar esquema del parquet
    print("\n¡Esquema Corregido!")
    
except Exception as e:
    print(f"Error al leer desde S3. Verifica la ruta, credenciales y permisos.")
    print(e)

print("Deteniendo la sesión.")
spark.stop()