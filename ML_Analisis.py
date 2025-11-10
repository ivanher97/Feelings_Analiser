from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql.functions import desc
from pyspark.sql.functions import when
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# Inicia una sesión simple, ya no necesita S3
spark = SparkSession.builder \
    .appName("ML Analysis") \
    .config("spark.driver.memory", "24g") \
    .getOrCreate()

print("Leyendo Parquet limpio desde disco local...")
df_analysis = spark.read.parquet("./data/amazon_reviews_fixed.parquet")

try:
    # --------------------------- LIMPIEZA COLUMNAS ----------------------------------
    #  Mostrar esquema del parquet que hemos almacenado en el script anterior
    df_analysis.printSchema()
    # Obtenermos el conteo de la columna 'star_rating', cuantas veces se encuentra cada elemento distinto de la columa
    df_analysis.groupBy('star_rating').count().orderBy(desc('count')).show()

    # Eliminamos las reviews con una puntuacion de 3 estrellas
    df_binary = df_analysis.filter("star_rating != 3")
    # Cambiamos los valores 1 y 2 por 0 y 4 y 5 por 1. Ademas luego solo nos quedamos con 2 columnas
    df_binary = df_binary.withColumn('label', when((df_binary.star_rating == 1) | (df_binary.star_rating == 2), 0)
                                     .when((df_binary.star_rating == 4) | (df_binary.star_rating == 5), 1)).select('review_body', 'label')
    
    df_binary.show()

    # --------------------------- DESBALANCE ----------------------------------
    print("Analizando desbalanceo del DataFrame binario:")
    # Acción 1: Conteo total
    ## Se puede hacer mas acedalnte despues del .collect() con la suma de neg_count y pos_count -> Linea 50
    total_binario = df_binary.count() ## Se podria quitar con linea 50
    print(f"Total de reviews (binario): {df_binary.count()}") ## Se podria quitar con linea 50

    # Acción 2: Conteo por label
    conteo_label = df_binary.groupBy('label').count() ## Lo tenemos con neg_count y con pos_count
    conteo_label.show()


    # --------------------------- CORRECCION DESBALANCE ----------------------------------
    # conteo_label es un df mucho mas pequeño que con los que hemos trabajando antes, por lo que podemos manejarlo desde cache
    conteo_label.cache()
    # Aqui si podemos hacer un .collect() por un menor tamaño de los datos. Podemos manejarlos sin que la memoria explote
    count_label_local = conteo_label.collect()
    # Pasamos los datos a un diccionario de python para trabajar con python puro, 0 spark
    count_dict = {row['label']: row['count'] for row in count_label_local}
    neg_count = count_dict[0]
    pos_count = count_dict[1]

    '''
    total_binario = neg_count + pos_count
    '''
    # Sacamos el valor de relacion entre positivos y negativos
    frac = neg_count / pos_count

    # Separamos el df en dos mas pequeños en funcion del tipo de review
    df_neg = df_binary.filter('label == 0')
    df_pos = df_binary.filter('label == 1')

    # Balanceamos la cantidad de positivos, para que haya un nuemro similar a negativos
    df_pos_sample = df_pos.sample(fraction = frac)
    # Unimos los df pos_sample y neg, los cuales tienen una cantidad de valores similar y obtenemos el total
    df_balanced = df_neg.union(df_pos_sample)
    # Alamacenamos en cache, ya que le usamos un par de veces
    df_balanced.persist(StorageLevel.MEMORY_AND_DISK)
    total_balanced = df_balanced.count()


    # --------------------------- MUESTREO 1M < ORIGINAL 39M ----------------------------------
    # Cogemos 1M de filas de las 39M originales
    print("Creando muestra para Scikit-learn (1.0M de filas)...")
    sample_fraction = 1_000_000 / total_balanced
    # Tomamos la muestra aleatoria
    df_sample_sklearn = df_balanced.sample(fraction=sample_fraction, seed=42)
    # FORZAMOS A UN ÚNICO FICHERO
    df_sample_sklearn_single_file = df_sample_sklearn.repartition(1)   
    # Guardamos el resultado que usaremos en la Fase 3
    df_sample_sklearn_single_file.write \
        .mode("overwrite") \
        .parquet("./data/sklearn_sample_1M.parquet")

    print("Muestra para Scikit-learn guardada.")

    # Detenemos la sesión de Spark
    spark.stop() 




    '''
    # Agrupamos por los diferentes valores del label y contamos por cada grupo
    conteo_label_bal = df_balanced.groupBy('label').count()
    conteo_label_bal.withColumn('percentage', (col('count') / total_balanced) * 100).show()
    '''
    
    # Entrenamiento con el df de 39M de filas
    '''
    # --------------------------- TOKENIZACION Y LIMPIEZA PALABRAS ----------------------------------

    # Divide 'review_body' en una lista de palabras (tokens)
    # "Me gusta" -> ["me", "gusta"]
    tokenizer = Tokenizer(
        inputCol = "review_body", 
        outputCol = "tokens"
    )

    # Quita palabras comunes. 
    # ["me", "gusta"] -> ["gusta"]
    # Usamos las listas por defecto para inglés
    stop_words = StopWordsRemover.loadDefaultStopWords("english")
    remover = StopWordsRemover(
        inputCol = "tokens", 
        outputCol = "clean_tokens", 
        stopWords = stop_words
    )

    # Convierte la lista de palabras en un vector de frecuencias.
    # ["gusta", "gusta"] -> Vector(índice_hash_gusta: 2.0)
    hashingTF = HashingTF(
        inputCol = "clean_tokens", 
        outputCol = "raw_features", 
        numFeatures = 2**18
    )

    # Pondera el vector de frecuencias. Da más peso a las palabras raras
    # y menos peso a las comunes (como "producto" o "tienda").
    # Este es un ESTIMADOR: debe APRENDER de los datos.
    idf = IDF(
        inputCol = "raw_features", 
        outputCol = "features"  # <-- La columna final que usará el modelo
    )


    # --------------------------- PIPELINE ----------------------------------
    # Definimos el orden de las operaciones
    pipeline_nlp = Pipeline(stages=[
        tokenizer, 
        remover, 
        hashingTF, 
        idf
    ])


    # --------------------------- TRANSFORMACION ----------------------------------
    pipeline_model = pipeline_nlp.fit(df_balanced)

    print("Transformando los datos...")
    df_features = pipeline_model.transform(df_balanced)

    print("Esquema del DataFrame listo para el modelo:")
    df_features.printSchema()

    df_listo_para_entrenar = df_features.select("label", "features")
    print("DataFrame final listo para el modelo (optimizado):")
    df_listo_para_entrenar.printSchema()


    # --------------------------- ENTRENAMIENTO ----------------------------------
    # Dividimos los datos en 80% entrenamiento 20% test
    df_train, df_test = df_listo_para_entrenar.randomSplit([0.8, 0.2], seed = 42)
    # Guardamos en disco los df para evitar saturar RAM
    df_test.persist(StorageLevel.MEMORY_AND_DISK)
    df_train.persist(StorageLevel.MEMORY_AND_DISK)

    # Entrenamos el modelo
    lr = LogisticRegression(featuresCol ='features', labelCol = 'label')
    lr_model = lr.fit(df_train)

    # Probamos con df_test
    predictions = lr_model.transform(df_test)
    # Calculo de prediccion AUC
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label")
    auc = evaluator.evaluate(predictions)
    print(f"AUC-ROC: {auc:.4f}")
    '''

    # ----------------- ESTO ESTA BIEN PERO ES CON SPARK-------------------
        # Menos eficiente
    '''
    # Opcional pero recomendado: Mostrar el porcentaje
    conteo_label.withColumn('percentage', (col('count') / total_binario) * 100).show()

    # Dividimos en reviews positivas y negativas por el label que hemos creado antes
    df_neg = df_binary.filter('label == 0')
    df_pos = df_binary.filter('label == 1')

    # Cantidades de cada uno
    pos_count = df_pos.count()
    neg_count = df_neg.count()
    print(f"Cantidad de negativos: {neg_count}")
    print(f"Cantidad de positivas: {pos_count}")

    # Reducumos el numero de data positiva
    df_pos_sample = df_pos.sample(fraction = (neg_count / pos_count))
    print(f"Cantidad de positivas despues de reducir muestras: {df_pos_sample.count()}")

    # Unimos en un df_balanceado
    df_balanced = df_neg.union(df_pos_sample)
    total_balanced = df_balanced.count()

    # Mostramos el contero del dataset balanceado
    conteo_label_bal = df_balanced.groupBy('label').count()
    conteo_label_bal.show()

    # Opcional pero recomendado: Mostrar el porcentaje
    conteo_label_bal.withColumn('percentage', (col('count') / total_balanced) * 100).show()
    '''

except Exception as e:
    print(f"Error al leer desde local. Verifica la ruta, credenciales y permisos.")
    print(e)