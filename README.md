# Cloud - XGBoost distribuido en Google Vertex AI usando PySpark

## Descripción general

Este script ejecuta un flujo de trabajo orientado a producción para la clasificación binaria distribuida de XGBoost en Google Cloud ML, utilizando PySpark para preprocesar un gran conjunto de datos de Parquet y aprovechando el entrenamiento distribuido de XGBoost (Dask o Rabit). La guía incorpora las mejores prácticas de Google Cloud ML en cada paso, con lógica condicional para el tamaño de los datos y la aceleración de la GPU.

## 1. Requisitos previos
• Proyecto de Google Cloud con facturación habilitada
• Bucket de Google Cloud Storage (GCS) para datos y modelos
• Google Cloud Dataproc (para clústeres Spark) o Vertex AI Workbench (para entornos Jupyter administrados)
• Permisos de IAM suficientes: Administrador de almacenamiento, permisos de Dataproc/VertexAI
• Python 3.8 o superior con PySpark, XGBoost, Dask y Google Cloud Storage instalados

Práctica recomendada:
Utilice notebooks administrados por Vertex AI o clústeres de Dataproc para una computación escalable y administrada. Utilice cuentas de servicio con privilegios mínimos y habilite los controles de servicio de VPC para mayor seguridad.

A continuación, se muestra un script de Python mínimo e integral (con comentarios explicativos):

## 2. Entorno y configuración

Propósito: Centralizar la configuración para reproducibilidad y gestión de recursos en la nube.  
Configurar variables de entorno para una configuración fácil y seleccionar recursos de computación según el tamaño de los datos.  

```  
import os  
```  

# Configuración de Google Cloud  
```  
PROJECT = "your-gcp-project-id"    # Tu ID de proyecto de GCP  
BUCKET = "gs://your-gcs-bucket"    # Bucket de Google Cloud Storage para datos/modelos  
PARQUET_PATH = f"{BUCKET}/data/large_dataset.parquet"  # Ruta al conjunto de datos de entrada  
MODEL_OUTPUT_PATH = f"{BUCKET}/models/xgboost_binary_model.bst"  # Ruta de salida del modelo  
```  

# Datos y columna objetivo  
```  
TARGET_COL = "target"  # Columna objetivo para clasificación binaria  
NUMERICAL_FEATURES = ["feature1", "feature2", "feature3", …] # Reemplazar con características  
```  

# Configuración del clúster (puede ser parametrizada)  
```  
CLUSTER_SIZE = "auto" # Opciones: small, medium, large, auto (escala automáticamente según los datos)  
USE_GPU = True         # Habilitar aceleración por GPU para XGBoost  
```  

# Mejores prácticas:  
# Parametrizar todas las rutas y configuraciones para reproducibilidad y automatización.  
# Almacenar la configuración en un archivo .yaml o de entorno para repetibilidad e integración con CI/CD.  
# Usar variables de entorno para parámetros sensibles.  

# 3. Carga y Procesamiento de Datos con PySpark  
# Propósito: Cargar y preprocesar grandes conjuntos de datos de manera eficiente usando procesamiento distribuido con Spark.  

# 3.1. Inicializar Sesión de Spark  
```  
from pyspark.sql import SparkSession  

spark = SparkSession.builder \  
    .appName("XGBoost-Binary-Classification") \  
    .config("spark.executor.memory", "8g") \  # Ajustar según el tamaño del conjunto de datos  
    .config("spark.executor.cores", "4") \    # Asignar suficientes núcleos  
    .getOrCreate()  
```  

# Mejores prácticas:  
# Ajustar `executorMemory`, `executorCores` y `numExecutors` para conjuntos de datos grandes.  
# Usar Parquet para almacenamiento columnar (E/S más rápida y retención de esquema).  

# 3.2. Cargar Datos Parquet desde GCS  
```  
df = spark.read.parquet(PARQUET_PATH)              # Lectura distribuida desde GCS  
df = df.select(NUMERICAL_FEATURES + [TARGET_COL])  # Mantener solo columnas relevantes  
df = df.na.drop()                                # Eliminar filas con valores faltantes  
```  

# Mejores prácticas:  
# Usar Parquet para E/S distribuida y eficiente.  
# Usar poda de columnas (.select()) para minimizar la transferencia de datos.  
# Manejar valores faltantes antes del entrenamiento (XGBoost no maneja NaNs de forma nativa).  

# 3.3. Conversión de Datos para XGBoost  
# Convertir Spark DataFrame a Pandas (para datos pequeños) o persistir como Dask DataFrame para entrenamiento distribuido con GPU.  
```  
data_count = df.count()                 # Obtener tamaño del conjunto de datos  

if data_count < 1_000_000:                      # Estrategia para datos pequeños (<1M filas)  
    pandas_df = df.toPandas()                  # Convertir a Pandas (nodo único)  
    X = pandas_df[NUMERICAL_FEATURES].values       # Matriz de características  
    y = pandas_df[TARGET_COL].values               # Vector objetivo  
else:  
    # Para datos grandes, escribir a CSV/Parquet y usar Dask para procesamiento distribuido  
    LOCAL_TMP_PATH = "/tmp/xgb_data/"  
    df.write.mode('overwrite').parquet(LOCAL_TMP_PATH)    # Persistir datos procesados  
    # Mejores prácticas:  
    # Usar formatos de archivo distribuidos y Dask para escalabilidad.  
    # Evitar errores de OOM en el driver al no recolectar grandes conjuntos de datos en un solo nodo.  
    # Usar formatos eficientes (Parquet) para almacenamiento intermedio.  
```  

# 5. Entrenamiento Distribuido de XGBoost (con Soporte para GPU)  
# Propósito: Entrenar modelo XGBoost con aceleración por GPU y computación distribuida.  
# Puedes usar la interfaz Dask de XGBoost para entrenamiento distribuido y escalable.  
    • Para datos pequeños: Usar xgboost.train nativo.  
    • Para datos grandes: Usar dask-xgboost o xgboost.dask.  
```  
import xgboost as xgb  

if data_count < 1_000_000:                # Conjunto de datos pequeño  
    # Datos pequeños: entrenamiento local  
    dtrain = xgb.DMatrix(X, label=y)      # Crear matriz de datos XGBoost  
    params = {                            # Parámetros de entrenamiento  
        "objective": "binary:logistic",       # Clasificación binaria  
        "eval_metric": "auc",                        # Métrica de evaluación (Área Bajo la Curva)  
        "tree_method": "gpu_hist" if USE_GPU else "hist",     # Aceleración por GPU  
        "verbosity": 2                                        # Mostrar logs de entrenamiento  
    }  
    model = xgb.train(params, dtrain, num_boost_round=100)    # Entrenamiento local  
else:  
    # Datos grandes: XGBoost distribuido con Dask  
    from dask.distributed import Client  
    import dask.dataframe as dd  

    # Inicializar cliente Dask (conecta al clúster)  
    client = Client(n_workers=4, threads_per_worker=1)  # Personalizar según el clúster  

    # Cargar datos persistidos como Dask DataFrame (distribuido)  
    ddf = dd.read_parquet(LOCAL_TMP_PATH)  
    X_dd = ddf[NUMERICAL_FEATURES]          # Características distribuidas  
    y_dd = ddf[TARGET_COL]                  # Etiquetas distribuidas  

    # Crear DMatrix distribuido (estructura de datos optimizada de XGBoost)  
    dtrain = xgb.dask.DaskDMatrix(client, X_dd, y_dd)  

    # Parámetros de entrenamiento distribuido  
    params = {  
        "objective": "binary:logistic",  
        "eval_metric": "auc",  
        "tree_method": "gpu_hist" if USE_GPU else "hist",  # Aceleración por GPU  
        "verbosity": 2  
    }  

    # Entrenamiento distribuido en el clúster Dask  
    output = xgb.dask.train(client, params, dtrain, num_boost_round=100)  
    model = output['booster']              # Objeto del modelo entrenado  
```  

# Mejores prácticas:  
# Usar 'gpu_hist' para entrenamiento 5-10x más rápido en GPUs NVIDIA.  
# Monitorear eval_metric (AUC) durante el entrenamiento para parada temprana.  
# Escalar trabajadores de Dask horizontalmente para conjuntos de datos más grandes.  

# 6. Evaluación y Guardado del Modelo  
# Propósito: Validar el rendimiento del modelo y guardar artefactos en el almacenamiento en la nube.  

# 6.1. Generar Predicciones  
```  
# Para datos pequeños  
if data_count < 1_000_000:  
    y_pred = model.predict(dtrain)             # Predicción local  
    # Puedes agregar cálculo de ROC/AUC aquí (p.ej., sklearn.metrics.roc_auc_score)  
else:  
   # Para Dask, usar xgb.dask.predict  
    y_pred=xgb.dask.predict(client, model, dtrain).compute() # Predicción distribuida  
    # Apagar clúster Dask después del entrenamiento  
    client.shutdown()  
```  

# 6.2. Guardar Modelo en GCS  
```  
model.save_model("/tmp/model.bst")     # Guardado temporal local  
from google.cloud import storage  

# Subir modelo a GCS  
client = storage.Client(project=PROJECT)  
bucket = client.bucket(BUCKET.replace("gs://", ""))    # Extraer nombre del bucket  
blob = bucket.blob("models/xgboost_binary_model.bst")  
blob.upload_from_filename("/tmp/model.bst")  
```  

# Mejores prácticas:  
# Siempre guardar modelos en almacenamiento en la nube para:  
* Control de versiones (versiones de objetos en GCS)  
* Despliegue en Vertex AI o Cloud Functions  
* Reproducibilidad en diferentes entornos  
# Agregar métricas de evaluación del modelo (AUC, precisión/exhaustividad) a un sistema de seguimiento.  

# Recomendaciones para Producción:  
1) Usar Vertex AI Pipelines para orquestación de flujos de trabajo.  
2) Implementar ajuste de hiperparámetros (Vertex AI Vizier).  
3) Agregar hooks de monitoreo para métricas de entrenamiento.  
4) Proteger datos con VPC Service Controls.  
5) Contenerizar usando Docker para portabilidad.  

# 7. Resumen de Mejores Prácticas  
    • Parametrizar configuraciones para reproducibilidad y automatización.  
    • Usar Parquet para almacenamiento, Dask/Spark para computación distribuida.  
    • Aprovechar GPUs para entrenamiento rápido y a gran escala de XGBoost (con gpu_hist).  
    • Siempre monitorear métricas como AUC durante el entrenamiento.  
    • Guardar modelos en GCS para control de versiones y despliegue.  
    • Proteger recursos con IAM, VPC Service Controls y permisos mínimos.  
    • Automatizar mediante Vertex AI Pipelines o Cloud Composer para flujos de trabajo repetibles.  

# Notas Finales  
    • Produc

