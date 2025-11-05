
# Sistema de Estimación de Pose en Salmones (Salmon-Pose-Estimation)

Este proyecto implementa un pipeline completo y modular para el *fine-tuning* de modelos **YOLOv8-Pose**. El objetivo es localizar con precisión 11 puntos anatómicos clave en salmones, sentando las bases para una futura estimación de dimensiones biométricas.

El diseño se fundamenta en principios de **modularidad, configuración externa y reproducibilidad** para facilitar la experimentación, el mantenimiento y el despliegue en entornos de producción.

## 1. Características Principales

* **Arquitectura Modular**: El código está organizado en módulos cohesivos con responsabilidades únicas (modelos, métricas, callbacks, etc.).
* **Configuración Centralizada**: Todo el comportamiento del pipeline se controla desde archivos YAML (`configs/`), permitiendo experimentar con hiperparámetros sin tocar el código fuente.
* **Métricas de Pose Avanzadas**: Integra métricas como **OKS** (Object Keypoint Similarity) y **PCK** (Percentage of Correct Keypoints) directamente en el ciclo de validación de `ultralytics` para una evaluación precisa del modelo.
* **Pipeline Automatizado**: Scripts para automatizar todo el flujo de trabajo: preparación del entorno y del dataset, entrenamiento, evaluación e inferencia.

## 2. Estructura del Proyecto

La organización del repositorio separa claramente la lógica del núcleo (`src`), la configuración (`configs`), los datos procesados (`data`) y los scripts de ejecución (`scripts`).

```
YoloV8_Pose_Fine_Tuning/
├── configs/                # Archivos de configuración (hiperparámetros, keypoints)
├── data/                   # Dataset procesado y estructurado para YOLO (generado)
├── notebooks/              # Jupyter Notebooks para exploración y análisis
├── outputs/                # Directorio para guardar resultados (modelos, métricas, logs)
├── scripts/                # Scripts para ejecutar el pipeline (setup, train, eval, infer)
├── src/                    # Código fuente principal de la aplicación
│   ├── callbacks/
│   ├── metrics/
│   ├── models/
│   └── utils/
├── tests/                  # Pruebas unitarias
├── README.md
└── requirements.txt        # Dependencias del proyecto
```

## 3. Guía de Inicio Rápido

### 3.1. Prerrequisitos

* Python 3.8 o superior
* PyTorch con soporte para CUDA (altamente recomendado para el entrenamiento)

### 3.2. Instalación

1. Clona este repositorio en tu máquina local.
2. Desde la raíz del proyecto, crea un entorno virtual e instala las dependencias:
   ```
   # Crear y activar entorno virtual
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate

   # Instalar dependencias
   pip install -r requirements.txt
   ```

### 3.3. Flujo de Trabajo Completo

El pipeline se opera a través de los scripts ubicados en el directorio `scripts/`.

#### Paso 1: Configuración del Entorno y Preparación del Dataset

Este es el primer y más importante paso. El script `00_setup.py` se encarga de:

* Crear toda la estructura de directorios necesaria (`outputs/`, `data/`).
* Verificar que los archivos de configuración existan.
* Descargar el modelo base de YOLOv8.
* **Procesar tu dataset de imágenes y etiquetas** para dejarlo listo para YOLO.

**Antes de ejecutar, asegúrate de tener una carpeta con todas tus imágenes (`.jpg`) y sus correspondientes archivos de etiquetas (`.txt`) de CVAT/YOLO.**

Luego, ejecuta el script `00_setup.py` desde la raíz del proyecto, indicando la ubicación de tu dataset con el argumento `--source-dir`:

```
# Ejemplo: si tus datos están en una carpeta llamada 'dataset_anotado'
python scripts/00_setup.py --source-dir dataset_anotado
```

El script dividirá tus datos en conjuntos de entrenamiento, validación y prueba, y generará automáticamente el archivo `data/data.yaml` que el entrenador necesita.

Si quieres ejecutar el script sin procesar un dataset (por ejemplo, solo para crear los directorios), simplemente omite el argumento `--source-dir`.

#### Paso 2: Entrenamiento del Modelo

Una vez que el setup se ha completado y tu dataset está procesado en la carpeta `data/`, puedes iniciar el *fine-tuning*.

```
python scripts/01_train.py
```

Este script cargará la configuración desde `configs/training_config.yaml` y comenzará el entrenamiento. Puedes monitorear el progreso en la consola y los resultados se guardarán en `outputs/runs/`. El mejor modelo se guardará como `best.pt`.

#### Paso 3: Evaluación del Modelo

Para obtener métricas finales sobre el conjunto de prueba utilizando tu mejor modelo entrenado:

```
python scripts/02_evaluate.py
```

#### Paso 4: Inferencia

Para visualizar las predicciones de tu modelo en una nueva imagen o video:

```
python scripts/03_inference.py
```

## 4. Arquitectura Detallada del Código (`src`)

* **`src/models/yolo_wrapper.py`**: Abstrae y orquesta el proceso de entrenamiento. Carga configuraciones, prepara hiperparámetros y permite la inyección de *callbacks* personalizados.
* **`src/metrics/`**: Contiene la implementación de las métricas de evaluación de pose (`oks.py`, `pck.py`) y un orquestador (`evaluator.py`) que las calcula y agrega.
* **`src/callbacks/custom_metrics_callback.py`**: Actúa como un puente entre el motor de `ultralytics` y nuestro sistema de métricas. Se activa al final de cada época de validación para calcular OKS y PCK.
* **`src/utils/download_utils.py`**: Utilidad robusta para gestionar la descarga de los modelos base de YOLO.

## 5. Configuración (`configs`)

* **`training_config.yaml`**: Controla todos los hiperparámetros del entrenamiento, incluyendo épocas, tamaño de batch, aumentos de datos y, crucialmente, los **pesos de la función de pérdida** para priorizar la precisión de la pose.
* **`keypoints_config.yaml`**: Define la semántica de la tarea: el número y nombre de los keypoints, y sus valores `sigma` para el cálculo de la métrica OKS.

## 6. Próximos Pasos y Mejoras

* **Tracking y Manejo de Oclusiones**: Implementar un algoritmo de tracking (ej., DeepSORT, Filtro de Kalman) para seguir a los peces a través de los frames de un video.
* **Calibración Píxel-Métrica**: Desarrollar un módulo de calibración de cámara (OpenCV) para convertir las mediciones de píxeles a unidades métricas reales (centímetros).
* **Optimización para Inferencia Edge**: Exportar el modelo entrenado a **NVIDIA TensorRT** para acelerar drásticamente su rendimiento en hardware embebido como la NVIDIA Jetson.

```

```
