
# Sistema de Estimación de Pose en Salmones (Salmon-Pose-Estimation)

Este proyecto implementa un pipeline completo para el *fine-tuning* de modelos **YOLOv8-Pose** con el objetivo de localizar con precisión 11 puntos anatómicos clave en salmones. El sistema está diseñado siguiendo principios de modularidad, configuración externa y reproducibilidad para facilitar la experimentación y el futuro despliegue en entornos de producción.

## 1. Características Principales

* **Arquitectura Modular**: El código está organizado en módulos cohesivos con responsabilidades únicas (modelos, métricas, callbacks, etc.).
* **Configuración Centralizada**: Todo el comportamiento del pipeline (hiperparámetros, rutas, etc.) se controla desde archivos YAML, permitiendo experimentar sin modificar el código.
* **Métricas Personalizadas**: Integra métricas avanzadas de pose como OKS (Object Keypoint Similarity) y PCK (Percentage of Correct Keypoints) directamente en el ciclo de validación de `ultralytics`.
* **Pipeline Automatizado**: Incluye scripts para automatizar todo el flujo de trabajo: preparación del dataset, entrenamiento, evaluación e inferencia.

## 2. Estructura del Proyecto

La organización del repositorio separa claramente la lógica del núcleo (`src`), la configuración (`configs`), los datos (`data`) y los scripts de ejecución (`scripts`).

```
YoloV8_Pose_Fine_Tuning/
├── configs/                # Archivos de configuración (hiperparámetros, etc.)
├── data/                   # Dataset procesado y listo para YOLO (generado automáticamente)
├── dataset_fuente/         # Directorio sugerido para TUS datos originales
├── notebooks/              # Jupyter Notebooks para exploración y análisis
├── outputs/                # Directorio para guardar resultados (modelos, métricas, logs)
├── scripts/                # Scripts para ejecutar el pipeline (setup, train, eval, infer)
├── src/                    # Código fuente principal de la aplicación
│   ├── callbacks/
│   ├── metrics/
│   ├── models/
│   └── utils/
├── tests/                  # Pruebas unitarias para asegurar la calidad del código
├── README.md
└── requirements.txt        # Dependencias del proyecto
```

## 3. Cómo Empezar

### 3.1. Prerrequisitos

* Python 3.8 o superior
* PyTorch con soporte para CUDA (recomendado)

### 3.2. Instalación

1. Clona este repositorio:

   ```
   gh repo clone a10ns0/SalmonMetric
   cd YoloV8_Pose_Fine_Tuning
   ```
2. Crea un entorno virtual e instala las dependencias:

   ```
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### 3.3. Flujo de Trabajo

El pipeline se ejecuta a través de los scripts ubicados en el directorio `scripts/`.

#### Paso 1: Configuración del Entorno y Preparación del Dataset

Antes de empezar, **coloca todas tus imágenes (`.jpg`, `.png`) y sus correspondientes archivos de etiquetas (`.txt`) en una única carpeta de origen**. Puedes nombrarla como desees (ej. `mis_datos_anotados`, `dataset_fuente`, etc.).

El script `00_setup.py` es multifuncional. Su tarea principal es preparar todo el entorno. Opcionalmente, si le proporcionas la ruta a tu carpeta de datos, también procesará el dataset: lo dividirá en conjuntos de entrenamiento, validación y prueba (por defecto 80/10/10), y generará el archivo `data/data.yaml` que YOLOv8 necesita.

Ejecuta el script desde la raíz del proyecto.

**Para configurar el entorno Y procesar tu dataset:**

```
# Ejemplo: si tus datos están en una carpeta llamada 'mis_datos_anotados'
python scripts/00_setup.py --source-dir mis_datos_anotados
```

**Para configurar SOLO el entorno (sin procesar datos):**

```
# Omite el argumento --source-dir
python scripts/00_setup.py
```

Este comando creará todos los directorios necesarios y descargará el modelo base de YOLO.

#### Paso 2: Entrenamiento del Modelo

Una vez que el dataset ha sido procesado y se encuentra en la carpeta `data/`, lanza el script de entrenamiento.

```
python scripts/01_train.py
```

Este script carga la configuración desde `configs/training_config.yaml`, instancia el modelo y los callbacks personalizados, y comienza el `fine-tuning`. Los resultados, incluyendo los pesos del modelo (`best.pt`, `last.pt`) y los logs, se guardarán en el directorio `outputs/`.

#### Paso 3: Evaluación del Modelo

Utiliza el mejor modelo guardado (`best.pt`) para realizar una evaluación formal sobre el conjunto de prueba.

```
python scripts/02_evaluate.py
```

#### Paso 4: Inferencia

Ejecuta una inferencia de ejemplo en una imagen o video para visualizar las predicciones del modelo.

```
python scripts/03_inference.py
```

## 4. Arquitectura Detallada

### `src` - El Corazón del Proyecto

* **`src/models/yolo_wrapper.py`**: Abstrae el proceso de entrenamiento de YOLOv8. Es una clase que carga configuraciones, prepara hiperparámetros y orquesta el `fine-tuning`, permitiendo la inyección de callbacks personalizados.
* **`src/metrics/`**: Contiene la implementación de las métricas de evaluación de pose.
  * `oks.py`: Implementa **Object Keypoint Similarity**, la métrica estándar de COCO que pondera los errores de los keypoints según su variabilidad.
  * `pck.py`: Implementa **Percentage of Correct Keypoints**, una métrica más simple que cuenta los keypoints predichos dentro de un umbral de distancia.
  * `evaluator.py`: Orquesta las métricas anteriores para evaluar un batch completo de predicciones y agregar los resultados.
* **`src/callbacks/custom_metrics_callback.py`**: Actúa como un puente entre el motor de `ultralytics` y nuestro sistema de métricas. "Escucha" el evento de fin de validación (`on_val_end`) para calcular y registrar las métricas OKS y PCK en cada época.
* **`src/utils/download_utils.py`**: Utilidad para gestionar la descarga de los modelos base de YOLO de forma segura y eficiente.

### `configs` - El Cerebro del Proyecto

* **`training_config.yaml`**: Archivo de configuración principal que controla todos los aspectos del entrenamiento: número de épocas, tamaño de batch, aumentos de datos y, de forma crítica, los **pesos de la función de pérdida**. Permite ajustar la importancia relativa de la localización de la caja (`box`), la clasificación (`cls`) y la pose (`pose`).
* **`keypoints_config.yaml`**: Define la semántica de la tarea: el número de keypoints, sus nombres y sus valores `sigma` para el cálculo de OKS. Externalizar esta información permite adaptar el código a diferentes esqueletos de pose sin modificar la lógica.

## 5. Próximos Pasos y Mejoras

El estado actual del proyecto proporciona una base sólida para el entrenamiento de modelos de pose. Las futuras líneas de trabajo se centran en la transición a un sistema de producción en tiempo real:

1. **Tracking y Manejo de Oclusiones**: Implementar un algoritmo de tracking (ej., DeepSORT, Filtro de Kalman) para seguir a los peces a través de los frames de un video y manejar oclusiones temporales.
2. **Calibración Píxel-Métrica**: Desarrollar un módulo de calibración de cámara para convertir las mediciones de píxeles a unidades métricas reales (centímetros).
3. **Optimización para Inferencia en Edge**: Exportar el modelo entrenado (`.pt`) a **NVIDIA TensorRT** para optimizar su rendimiento en hardware embebido como la NVIDIA Jetson y alcanzar los objetivos de FPS.
4. **Pruebas Unitarias**: Poblar el directorio `tests/` con pruebas unitarias para garantizar la robustez y fiabilidad del código.

```

```
