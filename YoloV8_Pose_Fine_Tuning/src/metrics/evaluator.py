"""
Evaluador principal que integra todas las métricas de Pose Estimation.
Este archivo actúa como el "juez" que califica qué tan bien el modelo
predice los keypoints en comparación con los datos reales (ground truth).
"""
import numpy as np
import pandas as pd
from typing import Dict
from .pck import PCKMetric
from .oks import OKSMetric


class PoseEvaluator:
    """
    Evaluador completo de métricas para pose estimation.
    
    Esta clase encapsula la lógica para:
    1. Calcular métricas clave como PCK y OKS.
    2. Calcular precisión y recall basados en OKS.
    3. Almacenar los resultados de cada época de entrenamiento.
    4. Guardar un resumen final en formato CSV para análisis posterior.
    """
    
    def __init__(self, config: Dict):
        """
        Constructor del evaluador. Se inicializa con la configuración del proyecto.
        
        Args:
            config (Dict): Diccionario de configuración (usualmente un .yaml) que contiene
                           información sobre los keypoints y los umbrales de validación.
        """
        # --- 1. Cargar información básica de los keypoints ---
        self.keypoint_names = config['keypoints']['names']
        self.num_keypoints = config['keypoints']['num_keypoints']
        
        # --- 2. Inicializar las métricas que se van a calcular ---
        
        # PCK (Percentage of Correct Keypoints): Mide el % de keypoints predichos
        # que están dentro de un umbral de distancia del keypoint real.
        # Se crea una instancia para cada umbral (e.g., 0.05, 0.1) especificado en el config.
        self.pck_metrics = {
            f'pck@{th}': PCKMetric(threshold=th)
            for th in config['validation']['custom_metrics']['pck_thresholds']
        }
        
        # OKS (Object Keypoint Similarity): Métrica más avanzada, similar al IoU pero para poses.
        # Considera la distancia, el tamaño del objeto y la desviación estándar de cada keypoint.
        # Es el estándar para la evaluación en datasets como COCO.
        self.oks_metric = OKSMetric(
            sigmas=np.array(config['keypoints']['oks_sigmas'])
        )
        
        # --- 3. Preparar almacenamiento de resultados ---
        # Esta lista guardará un diccionario de métricas por cada época.
        self.results = []
        
    def evaluate_batch(
        self,
        predictions: Dict,
        ground_truth: Dict
    ) -> Dict[str, float]:
        """
        Evalúa un batch de predicciones contra sus correspondientes ground truth.
        Este es el corazón del evaluador, donde se calculan todas las métricas.
        
        Args:
            predictions (Dict): Diccionario con las salidas del modelo (bboxes, keypoints).
            ground_truth (Dict): Diccionario con las etiquetas correctas.
            
        Returns:
            Dict[str, float]: Un diccionario con los nombres de las métricas y sus valores para el batch.
        """
        metrics = {}
        
        # Asegurar que usamos solo (x, y) para las métricas, ignorando la confianza
        # predictions['keypoints'] viene como [N, K, 3] (x, y, conf)
        # ground_truth['keypoints'] viene como [N, K, 2] (x, y)
        pred_xy = predictions['keypoints'][..., :2]
        gt_xy = ground_truth['keypoints']
        
        # --- 1. Calcular todas las métricas PCK ---
        for name, pck_metric in self.pck_metrics.items():
            # Calcula el PCK global (promedio de todos los keypoints) y el PCK por cada keypoint.
            pck_global, pck_per_kpt = pck_metric.compute(
                pred_xy,
                gt_xy,
                predictions['bboxes']
            )
            # Guardar el PCK global (ej: 'pck@0.05': 0.85)
            metrics[name] = pck_global
            
            # Guardar el PCK para cada keypoint individualmente (ej: 'pck@0.05_cabeza': 0.9)
            for i, kpt_name in enumerate(self.keypoint_names):
                metrics[f'{name}_{kpt_name}'] = pck_per_kpt[i]
        
        # --- 2. Calcular la métrica OKS ---
        oks_scores = self.oks_metric.compute_batch(
            pred_xy,
            gt_xy,
            predictions['bboxes'],
            ground_truth['visibilities']
        )
        # Guardar la media y desviación estándar de los scores OKS del batch.
        metrics['oks_mean'] = oks_scores.mean()
        metrics['oks_std'] = oks_scores.std()
        
        # --- 3. Calcular Precisión y Recall basados en OKS ---
        # Una detección se considera "válida" o "correcta" (True Positive) si su score OKS
        # supera un umbral comúnmente aceptado (en este caso, 0.5).
        valid_detections = (oks_scores > 0.5).sum()
        total_predictions = len(pred_xy)
        total_ground_truth = len(gt_xy)
        
        # Precisión: De todo lo que predije, ¿qué porcentaje fue correcto?
        metrics['precision'] = valid_detections / total_predictions if total_predictions > 0 else 0
        # Recall: De todo lo que debía encontrar, ¿qué porcentaje encontré?
        metrics['recall'] = valid_detections / total_ground_truth if total_ground_truth > 0 else 0
        
        return metrics
    
    def add_result(self, epoch: int, metrics: Dict):
        """
        Agrega el diccionario de métricas de una época completa a la lista de resultados.
        
        Args:
            epoch (int): El número de la época que acaba de terminar.
            metrics (Dict): El diccionario de métricas calculado para esa época.
        """
        metrics['epoch'] = epoch
        self.results.append(metrics)
    
    def get_summary(self) -> pd.DataFrame:
        """
        Convierte la lista de resultados acumulados en un DataFrame de Pandas.
        Esto facilita el análisis y la visualización posterior.
        
        Returns:
            pd.DataFrame: Una tabla con las métricas de todas las épocas.
        """
        return pd.DataFrame(self.results)
    
    def save(self, filepath: str):
        """
        Guarda el resumen de métricas en un archivo CSV.
        
        Args:
            filepath (str): La ruta donde se guardará el archivo .csv.
        """
        df = self.get_summary()
        df.to_csv(filepath, index=False)
        print(f"✅ Métricas guardadas en: {filepath}")
