"""
Callbacks personalizados para YOLOv8
"""
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from src.metrics.evaluator import PoseEvaluator


class CustomMetricsCallback:
    """Callback para calcular métricas personalizadas durante entrenamiento"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Ruta al archivo de configuración
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.evaluator = PoseEvaluator(self.config)
        self.batch_metrics = []
        
    def on_val_batch_end(self, validator):
        """Ejecutado al final de cada batch de validación"""
        try:
            # Verificar que existan predicciones y batch
            if not hasattr(validator, 'preds') or not validator.preds:
                return
            
            batch = validator.batch
            if batch is None:
                return

            # 1. Determinar el tamaño del batch actual
            # batch['keypoints'] tiene forma [Batch_Size, Num_Keypoints, 3]
            gt_kpts_batch = batch['keypoints'].detach().cpu().numpy()
            batch_size = gt_kpts_batch.shape[0]

            # 2. Recuperar las predicciones correspondientes a este batch
            # validator.preds es una lista acumulativa de objetos Results (uno por imagen)
            # Tomamos los últimos 'batch_size' elementos
            if len(validator.preds) < batch_size:
                return # Seguridad por si acaso
            
            current_preds = validator.preds[-batch_size:]

            # 3. Extraer keypoints de los objetos Results
            pred_kpts_list = []
            for result in current_preds:
                # result.keypoints.data es un Tensor [1, K, 3] (x, y, conf)
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    kpts = result.keypoints.data.cpu().numpy()
                    # Asegurar que sea [1, K, 3]
                    if len(kpts.shape) == 2: 
                        kpts = np.expand_dims(kpts, axis=0)
                    pred_kpts_list.append(kpts)
                else:
                    # Si no hay detección, rellenar con ceros [1, K, 3]
                    # K = gt_kpts_batch.shape[1]
                    pred_kpts_list.append(np.zeros((1, gt_kpts_batch.shape[1], 3)))

            # Concatenar para tener [Batch_Size, K, 3]
            if pred_kpts_list:
                pred_kpts_batch = np.concatenate(pred_kpts_list, axis=0)
            else:
                return

            # 4. Preparar diccionarios para el evaluador
            predictions = {
                'keypoints': pred_kpts_batch, 
                'bboxes': batch['bboxes'].detach().cpu().numpy() if 'bboxes' in batch else None
            }
            
            ground_truth = {
                'keypoints': gt_kpts_batch[..., :2], # [N, K, 2] (x, y)
                'visibilities': gt_kpts_batch[..., 2] # [N, K] (visibilidad)
            }
            
            # Calcular métricas del batch
            metrics = self.evaluator.evaluate_batch(predictions, ground_truth)
            self.batch_metrics.append(metrics)
            
        except Exception as e:
            # Capturamos errores silenciosamente para no detener el entrenamiento
            # pero imprimimos advertencia la primera vez
            if len(self.batch_metrics) == 0:
                print(f"⚠️ Advertencia en CustomMetricsCallback: {e}")

    def on_val_end(self, validator):
        """Ejecutado al final de cada validación"""
        if not self.batch_metrics:
            print("⚠️ No se calcularon métricas personalizadas (lista vacía).")
            return

        print("\n🔍 Calculando métricas personalizadas...")
        
        # Promediar métricas de todos los batches
        df_batch = pd.DataFrame(self.batch_metrics)
        avg_metrics = df_batch.mean().to_dict()
        
        # Agregar resultado al evaluador
        self.evaluator.add_result(validator.epoch, avg_metrics)
        
        # Mostrar resumen en consola
        print(f"   📊 Época {validator.epoch} - Resumen:")
        for k, v in avg_metrics.items():
            if 'pck' in k or 'oks' in k: # Mostrar solo las principales
                print(f"      • {k}: {v:.4f}")
        
        # Guardar CSV actualizado
        save_path = Path(validator.save_dir) / 'custom_metrics.csv'
        self.evaluator.save(str(save_path))
        
        # Limpiar para la próxima época
        self.batch_metrics = []
    
    def on_train_end(self, trainer):
        """Ejecutado al final del entrenamiento"""
        print("\n💾 Entrenamiento completado!")
        print(f"📂 Resultados en: {trainer.save_dir}")
