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
        
    def on_val_start(self, validator):
        """
        Ejecutado al inicio de la validación.
        Aplicamos un 'Monkey Patch' para interceptar las predicciones,
        ya que YOLOv8.1+ no las expone públicamente en el objeto validator.
        """
        # Evitar aplicar el parche dos veces
        if hasattr(validator, 'is_patched_by_custom_metrics'):
            return

        # Guardamos la referencia al método original
        original_update_metrics = validator.update_metrics

        # Definimos nuestro método envoltorio (wrapper)
        def patched_update_metrics(preds, batch):
            # 1. Guardamos las predicciones en el validador para usarlas luego
            validator.preds_batch = preds
            # 2. Llamamos al método original para que YOLO siga funcionando normal
            original_update_metrics(preds, batch)

        # Reemplazamos el método del objeto por el nuestro
        validator.update_metrics = patched_update_metrics
        validator.is_patched_by_custom_metrics = True
        # print("🔧 Monkey Patch aplicado exitosamente a validator.update_metrics")

    def on_val_batch_end(self, validator):
        """
        Ejecutado al final de cada batch de validación.
        Calculamos métricas usando las predicciones interceptadas.
        """
        try:
            # Verificar si tenemos las predicciones interceptadas
            if not hasattr(validator, 'preds_batch') or validator.preds_batch is None:
                return
            
            preds = validator.preds_batch
            batch = validator.batch
            
            if batch is None:
                return

            # 1. Extraer Ground Truth
            # batch['keypoints'] -> [Batch, K, 3]
            gt_kpts_batch = batch['keypoints'].detach().cpu().numpy()
            
            # 2. Procesar Predicciones
            # preds suele ser una lista de Tensores (uno por imagen) o un Tensor batch
            # En YOLOv8 Pose, suele ser una lista de resultados post-procesados
            
            pred_kpts_list = []
            
            # Iteramos sobre las predicciones (preds es una lista de tensores [N_det, 3] o similar)
            # Nota: preds viene de postprocess(), así que ya son coordenadas finales
            
            # Si preds es un Tensor directo (a veces pasa en validación simple)
            if isinstance(preds, torch.Tensor):
                # Lógica para tensor batch... (menos común en val)
                pass
            
            # Asumimos que es lista de predicciones por imagen (lo estándar)
            for i, pred in enumerate(preds):
                # pred es un tensor de detecciones [N, 56] (cajas + kpts + conf)
                # Ojo: postprocess devuelve [N, 6 + K*3]
                
                # Si no hay detecciones
                if pred is None or len(pred) == 0:
                    pred_kpts_list.append(np.zeros((1, gt_kpts_batch.shape[1], 3)))
                    continue

                # Extraer keypoints. 
                # En YOLOv8-Pose, los keypoints suelen estar a partir del índice 6
                # Estructura: [x, y, w, h, conf, cls, kpt1_x, kpt1_y, kpt1_conf, ...]
                
                # Sin embargo, postprocess() devuelve una lista de tensores.
                # Vamos a usar la lógica de parsing estándar de YOLO si es posible,
                # pero como es crudo, mejor intentamos inferir.
                
                # Si pred tiene forma [N, 6 + K*3]
                # K = (pred.shape[1] - 6) // 3
                
                kpts_raw = pred[:, 6:].detach().cpu().numpy()
                
                # Reshape a [N, K, 3]
                n_dets = kpts_raw.shape[0]
                k = gt_kpts_batch.shape[1]
                
                # Verificar dimensiones
                if kpts_raw.shape[1] != k * 3:
                    # Fallback o error silencioso
                    pred_kpts_list.append(np.zeros((1, k, 3)))
                    continue
                    
                kpts = kpts_raw.reshape(n_dets, k, 3)
                
                # Tomamos la detección con mayor confianza (la primera, si están ordenadas)
                # O idealmente evaluamos todas, pero para simplificar tomamos la mejor
                best_kpt = kpts[0:1] # [1, K, 3]
                pred_kpts_list.append(best_kpt)

            if not pred_kpts_list:
                return

            pred_kpts_batch = np.concatenate(pred_kpts_list, axis=0)

            # 3. Preparar diccionarios
            predictions = {
                'keypoints': pred_kpts_batch, 
                'bboxes': None # No necesitamos bboxes para OKS/PCK puro
            }
            
            ground_truth = {
                'keypoints': gt_kpts_batch[..., :2],
                'visibilities': gt_kpts_batch[..., 2]
            }
            
            # Calcular métricas
            metrics = self.evaluator.evaluate_batch(predictions, ground_truth)
            self.batch_metrics.append(metrics)
            
            # Limpiar para ahorrar memoria
            validator.preds_batch = None
            
        except Exception as e:
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
