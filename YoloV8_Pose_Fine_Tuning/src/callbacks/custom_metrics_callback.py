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
        self.debug_done = False
        
    def on_val_start(self, validator):
        """
        Ejecutado al inicio de la validación.
        Aplicamos un 'Monkey Patch' para interceptar las predicciones.
        """
        self.debug_done = False # Reset debug flag for new validation epoch
        
        # Evitar aplicar el parche dos veces
        if hasattr(validator, 'is_patched_by_custom_metrics'):
            return

        # Guardamos la referencia al método original
        original_update_metrics = validator.update_metrics

        # Definimos nuestro método envoltorio (wrapper)
        def patched_update_metrics(preds, batch):
            # 1. Guardamos las predicciones en el validador para usarlas luego
            # Aseguramos que preds no sea None
            if preds is not None:
                validator.preds_batch = preds
                validator.batch = batch
            
            # 2. Llamamos al método original para que YOLO siga funcionando normal
            original_update_metrics(preds, batch)

        # Reemplazamos el método del objeto por el nuestro
        validator.update_metrics = patched_update_metrics
        validator.is_patched_by_custom_metrics = True
        
    def on_val_batch_end(self, validator):
        """
        Ejecutado al final de cada batch de validación.
        Calculamos métricas usando las predicciones interceptadas.
        """
        try:
            # Verificar si tenemos las predicciones interceptadas
            if not hasattr(validator, 'preds_batch') or validator.preds_batch is None:
                # Intentar recuperar desde atributos internos si el patch falló
                if hasattr(validator, 'preds'):
                    validator.preds_batch = validator.preds
                else:
                    return
            
            preds = validator.preds_batch
            batch = getattr(validator, 'batch', None)
            
            if batch is None:
                return

            # 1. Extraer Ground Truth y Batch Index
            if 'keypoints' not in batch:
                return
            
            # batch['keypoints'] -> [N_total_instances, K, 3]
            gt_kpts_all = batch['keypoints']
            
            if gt_kpts_all is None:
                return

            if isinstance(gt_kpts_all, torch.Tensor):
                gt_kpts_all = gt_kpts_all.detach().cpu().numpy()
                
            # batch['batch_idx'] -> [N_total_instances]
            if 'batch_idx' in batch:
                batch_idx = batch['batch_idx']
                if isinstance(batch_idx, torch.Tensor):
                    batch_idx = batch_idx.detach().cpu().numpy().flatten()
            else:
                return

            # 2. Alinear Predicciones con Ground Truth (Matching simple)
            aligned_preds = []
            aligned_gts = []
            
            # Si preds es una lista (formato usual)
            if isinstance(preds, list):
                for img_i, pred in enumerate(preds):
                    # --- A. Obtener GT para esta imagen ---
                    mask = (batch_idx == img_i)
                    gt_instances = gt_kpts_all[mask] # [M, K, 3]
                    
                    if len(gt_instances) == 0:
                        continue 
                    
                    # --- B. Obtener Predicción para esta imagen ---
                    if pred is None:
                        continue

                    pred_instances = None
                    
                    # Caso 1: Objeto Results de Ultralytics (v8.1+)
                    if hasattr(pred, 'keypoints') and pred.keypoints is not None:
                        kpts_data = pred.keypoints.data
                        if isinstance(kpts_data, torch.Tensor):
                            kpts_data = kpts_data.detach().cpu().numpy()
                        if len(kpts_data) > 0:
                            pred_instances = kpts_data

                    # Caso 2: Tensor o Numpy array
                    elif isinstance(pred, (torch.Tensor, np.ndarray)):
                        if isinstance(pred, torch.Tensor):
                            pred = pred.detach().cpu().numpy()
                        if len(pred) > 0:
                            try:
                                kpts_raw = pred[:, 6:]
                                n_dets = kpts_raw.shape[0]
                                k = gt_instances.shape[1]
                                if kpts_raw.shape[1] == k * 3:
                                    pred_instances = kpts_raw.reshape(n_dets, k, 3)
                            except: pass
                    
                    if pred_instances is None:
                        continue

                    # --- C. Matching (Emparejamiento) ---
                    best_pred = pred_instances[0] # [K, 3]
                    dists = np.sum(np.linalg.norm(gt_instances[:, :, :2] - best_pred[np.newaxis, :, :2], axis=2), axis=1)
                    best_gt_idx = np.argmin(dists)
                    best_gt = gt_instances[best_gt_idx] # [K, 3]
                    
                    aligned_preds.append(best_pred)
                    aligned_gts.append(best_gt)

            if not aligned_preds:
                return

            # Convertir a arrays
            pred_kpts_batch = np.stack(aligned_preds, axis=0) # [Batch_Matched, K, 3]
            gt_kpts_batch = np.stack(aligned_gts, axis=0)     # [Batch_Matched, K, 3]

            # Calcular bboxes a partir de GT keypoints para normalización PCK/OKS
            x_coords = gt_kpts_batch[..., 0]
            y_coords = gt_kpts_batch[..., 1]
            
            x1 = np.min(x_coords, axis=1)
            y1 = np.min(y_coords, axis=1)
            x2 = np.max(x_coords, axis=1)
            y2 = np.max(y_coords, axis=1)
            
            bboxes_batch = np.stack([x1, y1, x2, y2], axis=1) # [N, 4]

            # 3. Preparar diccionarios
            predictions = {
                'keypoints': pred_kpts_batch, 
                'bboxes': bboxes_batch 
            }
            
            ground_truth = {
                'keypoints': gt_kpts_batch[..., :2],
                'visibilities': gt_kpts_batch[..., 2]
            }
            
            # Calcular métricas
            metrics = self.evaluator.evaluate_batch(predictions, ground_truth)
            self.batch_metrics.append(metrics)
            
            # Limpiar
            validator.preds_batch = None
            
        except Exception as e:
            # Silenciar errores para no interrumpir entrenamiento
            pass

    def on_val_end(self, validator):
        """Ejecutado al final de cada validación"""
        if not self.batch_metrics:
            print("⚠️ No se calcularon métricas personalizadas (lista vacía).")
            return

        print("\n🔍 Calculando métricas personalizadas...")
        
        # Promediar métricas de todos los batches
        df_batch = pd.DataFrame(self.batch_metrics)
        avg_metrics = df_batch.mean().to_dict()
        
        # Obtener época actual de forma segura
        current_epoch = 0
        if hasattr(validator, 'trainer') and validator.trainer:
            current_epoch = validator.trainer.epoch
        elif hasattr(validator, 'epoch'):
            current_epoch = validator.epoch

        # Agregar resultado al evaluador
        self.evaluator.add_result(current_epoch, avg_metrics)
        
        # Mostrar resumen en consola
        print(f"   📊 Época {current_epoch} - Resumen:")
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
