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
        Aplicamos un 'Monkey Patch' para interceptar las predicciones.
        """
        # Evitar aplicar el parche dos veces
        if hasattr(validator, 'is_patched_by_custom_metrics'):
            return

        # Guardamos la referencia al método original
        original_update_metrics = validator.update_metrics
        print("🔧 CustomMetricsCallback: Monkey Patch aplicado a validator.update_metrics")

        # Definimos nuestro método envoltorio (wrapper)
        def patched_update_metrics(preds, batch):
            # 1. Guardamos las predicciones en el validador para usarlas luego
            # Aseguramos que preds no sea None
            if preds is not None:
                validator.preds_batch = preds
                validator.batch = batch
            else:
                print("⚠️ CustomMetricsCallback: preds es None en patched_update_metrics")
            
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
        # print(f"DEBUG: on_val_batch_end called. Validator: {id(validator)}")
        try:
            # Verificar si tenemos las predicciones interceptadas
            if not hasattr(validator, 'preds_batch') or validator.preds_batch is None:
                # Intentar recuperar desde atributos internos si el patch falló
                if hasattr(validator, 'preds'):
                    validator.preds_batch = validator.preds
                else:
                    print("⚠️ CustomMetricsCallback: No se encontraron predicciones (preds_batch/preds)")
                    print(f"   Validator keys: {validator.__dict__.keys()}")
                    return
            
            preds = validator.preds_batch
            batch = getattr(validator, 'batch', None)
            
            if batch is None:
                print("⚠️ CustomMetricsCallback: batch es None")
                return

            # 1. Extraer Ground Truth y Batch Index
            if 'keypoints' not in batch:
                print("⚠️ CustomMetricsCallback: 'keypoints' no está en batch")
                return
            
            # batch['keypoints'] -> [N_total_instances, K, 3]
            gt_kpts_all = batch['keypoints']
            
            if gt_kpts_all is None:
                return

            if isinstance(gt_kpts_all, torch.Tensor):
                gt_kpts_all = gt_kpts_all.detach().cpu().numpy()
            
            # --- FIX: Denormalize GT keypoints if they are normalized ---
            # Check if coordinates are normalized (<= 1.0)
            # Note: Visibility (index 2) is usually 0, 1, 2, so we check indices 0 and 1.
            if gt_kpts_all[..., :2].max() <= 1.0:
                # Try to get image size from batch['ori_shape'] or batch['resized_shape']
                # But batch usually contains 'img' which is the resized image tensor [B, 3, H, W]
                if 'img' in batch:
                    _, _, h, w = batch['img'].shape
                    gt_kpts_all[..., 0] *= w
                    gt_kpts_all[..., 1] *= h
                else:
                    # Fallback to 960 if img not found (based on config)
                    gt_kpts_all[..., 0] *= 960
                    gt_kpts_all[..., 1] *= 960
            # ------------------------------------------------------------
                
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
            
            # print(f"DEBUG: preds type: {type(preds)}")
            
            # Si preds es una lista (formato usual)
            if isinstance(preds, list):
                # print(f"DEBUG: preds list length: {len(preds)}")
                for img_i, pred in enumerate(preds):
                    # --- A. Obtener GT para esta imagen ---
                    mask = (batch_idx == img_i)
                    gt_instances = gt_kpts_all[mask] # [M, K, 3]
                    
                    if len(gt_instances) == 0:
                        # print(f"DEBUG: No GT for img {img_i}")
                        continue 
                    
                    # --- B. Obtener Predicción para esta imagen ---
                    if pred is None:
                        print(f"DEBUG: pred is None for img {img_i}")
                        continue

                    pred_instances = None
                    
                    # Caso 1: Objeto Results de Ultralytics (v8.1+)
                    if hasattr(pred, 'keypoints') and pred.keypoints is not None:
                        kpts_data = pred.keypoints.data
                        if isinstance(kpts_data, torch.Tensor):
                            kpts_data = kpts_data.detach().cpu().numpy()
                        if len(kpts_data) > 0:
                            pred_instances = kpts_data
                        else:
                             # print(f"DEBUG: pred.keypoints.data empty for img {img_i}")
                             pass

                    # Caso 1.5: Diccionario (Fix para este entorno)
                    elif isinstance(pred, dict) and 'keypoints' in pred:
                        kpts_data = pred['keypoints']
                        if isinstance(kpts_data, torch.Tensor):
                            kpts_data = kpts_data.detach().cpu().numpy()
                        
                        # Si tiene forma [N, K*3] o similar, asegurar [N, K, 3]
                        # En este caso parece que ya viene bien o como tensor
                        if len(kpts_data) > 0:
                            pred_instances = kpts_data
                        else:
                             # print(f"DEBUG: pred['keypoints'] empty for img {img_i}")
                             pass

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
                        # print(f"DEBUG: No pred_instances extracted for img {img_i}. Type: {type(pred)}")
                        continue

                    # --- C. Matching (Emparejamiento) ---
                    best_pred = pred_instances[0] # [K, 3]
                    dists = np.sum(np.linalg.norm(gt_instances[:, :, :2] - best_pred[np.newaxis, :, :2], axis=2), axis=1)
                    best_gt_idx = np.argmin(dists)
                    best_gt = gt_instances[best_gt_idx] # [K, 3]
                    
                    aligned_preds.append(best_pred)
                    aligned_gts.append(best_gt)

            if not aligned_preds:
                print("DEBUG: aligned_preds is empty")
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
            print(f"❌ Error en CustomMetricsCallback: {e}")
            import traceback
            traceback.print_exc()
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
