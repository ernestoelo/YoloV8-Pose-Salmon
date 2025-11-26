# scripts/02_evaluate.py
"""
Script de evaluación robusto para YOLOv8-Pose.
Calcula métricas estándar (mAP) y personalizadas (PCK, OKS) iterando sobre el dataset de validación.
"""
import sys
import yaml
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from src.metrics.evaluator import PoseEvaluator

def load_ground_truth(label_path, img_shape):
    """
    Carga las etiquetas de un archivo .txt de YOLO y las desnormaliza.
    Formato YOLO Pose: class x_center y_center width height px1 py1 pvis1 px2 py2 pvis2 ...
    """
    if not label_path.exists():
        return None, None

    h, w = img_shape[:2]
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    bboxes = []
    keypoints = []
    
    for line in lines:
        data = list(map(float, line.strip().split()))
        
        # Bounding Box (x_center, y_center, width, height) -> (x1, y1, x2, y2)
        xc, yc, bw, bh = data[1:5]
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h
        bboxes.append([x1, y1, x2, y2])
        
        # Keypoints (px, py, pvis)
        kpts_raw = data[5:]
        kpts = []
        for i in range(0, len(kpts_raw), 3):
            px, py, pvis = kpts_raw[i:i+3]
            kpts.append([px * w, py * h, pvis]) # Desnormalizar y guardar visibilidad
        keypoints.append(kpts)
        
    return np.array(bboxes), np.array(keypoints)

def main():
    print("\n" + "="*80)
    print("📊 EVALUACIÓN DETALLADA DE MODELO YOLOv8-POSE")
    print("="*80)

    try:
        # 1️⃣ Cargar configuración
        print("\n1️⃣  Cargando Configuración...")
        with open('config/training_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Combinar configs para el evaluador
        full_config = config.copy()
        with open('config/keypoints_config.yaml', 'r') as f:
            full_config.update(yaml.safe_load(f))

        # 2️⃣ Buscar modelo entrenado
        print("\n2️⃣  Buscando Modelo Entrenado...")
        project_dir = Path(config['paths']['output_dir'])
        
        if project_dir.exists():
            run_dirs = [d for d in project_dir.iterdir() if d.is_dir() and d.name.startswith('salmon_pose_v')]
            if run_dirs:
                latest_run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
                best_model_path = latest_run_dir / 'weights/best.pt'
                print(f"   ℹ️  Directorio de run detectado: {latest_run_dir.name}")
            else:
                best_model_path = project_dir / 'salmon_pose_v1/weights/best.pt'
        else:
            best_model_path = project_dir / 'salmon_pose_v1/weights/best.pt'

        if not best_model_path.exists():
            print(f"❌ Modelo no encontrado: {best_model_path}")
            return

        # 3️⃣ Cargar modelo y evaluador
        print(f"   ✅ Cargando modelo desde: {best_model_path}")
        model = YOLO(best_model_path)
        evaluator = PoseEvaluator(full_config)

        # 4️⃣ Preparar dataset de validación
        print("\n3️⃣  Preparando Dataset de Validación...")
        data_yaml_path = Path(config['paths']['data_yaml'])
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            
        # Asumimos estructura estándar de YOLO: data/images/val y data/labels/val
        base_path = data_yaml_path.parent
        val_images_dir = base_path / data_config.get('val', 'images/val')
        # Si la ruta en yaml es relativa a data.yaml, ajustamos. Si es 'images/val', buscamos labels en 'labels/val'
        if not val_images_dir.exists():
             # Intento de corrección de ruta común
             val_images_dir = Path('data/images/val')
        
        val_labels_dir = Path(str(val_images_dir).replace('images', 'labels'))
        
        image_files = sorted(list(val_images_dir.glob('*.png')) + list(val_images_dir.glob('*.jpg')))
        print(f"   ℹ️  Imágenes encontradas: {len(image_files)}")

        if len(image_files) == 0:
            print("❌ No se encontraron imágenes de validación.")
            return

        # 5️⃣ Ejecutar Inferencia y Evaluación
        print("\n4️⃣  Ejecutando Inferencia y Cálculo de Métricas...")
        
        all_metrics = []
        
        for img_path in tqdm(image_files, desc="Evaluando"):
            # A. Cargar imagen y GT
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            label_path = val_labels_dir / img_path.with_suffix('.txt').name
            gt_bboxes, gt_keypoints = load_ground_truth(label_path, img.shape)
            
            if gt_bboxes is None or len(gt_bboxes) == 0:
                continue # Saltar imágenes sin anotaciones
                
            # B. Inferencia
            results = model(img, verbose=False, conf=0.25) # Confianza mínima razonable
            result = results[0]
            
            if result.keypoints is None or len(result.keypoints) == 0:
                # Si no hay detecciones pero había GT, cuenta como fallo (recall baja)
                # Para simplificar, pasamos predicciones vacías al evaluador
                pred_kpts = np.zeros((0, gt_keypoints.shape[1], 3))
                pred_bboxes = np.zeros((0, 4))
            else:
                pred_kpts = result.keypoints.data.cpu().numpy() # [N, K, 3]
                pred_bboxes = result.boxes.xyxy.cpu().numpy()   # [N, 4]

            # C. Emparejamiento (Matching) Simple
            # Necesitamos alinear Predicciones con GT para calcular PCK/OKS
            # Estrategia: Para cada GT, buscar la predicción más cercana (por centro de bbox)
            
            aligned_preds_kpts = []
            aligned_gt_kpts = []
            aligned_gt_vis = []
            aligned_bboxes = [] # Usamos bbox de GT para normalizar PCK
            
            # Centros de GT
            gt_centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
            
            if len(pred_bboxes) > 0:
                pred_centers = (pred_bboxes[:, :2] + pred_bboxes[:, 2:]) / 2
                
                # Matriz de distancias
                dists = np.linalg.norm(gt_centers[:, None] - pred_centers[None, :], axis=2)
                
                # Asignación voraz (Greedy)
                for i in range(len(gt_bboxes)):
                    best_match_idx = np.argmin(dists[i])
                    min_dist = dists[i, best_match_idx]
                    
                    # Umbral de distancia para considerar match (ej. 10% de la diagonal de la imagen)
                    diag = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
                    if min_dist < 0.1 * diag:
                        aligned_preds_kpts.append(pred_kpts[best_match_idx])
                        aligned_gt_kpts.append(gt_keypoints[i, :, :2]) # Solo x,y
                        aligned_gt_vis.append(gt_keypoints[i, :, 2])   # Visibilidad
                        aligned_bboxes.append(gt_bboxes[i])
            
            if not aligned_preds_kpts:
                continue

            # D. Calcular métricas del batch (imagen)
            batch_preds = {
                'keypoints': np.array(aligned_preds_kpts),
                'bboxes': np.array(aligned_bboxes)
            }
            batch_gt = {
                'keypoints': np.array(aligned_gt_kpts),
                'visibilities': np.array(aligned_gt_vis)
            }
            
            metrics = evaluator.evaluate_batch(batch_preds, batch_gt)
            all_metrics.append(metrics)

        # 6️⃣ Resumen Final
        if not all_metrics:
            print("❌ No se pudieron calcular métricas (posiblemente sin coincidencias Pred-GT).")
            return

        print("\n" + "="*80)
        print("✅ RESULTADOS FINALES (Promedio sobre dataset de validación)")
        print("="*80)
        
        df = pd.DataFrame(all_metrics)
        mean_metrics = df.mean()
        
        # Imprimir bonito
        print(f"\n🔹 Métricas Globales:")
        print(f"   • OKS Mean:       {mean_metrics['oks_mean']:.4f}")
        print(f"   • PCK@0.05 (Estricto): {mean_metrics.get('pck@0.05', 0):.4f}")
        print(f"   • PCK@0.10 (Medio):    {mean_metrics.get('pck@0.1', 0):.4f}")
        print(f"   • PCK@0.20 (Laxo):     {mean_metrics.get('pck@0.2', 0):.4f}")
        
        print(f"\n🔹 PCK@0.1 por Keypoint (Precisión por parte del cuerpo):")
        for k in mean_metrics.keys():
            if k.startswith('pck@0.1_'):
                part_name = k.replace('pck@0.1_', '')
                print(f"   • {part_name:<15}: {mean_metrics[k]:.4f}")

        # Guardar
        save_path = latest_run_dir / 'final_evaluation_metrics.csv'
        mean_metrics.to_csv(save_path)
        print(f"\n💾 Reporte detallado guardado en: {save_path}")

    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
