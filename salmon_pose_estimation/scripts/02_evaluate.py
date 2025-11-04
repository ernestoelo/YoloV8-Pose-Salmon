# scripts/02_evaluate.py
"""
Script de evaluación con descarga automática del modelo
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from src.utils.download_utils import ModelDownloader
from src.metrics.evaluator import PoseEvaluator
import yaml


def main():
    print("\n" + "="*80)
    print("📊 EVALUACIÓN DE MODELO YOLOv8-POSE")
    print("="*80)

    try:
        # 1️⃣ Descargar modelo si es necesario
        print("\n1️⃣  Verificando Modelo Base...")
        model_path = ModelDownloader.download_model(
            'yolov8s-pose.pt',
            verbose=True
        )

        # 2️⃣ Buscar modelo entrenado
        print("\n2️⃣  Buscando Modelo Entrenado...")
        best_model_path = 'outputs/runs/salmon_pose_v1/weights/best.pt'

        if not Path(best_model_path).exists():
            print(f"❌ Modelo entrenado no encontrado: {best_model_path}")
            print("   Ejecute primero: python scripts/01_train.py")
            return

        print(f"   ✅ Modelo encontrado: {best_model_path}")

        # 3️⃣ Cargar modelo
        print("\n3️⃣  Cargando Modelo Entrenado...")
        model = YOLO(best_model_path)

        # 4️⃣ Cargar config
        print("\n4️⃣  Cargando Configuración...")
        with open('config/training_config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        with open('config/keypoints_config.yaml', 'r') as f:
            kpt_config = yaml.safe_load(f)

        # 5️⃣ Ejecutar validación
        print("\n5️⃣  Ejecutando Validación...")
        metrics = model.val(
            data=config['paths']['data_yaml'],
            split='test',
            batch=16,
            imgsz=config['model']['input_size'],
            conf=config['validation']['conf_threshold'],
            iou=config['validation']['iou_threshold'],
            save_json=True,
            plots=True
        )

        print("\n" + "="*80)
        print("✅ EVALUACIÓN COMPLETADA")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error durante evaluación: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
