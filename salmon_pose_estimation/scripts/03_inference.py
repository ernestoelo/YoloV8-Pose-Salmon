# scripts/03_inference.py
"""
Script de inferencia en nuevas imágenes
"""
import argparse
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from src.utils.download_utils import ModelDownloader


def main():
    parser = argparse.ArgumentParser(description='Inferencia YOLOv8-Pose')
    parser.add_argument('--image', type=str, help='Ruta a la imagen')
    parser.add_argument('--model', type=str, 
                       default='outputs/runs/salmon_pose_v1/weights/best.pt',
                       help='Ruta al modelo entrenado')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confianza mínima')
    parser.add_argument('--output', type=str, default='outputs/inference_result.jpg',
                       help='Ruta para guardar resultado')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🎯 INFERENCIA YOLOv8-POSE")
    print("="*80)

    try:
        # 1️⃣ Descargar modelo base si es necesario
        print("\n1️⃣  Verificando Modelo Base...")
        ModelDownloader.download_model('yolov8s-pose.pt', verbose=False)

        # 2️⃣ Cargar modelo entrenado
        print("\n2️⃣  Cargando Modelo...")
        if not Path(args.model).exists():
            print(f"❌ Modelo no encontrado: {args.model}")
            return

        model = YOLO(args.model)
        print(f"   ✅ Modelo cargado: {args.model}")

        # 3️⃣ Cargar imagen
        print("\n3️⃣  Cargando Imagen...")
        image = cv2.imread(args.image)

        if image is None:
            print(f"❌ No se pudo cargar: {args.image}")
            return

        print(f"   ✅ Imagen cargada: {args.image}")

        # 4️⃣ Inferencia
        print("\n4️⃣  Realizando Inferencia...")
        results = model(image, conf=args.conf)

        # 5️⃣ Procesar resultados
        for result in results:
            if len(result.boxes) > 0:
                print(f"   ✅ Detecciones: {len(result.boxes)}")
            else:
                print(f"   ℹ️  No se encontraron salmones")

        # 6️⃣ Guardar resultado
        print("\n5️⃣  Guardando Resultado...")
        annotated_image = results.plot()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated_image)

        print("\n" + "="*80)
        print("✅ INFERENCIA COMPLETADA")
        print("="*80)
        print(f"\n💾 Resultado: {output_path}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
