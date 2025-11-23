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
        # 1️⃣ Cargar modelo entrenado
        print("\n1️⃣  Cargando Modelo...")
        if not Path(args.model).exists():
            print(f"❌ Modelo no encontrado: {args.model}")
            print("   Asegúrese de haber entrenado el modelo o especifique la ruta correcta con --model")
            return

        model = YOLO(args.model)
        print(f"   ✅ Modelo cargado: {args.model}")

        # 2️⃣ Cargar imagen
        print("\n2️⃣  Cargando Imagen...")
        # Verificar si la imagen existe antes de intentar leerla con cv2
        if not Path(args.image).exists():
             print(f"❌ Archivo de imagen no encontrado: {args.image}")
             return

        image = cv2.imread(args.image)

        if image is None:
            print(f"❌ No se pudo leer la imagen (formato inválido o corrupto): {args.image}")
            return

        print(f"   ✅ Imagen cargada: {args.image}")

        # 3️⃣ Inferencia
        print("\n3️⃣  Realizando Inferencia...")
        # plot=True en model() no retorna la imagen anotada directamente, 
        # hay que llamar a result.plot() después.
        results = model(image, conf=args.conf)

        # 4️⃣ Procesar resultados
        # results es una lista (un resultado por imagen de entrada)
        result = results[0] 
        
        if len(result.boxes) > 0:
            print(f"   ✅ Detecciones: {len(result.boxes)} salmón(es)")
            # Opcional: Mostrar confianza promedio
            confs = result.boxes.conf.cpu().numpy()
            print(f"      Confianza promedio: {confs.mean():.2f}")
        else:
            print(f"   ℹ️  No se encontraron salmones con confianza > {args.conf}")

        # 5️⃣ Guardar resultado
        print("\n4️⃣  Guardando Resultado...")
        # result.plot() genera la imagen con las cajas y keypoints dibujados
        annotated_image = result.plot()
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated_image)

        print("\n" + "="*80)
        print("✅ INFERENCIA COMPLETADA")
        print("="*80)
        print(f"\n💾 Resultado guardado en: {output_path.absolute()}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
