# scripts/01_train.py
"""
Script de entrenamiento principal
Integra: Descarga de modelo + Entrenamiento + Callbacks
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.yolo_wrapper import YOLOv8PoseTrainer
from src.callbacks.custom_metrics_callback import CustomMetricsCallback


def main():
    print("\n" + "="*80)
    print("🚀 ENTRENAMIENTO YOLOv8-POSE PARA SALMONES")
    print("   Sistema Modular con Descarga Automática")
    print("="*80)

    try:
        # 1️⃣ Crear trainer (carga config)
        print("\n1️⃣  Inicializando Trainer...")
        trainer = YOLOv8PoseTrainer('config/training_config.yaml')
        print("   ✅ Configuración cargada")

        # 2️⃣ Cargar modelo (descarga automática)
        print("\n2️⃣  Cargando Modelo...")
        model = trainer.load_model(
            model_path='yolov8s-pose.pt',
            auto_download=True  # ← DESCARGA AUTOMÁTICA
        )
        print("   ✅ Modelo listo")

        # 3️⃣ Registrar callbacks
        print("\n3️⃣  Registrando Callbacks...")
        callback = CustomMetricsCallback('config/keypoints_config.yaml')
        trainer.register_callbacks([
            ("on_val_end", callback.on_val_end),
            ("on_train_end", callback.on_train_end)
        ])
        print("   ✅ Callbacks registrados")

        # 4️⃣ Entrenar
        print("\n4️⃣  Iniciando Entrenamiento...")
        results = trainer.train()

        # 5️⃣ Mostrar resultados
        print("\n" + "="*80)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("="*80)
        print(f"\n📂 Resultados en: {trainer.get_results_dir()}")
        print("\n📊 Próximos pasos:")
        print("   1. Evaluar modelo: python scripts/02_evaluate.py")
        print("   2. Hacer inferencia: python scripts/03_inference.py")

    except Exception as e:
        print(f"\n❌ Error durante entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
