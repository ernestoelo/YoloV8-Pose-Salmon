# scripts/00_setup.py
"""
Setup inicial del proyecto - Ejecutar UNA SOLA VEZ
Descarga modelos, crea directorios y verifica configuración
"""
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.download_utils import ModelDownloader
from src.models.yolo_wrapper import YOLOv8PoseTrainer
import torch
import yaml


def create_directories():
    """Crear estructura de directorios"""
    print("\n📁 Creando estructura de directorios...")

    directories = [
        'outputs/runs',
        'outputs/checkpoints',
        'outputs/metrics',
        'outputs/visualizations',
        'data/raw',
        'data/processed/images/train',
        'data/processed/images/val',
        'data/processed/images/test',
        'data/processed/labels/train',
        'data/processed/labels/val',
        'data/processed/labels/test',
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}/")


def verify_configs():
    """Verificar archivos de configuración"""
    print("\n📋 Verificando configuraciones...")

    required_configs = [
        'config/training_config.yaml',
        'config/keypoints_config.yaml'
    ]

    all_exist = True
    for config_file in required_configs:
        if Path(config_file).exists():
            print(f"   ✅ {config_file}")
        else:
            print(f"   ❌ {config_file} NO ENCONTRADO")
            all_exist = False

    if not all_exist:
        print("\n⚠️  ADVERTENCIA: Faltan archivos de configuración")
        print("   Cree los archivos .yaml antes de entrenar")

    return all_exist


def setup_environment():
    """Ejecutar setup completo"""

    print("\n" + "="*80)
    print("⚙️  SETUP INICIAL - SALMON POSE ESTIMATION")
    print("="*80)

    # 1️⃣ Verificar sistema
    print("\n1️⃣  Verificando Sistema...")
    system_info = YOLOv8PoseTrainer.check_system_info()

    # 2️⃣ Crear directorios
    print("\n2️⃣  Creando Directorios...")
    create_directories()

    # 3️⃣ Verificar configuraciones
    print("\n3️⃣  Verificando Configuraciones...")
    configs_ok = verify_configs()

    # 4️⃣ Descargar modelo
    print("\n4️⃣  Descargando Modelo Base...")
    try:
        model_path = ModelDownloader.download_model(
            'yolov8s-pose.pt',
            verbose=True
        )
        model_ok = True
    except Exception as e:
        print(f"\n❌ Error descargando modelo: {e}")
        model_ok = False

    # 5️⃣ Resumen final
    print("\n" + "="*80)
    print("📊 RESUMEN DEL SETUP")
    print("="*80)

    print(f"\nGPU: {'✅ Disponible' if system_info['cuda_available'] else '⚠️  No disponible'}")
    print(f"Directorios: ✅ Creados")
    print(f"Configuración: {'✅ OK' if configs_ok else '❌ Incompleta'}")
    print(f"Modelo: {'✅ Listo' if model_ok else '❌ Error'}")

    if configs_ok and model_ok:
        print("\n" + "="*80)
        print("✅ SETUP COMPLETADO EXITOSAMENTE")
        print("="*80)
        print("\n🚀 Próximos pasos:")
        print("   1. Copiar dataset a data/processed/")
        print("   2. Ejecutar: python scripts/01_train.py")
        print("   3. Monitorear entrenamiento")
        return True
    else:
        print("\n⚠️  Corrija los problemas antes de continuar")
        return False


if __name__ == '__main__':
    try:
        success = setup_environment()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error durante setup: {e}")
        sys.exit(1)
