# src/utils/download_utils.py
"""
Utilidades para descargar y gestionar modelos YOLO
"""
from pathlib import Path
from ultralytics import YOLO
import torch


class ModelDownloader:
    """Gestor de descargas de modelos YOLO"""

    MODELS_DIR = Path.home() / '.yolo' / 'weights'

    AVAILABLE_MODELS = {
        'yolov8s-pose.pt': {
            'size': 21.4,
            'description': 'YOLOv8 Small - Pose Estimation (RECOMENDADO)',
            'params': 21.4e6,
            'inference_speed': '32ms',
            'recommended': True
        },
        'yolov8m-pose.pt': {
            'size': 49.7,
            'description': 'YOLOv8 Medium - Pose Estimation',
            'params': 53.1e6,
            'inference_speed': '66ms',
            'recommended': False
        },
        'yolov8l-pose.pt': {
            'size': 99.2,
            'description': 'YOLOv8 Large - Pose Estimation',
            'params': 108.2e6,
            'inference_speed': '139ms',
            'recommended': False
        }
    }

    @staticmethod
    def get_model_info(model_name: str) -> dict:
        """Obtener información del modelo"""
        return ModelDownloader.AVAILABLE_MODELS.get(
            model_name,
            {'size': 'unknown', 'description': 'Modelo no encontrado'}
        )

    @staticmethod
    def check_disk_space(required_mb: float = 100) -> bool:
        """Verificar espacio disponible en disco"""
        import shutil

        stat = shutil.disk_usage(ModelDownloader.MODELS_DIR.parent)
        available_mb = stat.free / 1e6

        if available_mb < required_mb:
            print(f"⚠️  Advertencia: Solo {available_mb:.0f} MB disponibles")
            print(f"   Se requieren ~{required_mb} MB")
            return False

        return True

    @staticmethod
    def model_exists(model_name: str = 'yolov8s-pose.pt') -> bool:
        """Verificar si el modelo ya está descargado"""
        model_path = ModelDownloader.MODELS_DIR / model_name

        if model_path.exists():
            size_mb = model_path.stat().st_size / 1e6
            return True, size_mb, model_path

        return False, 0, model_path

    @staticmethod
    def download_model(model_name: str = 'yolov8s-pose.pt',
                      verbose: bool = True) -> Path:
        """
        Descargar modelo YOLO (con caché automático)

        Args:
            model_name: Nombre del modelo
            verbose: Mostrar información

        Returns:
            Path al modelo

        Raises:
            RuntimeError: Si falla la descarga
        """
        if verbose:
            print("\n" + "="*70)
            print("📦 GESTOR DE DESCARGAS DE MODELOS YOLO")
            print("="*70)

        # Obtener información
        info = ModelDownloader.get_model_info(model_name)

        if verbose:
            print(f"\nModelo: {model_name}")
            print(f"Descripción: {info.get('description', 'N/A')}")
            print(f"Tamaño: {info.get('size', 'N/A')} MB")
            print(f"Parámetros: {info.get('params', 'N/A') / 1e6:.1f}M")

        # Crear directorio
        ModelDownloader.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        model_path = ModelDownloader.MODELS_DIR / model_name

        # Verificar si ya existe
        if model_path.exists():
            if verbose:
                size_mb = model_path.stat().st_size / 1e6
                print(f"\n✅ Modelo ya en caché")
                print(f"   {model_path}")
                print(f"   Tamaño: {size_mb:.1f} MB")
            return model_path

        # Verificar espacio
        size = info.get('size', 50)
        if not ModelDownloader.check_disk_space(size + 50):
            raise RuntimeError("Espacio en disco insuficiente")

        # Descargar
        if verbose:
            print(f"\n⏳ Descargando modelo...")
            print(f"   Tamaño: {size} MB")
            print(f"   Destino: {ModelDownloader.MODELS_DIR}")
            print(f"   Tiempo estimado: 30-120 segundos")

        try:
            # Ultralytics descarga automáticamente
            YOLO(model_name)

            # Mover al directorio de caché si se descargó en el directorio actual
            local_file = Path(model_name)
            if local_file.exists() and local_file != model_path:
                import shutil
                shutil.move(str(local_file), str(model_path))

            if verbose:
                size_mb = model_path.stat().st_size / 1e6
                print(f"\n✅ Descarga completada!")
                print(f"   {model_path}")
                print(f"   Tamaño: {size_mb:.1f} MB")

            return model_path

        except Exception as e:
            print(f"\n❌ Error descargando modelo:")
            print(f"   {e}")
            raise

    @staticmethod
    def list_available_models() -> None:
        """Listar todos los modelos disponibles"""
        print("\n📋 Modelos YOLO Pose disponibles:\n")

        for name, info in ModelDownloader.AVAILABLE_MODELS.items():
            tag = " ⭐ RECOMENDADO" if info.get('recommended') else ""
            print(f"  {name}{tag}")
            print(f"     └─ {info['description']}")
            print(f"        Tamaño: {info['size']} MB | Params: {info['params']/1e6:.1f}M | {info['inference_speed']}")
            print()
