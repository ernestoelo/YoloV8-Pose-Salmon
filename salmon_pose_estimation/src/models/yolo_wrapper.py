# src/models/yolo_wrapper.py
"""
Wrapper para YOLOv8 con descarga automática del modelo
"""
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
from ..utils.download_utils import ModelDownloader


class YOLOv8PoseTrainer:
    """Trainer modular para YOLOv8-Pose con descarga automática"""

    BASE_MODEL = 'yolov8s-pose.pt'

    def __init__(self, training_config_path: str):
        """
        Args:
            training_config_path: Ruta al archivo training_config.yaml
        """
        with open(training_config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model = None
        self.results = None

    @staticmethod
    def check_system_info() -> dict:
        """
        Verificar información del sistema

        Returns:
            dict con información de GPU/CPU
        """
        print("\n" + "="*70)
        print("🖥️  INFORMACIÓN DEL SISTEMA")
        print("="*70)

        cuda_available = torch.cuda.is_available()

        system_info = {
            'cuda_available': cuda_available,
            'pytorch_version': torch.__version__,
        }

        print(f"CUDA disponible: {cuda_available}")
        print(f"PyTorch: {torch.__version__}")

        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9

            system_info['gpu_name'] = gpu_name
            system_info['vram_gb'] = vram

            print(f"GPU: {gpu_name}")
            print(f"VRAM: {vram:.1f} GB")
            print("✅ GPU DETECTADA - Entrenamiento será rápido")
        else:
            print("⚠️  GPU NO DETECTADA")
            print("   Considere usar Google Colab para entrenamiento real")

        print("="*70)
        return system_info

    def load_model(self, model_path: str = None, auto_download: bool = True):
        """
        Carga el modelo con opción de descarga automática

        Args:
            model_path: Ruta al modelo (si None, usa predeterminado)
            auto_download: Si True, descarga automáticamente si no existe

        Returns:
            Modelo cargado
        """
        if model_path is None:
            model_path = self.BASE_MODEL

        # 1️⃣ Verificar sistema
        self.check_system_info()

        # 2️⃣ Descargar modelo si es necesario
        if auto_download:
            print("\n📦 GESTIÓN DEL MODELO BASE")
            print("="*70)
            ModelDownloader.download_model(model_path, verbose=True)

        # 3️⃣ Cargar modelo
        print(f"\n📦 Cargando modelo: {model_path}")
        print("-"*70)
        self.model = YOLO(model_path)
        self.model.info()

        return self.model

    def register_callbacks(self, callbacks_list: list):
        """Registrar callbacks para entrenamiento"""
        print("\n🔧 Registrando callbacks...")
        for callback_name, callback_func in callbacks_list:
            print(f"   ↳ {callback_name}")
            self.model.add_callback(callback_name, callback_func)

    def train(self):
        """Ejecutar entrenamiento con parámetros del magíster"""

        if self.model is None:
            print("❌ Error: Debe cargar el modelo primero (load_model())")
            return None

        print("\n" + "="*70)
        print("🔄 INICIANDO ENTRENAMIENTO")
        print("="*70 + "\n")

        self.results = self.model.train(
            # Paths
            data=self.config['paths']['data_yaml'],
            project=self.config['paths']['output_dir'],
            name="salmon_pose_v1",

            # Épocas
            epochs=self.config['training']['epochs'],
            patience=self.config['training']['patience'],

            # Tamaño
            imgsz=self.config['model']['input_size'],
            batch=self.config['training']['batch_size'],

            # Hardware
            device=self.config['training']['device'],
            workers=self.config['training']['workers'],

            # Loss weights del magíster ⭐
            box=self.config['training']['loss_weights']['box'],
            cls=self.config['training']['loss_weights']['cls'],
            dfl=self.config['training']['loss_weights']['dfl'],
            pose=self.config['training']['loss_weights']['pose'],
            kobj=self.config['training']['loss_weights']['kobj'],

            # Augmentations del magíster ⭐
            hsv_h=self.config['augmentation']['hsv_h'],
            hsv_s=self.config['augmentation']['hsv_s'],
            hsv_v=self.config['augmentation']['hsv_v'],
            degrees=self.config['augmentation']['degrees'],
            translate=self.config['augmentation']['translate'],
            scale=self.config['augmentation']['scale'],
            shear=self.config['augmentation']['shear'],
            perspective=self.config['augmentation']['perspective'],
            flipud=self.config['augmentation']['flipud'],
            fliplr=self.config['augmentation']['fliplr'],
            mosaic=self.config['augmentation']['mosaic'],

            # Config
            rect=False,
            cos_lr=False,
            dropout=0,
            optimizer="auto",
            exist_ok=False,
            verbose=True,
            save=True,
            plots=True
        )

        return self.results

    def get_results_dir(self):
        """Obtener directorio de resultados"""
        return self.results.save_dir if self.results else None
