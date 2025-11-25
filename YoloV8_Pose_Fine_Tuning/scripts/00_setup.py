# scripts/00_setup.py
"""
Script de configuración y preparación del dataset para el proyecto.

Funciones:
1.  Verifica el sistema (disponibilidad de GPU).
2.  Crea la estructura de directorios necesaria para el proyecto.
3.  Verifica la existencia de los archivos de configuración.
4.  Descarga el modelo base de YOLOv8 si no existe.
5.  (Opcional) Procesa un directorio de datos de origen, dividiéndolo en
    conjuntos de entrenamiento, validación y prueba, y generando el 
    archivo 'data.yaml' necesario para YOLO.
"""
import sys  # Importa el módulo sys para manipular el path y salir del script
import argparse  # Importa argparse para manejar argumentos de línea de comandos
from pathlib import Path  # Importa Path para manejo robusto de rutas de archivos
import shutil  # Importa shutil para operaciones de archivos como copiar
from sklearn.model_selection import train_test_split  # Importa función para dividir datasets
import yaml  # Importa yaml para leer y escribir archivos de configuración

# Agregar src al path para poder importar los módulos del proyecto
# Esto es necesario porque el script está en una subcarpeta 'scripts/'
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importamos nuestras utilidades personalizadas
from src.utils.download_utils import ModelDownloader  # Para descargar el modelo YOLO
from src.models.yolo_wrapper import YOLOv8PoseTrainer  # Para verificar GPU

# --- Funciones de Configuración del Entorno ---

def create_directories():
    """Crea la estructura de directorios estándar del proyecto."""
    print("\n📁 Creando estructura de directorios...")
    # Lista de directorios que necesitamos crear
    directories = [
        'outputs/runs',            # Donde YOLO guarda las corridas
        'outputs/checkpoints',     # Donde guardaremos pesos intermedios
        'outputs/metrics',         # Donde guardaremos CSVs de métricas
        'outputs/visualizations',  # Para guardar gráficos
        'data/images/train',       # Imágenes de entrenamiento
        'data/images/val',         # Imágenes de validación
        'data/images/test',        # Imágenes de prueba
        'data/labels/train',       # Etiquetas de entrenamiento
        'data/labels/val',         # Etiquetas de validación
        'data/labels/test'         # Etiquetas de prueba
    ]
    # Iteramos sobre la lista y creamos cada directorio si no existe
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("   ✅ Estructura de directorios creada/verificada.")

def verify_configs():
    """Verifica que los archivos de configuración principales existan."""
    print("\n📋 Verificando archivos de configuración...")
    # Archivos YAML esenciales para que el proyecto funcione
    required = ['config/training_config.yaml', 'config/keypoints_config.yaml']
    for config_file in required:
        # Si alguno no existe, detenemos el proceso
        if not Path(config_file).exists():
            print(f"   ❌ ERROR: El archivo de configuración '{config_file}' no fue encontrado.")
            return False
    print("   ✅ Archivos de configuración encontrados.")
    return True

# --- Funciones de Procesamiento del Dataset ---

def process_dataset(source_dir: Path, test_size: float):
    """
    Procesa el dataset de origen.
    Detecta si ya existe una estructura train/val o si es un directorio plano.
    """
    print(f"\n📦 Procesando dataset desde: '{source_dir}'")
    
    # Verificamos si el usuario ya nos dio las carpetas separadas
    has_train = (source_dir / 'train').exists()
    # Aceptamos 'validation' o 'val' como nombre
    has_val = (source_dir / 'validation').exists() or (source_dir / 'val').exists()
    
    if has_train and has_val:
        print("   ℹ️  Estructura 'train/validation' detectada. Usando splits existentes.")
        process_existing_splits(source_dir)
    else:
        print("   ℹ️  Estructura plana detectada. Realizando división automática.")
        process_flat_dataset(source_dir, test_size)

    # 3. Generar data.yaml
    # Este archivo le dice a YOLO dónde están las imágenes
    create_data_yaml(
        train_path='../data/images/train',
        val_path='../data/images/val',
        test_path='../data/images/test'
    )
    print("   - Archivo 'data/data.yaml' generado exitosamente.")
    return True

def process_existing_splits(source_dir: Path):
    """Procesa un dataset que ya viene dividido en carpetas."""
    # Determinar la carpeta de validación correcta (puede llamarse 'val' o 'validation')
    val_src = source_dir / 'validation' if (source_dir / 'validation').exists() else source_dir / 'val'
    
    # Mapeo de nombres estándar a rutas reales
    splits = {
        'train': source_dir / 'train',
        'val': val_src
    }
    
    for split_name, split_path in splits.items():
        print(f"   - Procesando split '{split_name}' desde '{split_path}'...")
        # Buscar imágenes recursivamente (para soportar subcarpetas como 'Train' o 'Validation')
        images = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            images.extend(list(split_path.rglob(ext)))
            
        print(f"     Encontradas {len(images)} imágenes en {split_name}.")
        # Copiamos las imágenes encontradas a nuestra estructura 'data/'
        copy_files(images, split_name)

def process_flat_dataset(source_dir: Path, test_size: float):
    """Procesa un dataset plano, dividiéndolo automáticamente."""
    image_files = []
    # Buscamos todas las imágenes en el directorio raíz
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(list(source_dir.glob(ext)))
    
    # Ordenamos para asegurar reproducibilidad
    image_files = sorted(image_files)
    if not image_files:
        print("   ❌ ERROR: No se encontraron imágenes (jpg/png/jpeg) en el directorio de origen.")
        return False

    print(f"   - {len(image_files)} imágenes encontradas.")

    # 1. División Train/Val/Test
    # Primero separamos Test del resto
    train_val_files, test_files = train_test_split(image_files, test_size=test_size, random_state=42)
    # Luego separamos Train de Validation (ajustando el porcentaje relativo)
    val_size_relative = test_size / (1 - test_size)
    train_files, val_files = train_test_split(train_val_files, test_size=val_size_relative, random_state=42)
    
    print(f"   - División de datos: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test.")

    # 2. Copiar archivos a la estructura de 'data/'
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')
    print("   - Archivos copiados a la estructura 'data/images' y 'data/labels'.")

def get_label_path(img_path: Path):
    """Busca el archivo de etiqueta correspondiente a una imagen."""
    # Estrategia 1: Buscar en la misma carpeta (ej: imagen.png y imagen.txt juntos)
    lbl = img_path.with_suffix('.txt')
    if lbl.exists(): return lbl
    
    # Estrategia 2: Reemplazar 'images' por 'labels' en el path
    # Maneja estructuras como: raw_data/train/images/Sub/img.png -> raw_data/train/labels/Sub/img.txt
    parts = list(img_path.parts)
    if 'images' in parts:
        # Encontrar la última ocurrencia de 'images' por si acaso
        idx = len(parts) - 1 - parts[::-1].index('images')
        parts[idx] = 'labels'
        lbl = Path(*parts).with_suffix('.txt')
        if lbl.exists(): return lbl
        
    return None

def copy_files(file_list: list[Path], split: str):
    """Copia imágenes y sus etiquetas a las carpetas de destino."""
    # Definimos destinos basados en el split (train, val, test)
    img_dest = Path(f'data/images/{split}')
    lbl_dest = Path(f'data/labels/{split}')
    
    for img_path in file_list:
        # Buscamos el archivo de texto asociado a la imagen
        lbl_path = get_label_path(img_path)
        
        # Si existe la etiqueta, copiamos ambos archivos
        if lbl_path and lbl_path.exists():
            shutil.copy(img_path, img_dest)
            shutil.copy(lbl_path, lbl_dest)
        else:
            # Opcional: Avisar si falta etiqueta (comentado para no saturar la consola)
            # print(f"⚠️ Aviso: No se encontró etiqueta para {img_path.name}")
            pass

def create_data_yaml(train_path: str, val_path: str, test_path: str):
    """Crea el archivo .yaml requerido por YOLOv8."""
    # Cargar los nombres de los keypoints desde el archivo de configuración
    with open('config/keypoints_config.yaml', 'r') as f:
        keypoints_data = yaml.safe_load(f)
    
    names = keypoints_data['keypoints']['names']
    nc = len(names) # Número de clases (en pose es 1, pero aquí se refiere a keypoints)
    kpt_shape = [nc, 3] # [número de keypoints, 3 (x, y, visibilidad)]

    # Estructura del diccionario que YOLO espera
    data = {
        'path': str(Path.cwd() / 'data'), # Ruta base absoluta
        'train': train_path,              # Ruta relativa a train
        'val': val_path,                  # Ruta relativa a val
        'test': test_path,                # Ruta relativa a test
        'nc': 1,                          # Siempre 1 para la detección de la clase "salmón"
        'names': ['salmon'],              # Nombre de la clase
        'kpt_shape': kpt_shape,           # Forma de los keypoints
        'flip_idx': []                    # No usamos flip horizontal, así que lista vacía
    }
    
    # Escribimos el archivo YAML
    with open('data/data.yaml', 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)

# --- Script Principal ---

def main(args):
    """Orquesta todo el proceso de setup."""
    print("\n" + "="*80)
    print("⚙️  INICIANDO SETUP DEL PROYECTO - SALMON POSE ESTIMATION")
    print("="*80)

    # 1. Setup del entorno
    # Verificamos si hay GPU disponible
    YOLOv8PoseTrainer.check_system_info()
    # Creamos las carpetas necesarias
    create_directories()
    # Verificamos que existan los configs
    if not verify_configs():
        return False
    
    # Intentamos descargar el modelo base
    try:
        ModelDownloader.download_model('yolov8s-pose.pt', verbose=True)
    except Exception as e:
        print(f"\n❌ Error descargando modelo: {e}")
        return False

    # 2. Procesamiento del dataset (si se especificó)
    if args.source_dir:
        source_path = Path(args.source_dir)
        # Validamos que la ruta de origen exista
        if not source_path.exists() or not source_path.is_dir():
            print(f"\n❌ ERROR: El directorio de origen '{args.source_dir}' no es válido.")
            return False
        # Ejecutamos el procesamiento
        if not process_dataset(source_path, args.test_size):
            return False
    else:
        print("\n🟡 AVISO: No se especificó un directorio de origen (`--source-dir`).")
        print("   El script solo configurará el entorno. El dataset no será procesado.")

    # 3. Resumen final
    print("\n" + "="*80)
    print("✅ SETUP COMPLETADO EXITOSAMENTE")
    print("="*80)
    print("\n🚀 Próximos pasos:")
    print("   1. Revisa que la carpeta 'data/' contenga tu dataset procesado.")
    print("   2. Ejecuta: python scripts/01_train.py")
    return True

if __name__ == '__main__':
    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Script de Setup para el proyecto de Pose Estimation.")
    parser.add_argument(
        '--source-dir',
        type=str,
        default=None,
        help='(Opcional) Ruta al directorio con las imágenes y etiquetas originales.'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proporción del dataset a reservar para el conjunto de prueba (ej. 0.2 para 20%).'
    )
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Ejecutar función principal y manejar errores
    try:
        success = main(args)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO durante el setup: {e}")
        sys.exit(1)
