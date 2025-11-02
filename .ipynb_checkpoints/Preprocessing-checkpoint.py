import numpy as np
import cv2
import glob
import os

input_folder = "Dataset_PDI/rgb/"
output_folder = "Dataset_PDI/output/"

image_files = sorted(glob.glob(os.path.join(input_folder, "*.png")))

for file in image_files:
    img = cv2.imread(file)
    if img is None:
        print(f"Could not load {file}")
        continue

    #float para mas precision
    img_float = img.astype(np.float32) / 255.0

    #subir valores del canal rojo
    img_float[..., 2] *= 1.5  #BGR
    img_float = np.clip(img_float, 0, 1)

    # Convertir a LAB y usar clahe
    lab = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    contrast_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Agudizar con gaussian blur
    blur = cv2.GaussianBlur(contrast_img, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(contrast_img, 1.5, blur, -0.5, 0)
  
    # Converitr a hsv para bajar saturacion
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 0.9)
    enhanced = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    #Guardar las imagenes
    filename = os.path.basename(file)
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, enhanced)
    print(f" Saved enhanced: {save_path}")
