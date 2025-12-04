import os
from PIL import Image
import hashlib
import shutil
import random

# --- Configuración ---
DATASET_ROOT = r"C:\Datashet incendios\dataset_icrawler"
OUTPUT_ROOT = r"C:\Datashet incendios\dataset_final"
IMAGE_SIZE = (224, 224)  # tamaño final
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# --- Funciones ---
def is_image_valid(path):
    try:
        with Image.open(path) as img:
            img.verify()  # verifica si está corrupta
        return True
    except:
        return False

def hash_file(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# --- Procesar cada categoría ---
for categoria in os.listdir(DATASET_ROOT):
    cat_path = os.path.join(DATASET_ROOT, categoria)
    if not os.path.isdir(cat_path):
        continue

    print(f"\nProcesando categoría: {categoria}")
    images = [os.path.join(cat_path, f) for f in os.listdir(cat_path)]
    
    # Filtrar imágenes corruptas
    images = [img for img in images if is_image_valid(img)]
    print(f"Imágenes válidas: {len(images)}")

    # Eliminar duplicados
    hashes = set()
    unique_images = []
    for img in images:
        h = hash_file(img)
        if h not in hashes:
            hashes.add(h)
            unique_images.append(img)
    print(f"Imágenes únicas: {len(unique_images)}")

    # Mezclar
    random.shuffle(unique_images)

    # Dividir en train/val/test
    n = len(unique_images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    splits = {
        'train': unique_images[:n_train],
        'val': unique_images[n_train:n_train+n_val],
        'test': unique_images[n_train+n_val:]
    }

    # Crear carpetas y redimensionar/copiar
    for split, imgs in splits.items():
        split_dir = os.path.join(OUTPUT_ROOT, split, categoria)
        os.makedirs(split_dir, exist_ok=True)
        for img_path in imgs:
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                    im = im.resize(IMAGE_SIZE)
                    basename = os.path.basename(img_path)
                    im.save(os.path.join(split_dir, basename))
            except:
                pass
    print(f"Categoría {categoria} procesada: Train={n_train}, Val={n_val}, Test={n_test}")

print("\n✅ Dataset final listo en 'dataset_final' con estructura:")
print("dataset_final/train/<categoria>/")
print("dataset_final/val/<categoria>/")
print("dataset_final/test/<categoria>/")
