from icrawler.builtin import BingImageCrawler
import os

# --- Categorías y palabras clave ---
categorias = [
    ("vehiculo", ["car on fire", "vehicle fire", "truck on fire", "auto incendiado"]),
    ("casa", ["house on fire", "home fire", "building fire", "casa incendiada"]),
    ("cocina", ["kitchen fire", "cooking fire", "stove fire", "incendio cocina"]),
    ("forestal", ["forest fire", "wildfire", "bosque en llamas", "wildfire flames"]),
    ("industrial", ["industrial fire", "factory fire", "warehouse fire"]),
    ("urbano", ["urban fire", "city fire", "fuego urbano"]),
    ("rural", ["rural fire", "grass fire", "campo incendiado"]),
    ("natural", ["natural fire", "lightning fire", "fire caused by lightning"])
]

IMAGENES_POR_CATEGORIA = 300  # cantidad por categoría
DATASET_ROOT = r"C:\Datashet incendios\dataset_icrawler"

os.makedirs(DATASET_ROOT, exist_ok=True)

for cat_name, keywords in categorias:
    dest = os.path.join(DATASET_ROOT, cat_name)
    os.makedirs(dest, exist_ok=True)
    crawler = BingImageCrawler(storage={'root_dir': dest})
    per_kw = IMAGENES_POR_CATEGORIA // len(keywords)
    for kw in keywords:
        print(f"Descargando '{kw}' → {cat_name} (hasta {per_kw} imágenes)")
        crawler.crawl(keyword=kw, max_num=per_kw, file_idx_offset='auto')
    print(f"✅ Categoría {cat_name} completada.\n")

print("🔥 Descarga de todas las categorías completada.")
