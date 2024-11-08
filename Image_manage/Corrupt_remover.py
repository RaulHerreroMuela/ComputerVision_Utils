import shutil
from tqdm import tqdm
import os
import subprocess
from pathlib import Path
folder_path = "/home/images/VelillaEntrada/dataset_anomalias_revisado"
if not os.path.isdir(os.path.join(folder_path, "corruptas")):
    os.mkdir(os.path.join(folder_path, "corruptas"))

cont = 0

img_list = [str(img) for img in Path(folder_path).rglob('*.[jJ][pP][gG]')]
print(img_list)
for filename in tqdm(img_list):
    try:
        output = subprocess.check_output(["jpeginfo", "-c", filename], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        # Archivo JPEG corrupto
        cont += 1
        print(f"Error al cargar imagen {filename}: {e.output.decode('utf-8')}")
        shutil.move(filename, os.path.join(folder_path, "corruptas/"))

print(f"Se han movido a la carpeta malas {cont} imagenes corruptas")