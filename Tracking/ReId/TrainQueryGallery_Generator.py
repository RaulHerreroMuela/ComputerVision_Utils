from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp
import random

dataset_dir = '.../Escritorio/ReId/ReIdDataSetTODASIMAGENES'

all_files = []
for root, dirs, files in os.walk(dataset_dir):
    all_files.extend([osp.join(root, file) for file in files])

train = []
query = []
gallery = []

pid_train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 27]
pid_GalleryQuery = [19, 20, 21, 22]

camId_gallery = [3, 4, 5, 6, 7, 9, 10]
camId_query = [1, 2, 8]

for img_path in all_files:

    partes = img_path.split('_')
    pid = int(partes[1].replace('id', ''))
    camId = int(partes[3].replace(('.jpg'), ''))


    if pid in pid_train:
        train.append((img_path, pid, camId))

    if pid in pid_GalleryQuery:
        if camId in camId_query:
            query.append((img_path, pid, camId))

        else:
            gallery.append((img_path, pid, camId))


query_size = int(0.02 * len(gallery))  # Se define el tamaño de query

random.shuffle(query)
query_def = query[0:query_size]

ruta_train = '.../Escritorio/ReId/dS_Train.txt'
ruta_query = '.../Escritorio/ReId/dS_Query.txt'
ruta_gallery = '.../Escritorio/ReId/dS_Gallery.txt'

with open(ruta_train, 'w') as archivo:
    for t_elemento in train:
        linea = ','.join(map(str, t_elemento))
        archivo.write(linea + '\n')

with open(ruta_query, 'w') as archivo:
    for q_elemento in query_def:
        linea = ','.join(map(str, q_elemento))
        archivo.write(linea + '\n')

with open(ruta_gallery, 'w') as archivo:
    for g_elemento in gallery:
        linea = ','.join(map(str, g_elemento))
        archivo.write(linea + '\n')