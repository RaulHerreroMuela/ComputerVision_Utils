from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import random
import torchreid

from torchreid.data import ImageDataset


class NewDataset(ImageDataset):
    dataset_dir = '.../ReId/ReId_DataSet'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).

        train, query, gallery = self.generar_dataSets(self.dataset_dir, range(0, 15))
        super(NewDataset, self).__init__(train, query, gallery, **kwargs)

    def get_all_files(self, dir_path):
        all_files = []
        for root, dirs, files in os.walk(dir_path):
            all_files.extend([osp.join(root, file) for file in files])
        return all_files

    def generar_dataSets(self, dir_path, pid_train_range):
        all_image_paths = self.get_all_files(dir_path)
        train = []
        query = []  # 2% de los datos de los ids no contenidos en el rango de train
        gallery = []  # 98% de los datos de los ids no contenidos en el rango de train
        prefijo_a_quitar = '.../Escritorio/ReId/ReId_DataSet/'

        for img_path in all_image_paths:
            path_sin_prefijo = img_path.replace(prefijo_a_quitar,
                                                    '')  # Extrae la ruta a la carpeta donde estan las imagenes
            nombre_archivo, extension = os.path.splitext(
                path_sin_prefijo)  # Separa extensión para quedarnos solo con id18_135 (Ej)
            pid_string, frame = nombre_archivo.split('_')
            pid = int(pid_string.replace('id', ''))

            if pid in pid_train_range:
                train.append((img_path, pid, 0))
            else:
                gallery.append((img_path, pid, 0))

        random.shuffle(gallery)
        query_size = int(0.02 * len(gallery))  # Se define el tamaño de query

        for img_path, pid, _ in gallery[0:query_size]:
            query.append((img_path, pid, 1))

        gallery = gallery[query_size:]

        return train, query, gallery


# Crear una instancia del conjunto de datos

torchreid.data.register_image_dataset('dataSet_raulherreromuela', NewDataset)

# use your own dataset only
datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='dataSet_raulherreromuela'
)

datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='dataSet_raulherreromuela',
    #height=1024,
    #width=512,
    #height=620,
    #width=444,
    batch_size_train=16,
    batch_size_test=16,
)

model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=datamanager.num_train_pids,
    loss='softmax'
)

model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='adam',
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler
)

engine.run(
    save_dir='log/osnet_x1_0',
    max_epoch=60,
    eval_freq=10,
    print_freq=10
)