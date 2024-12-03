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
    #dataset_dir = '.../ReId/ReId_DataSet'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        #self.dataset_dir = osp.join(self.root, self.dataset_dir)
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

        #train, query, gallery = self.generar_dataSets(self.dataset_dir, range(0, 15))

        ruta_train = '.../Escritorio/ReId/dS_Train.txt'
        ruta_query = '.../Escritorio/ReId/dS_Query.txt'
        ruta_gallery = '.../Escritorio/ReId/dS_Gallery.txt'
        train = []
        gallery = []
        query = []

        with open(ruta_train, 'r') as archivo:
            for linea in archivo:
                # Dividir la línea en elementos utilizando la coma como separador
                partes = linea.strip().split(',')
                # Convertir las partes a los tipos de datos originales
                elemento = (partes[0], int(partes[1]), int(partes[2]))
                # Agregar el elemento a la lista
                train.append(elemento)

        with open(ruta_query, 'r') as archivo:
            for linea in archivo:
                # Dividir la línea en elementos utilizando la coma como separador
                partes = linea.strip().split(',')
                # Convertir las partes a los tipos de datos originales
                elemento = (partes[0], int(partes[1]), int(partes[2]))
                # Agregar el elemento a la lista
                query.append(elemento)

        with open(ruta_gallery, 'r') as archivo:
            for linea in archivo:
                # Dividir la línea en elementos utilizando la coma como separador
                partes = linea.strip().split(',')
                # Convertir las partes a los tipos de datos originales
                elemento = (partes[0], int(partes[1]), int(partes[2]))
                # Agregar el elemento a la lista
                gallery.append(elemento)

        super(NewDataset, self).__init__(train, query, gallery, **kwargs)


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
    num_classes=28, #datamanager.num_train_pids,
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
    save_dir='log/osnet_x1_0_050124',
    # Poner a true flag de rank
    visrank=True,
    test_only=True,
    max_epoch=60,
    eval_freq=10,
    print_freq=10
)