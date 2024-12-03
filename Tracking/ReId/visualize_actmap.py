"""Visualizes CNN activation maps to see where the CNN focuses on to extract features.

Reference:
    - Zagoruyko and Komodakis. Paying more attention to attention: Improving the
      performance of convolutional neural networks via attention transfer. ICLR, 2017
    - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
"""
import numpy as np
import os.path as osp
import argparse
import cv2
import torch
from torch.nn import functional as F
from torchreid.data import ImageDataset
import os
import random

import torchreid
from torchreid.utils import (
    check_isfile, mkdir_if_missing, load_pretrained_weights, FeatureExtractor
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRID_SPACING = 10


# Generación dataSet RHM

class NewDataset(ImageDataset):
    dataset_dir = '.../Escritorio/ReId/ReId_DataSet'

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

        ruta_train = '/home/deepfarm/Escritorio/ReId/dS_Train.txt'
        ruta_query = '/home/deepfarm/Escritorio/ReId/dS_Query.txt'
        ruta_gallery = '/home/deepfarm/Escritorio/ReId/dS_Gallery.txt'
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


@torch.no_grad()
def visactmap(
    model,
    test_loader,
    save_dir,
    width,
    height,
    use_gpu,
    img_mean=None,
    img_std=None
):
    if img_mean is None or img_std is None:
        # use imagenet mean and std
        img_mean = IMAGENET_MEAN
        img_std = IMAGENET_STD

    model.eval()

    # descriptors = [] # Añadido RHM para almacenar los descriptores

    for target in list(test_loader.keys()):
        data_loader = test_loader[target]['query'] # only process query images
        # original images and activation maps are saved individually
        actmap_dir = osp.join(save_dir, 'actmap_' + target)
        mkdir_if_missing(actmap_dir)
        print('Visualizing activation maps for {} ...'.format(target))

        for batch_idx, data in enumerate(data_loader):
            imgs, paths = data['img'], data['impath']
            if use_gpu:
                imgs = imgs.cuda()

            # forward to get convolutional feature maps
            try:
                # outputs = model(imgs, return_featuremaps=True)
                outputs = model(imgs, return_featuremaps=True)

            except TypeError:
                raise TypeError(
                    'forward() got unexpected keyword argument "return_featuremaps". '
                    'Please add return_featuremaps as an input argument to forward(). When '
                    'return_featuremaps=True, return feature maps only.'
                )

            if outputs.dim() != 4:
                raise ValueError(
                    'The model output is supposed to have '
                    'shape of (b, c, h, w), i.e. 4 dimensions, but got {} dimensions. '
                    'Please make sure you set the model output at eval mode '
                    'to be the last convolutional feature maps'.format(
                        outputs.dim()
                    )
                )

            # RHM descriptors.append(outputs)
            # compute activation maps
            outputs = (outputs**2).sum(1)
            b, h, w = outputs.size()
            outputs = outputs.view(b, h * w)
            outputs = F.normalize(outputs, p=2, dim=1)
            outputs = outputs.view(b, h, w)

            #descriptors.append(features) # Añadido RHM para guardar los descriptores

            if use_gpu:
                imgs, outputs = imgs.cpu(), outputs.cpu()

            for j in range(outputs.size(0)):
                # get image name
                path = paths[j]
                imname = osp.basename(osp.splitext(path)[0])

                # RGB image
                img = imgs[j, ...]
                for t, m, s in zip(img, img_mean, img_std):
                    t.mul_(s).add_(m).clamp_(0, 1)
                img_np = np.uint8(np.floor(img.numpy() * 255))
                img_np = img_np.transpose((1, 2, 0)) # (c, h, w) -> (h, w, c)

                # activation map
                am = outputs[j, ...].numpy()
                am = cv2.resize(am, (width, height))
                am = 255 * (am - np.min(am)) / (
                    np.max(am) - np.min(am) + 1e-12
                )
                am = np.uint8(np.floor(am))
                am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

                # overlapped
                overlapped = img_np*0.3 + am*0.7
                overlapped[overlapped > 255] = 255
                overlapped = overlapped.astype(np.uint8)

                # save images in a single figure (add white spacing between images)
                # from left to right: original image, activation map, overlapped image
                grid_img = 255 * np.ones(
                    (height, 3*width + 2*GRID_SPACING, 3), dtype=np.uint8
                )
                grid_img[:, :width, :] = img_np[:, :, ::-1]
                grid_img[:,
                         width + GRID_SPACING:2*width + GRID_SPACING, :] = am
                grid_img[:, 2*width + 2*GRID_SPACING:, :] = overlapped
                cv2.imwrite(osp.join(actmap_dir, imname + '.jpg'), grid_img)

            if (batch_idx+1) % 10 == 0:
                print(
                    '- done batch {}/{}'.format(
                        batch_idx + 1, len(data_loader)
                    )
                )
    #return descriptors # Añadido RHM



use_gpu = torch.cuda.is_available()

torchreid.data.register_image_dataset('dataSet_raulherreromuela', NewDataset)

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



test_loader = datamanager.test_loader

model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=datamanager.num_train_pids,
    loss='softmax'
)


model = model.cuda()
#if args.weights and check_isfile(args.weights):
path_pesos_osnet = '.../Escritorio/ultralytics_github/Tracking_deep_farm/log/osnet_x1_0_050124/model/osnet_x1_0_DeepFarm050124.pt'
load_pretrained_weights(model, path_pesos_osnet)

visactmap(
    model, test_loader, '.../Escritorio/Resultados/mapa_de_activacion_PRUEBA_DESCRIPTORES', 128, 256, True
)

rutas_gallery = '.../Escritorio/ReId/dS_Gallery.txt'
r_gallery = []
with open(rutas_gallery, 'r') as archivo:
    for linea in archivo:
        # Dividir la línea en elementos utilizando la coma como separador
        partes = linea.strip().split(',')
        # Convertir las partes a los tipos de datos originales
        elemento = (partes[0])
        # Agregar el elemento a la lista
        r_gallery.append(elemento)


extractor = FeatureExtractor('osnet_x1_0', path_pesos_osnet)
features = extractor(r_gallery)
print(features)

'''
if __name__ == '__main__':
    main()
'''