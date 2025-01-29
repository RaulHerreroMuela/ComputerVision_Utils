import os
from flask import Flask
from flask import request
import numpy as np
import cv2
import yaml

import base64
from .image_processor import ImageProcessor
from .model_loader import ModelLoader

#from monicas.models.autoencoder_to_regression_tflite import AutoencoderToRegressionTFlite
#from monicas.models.foam_percentage_segmentation_tflite import FoamSegmentationTFLite
#from monicas.models.EfficieNet_reg_and_classif_tflite import EfficientnetModelTFlite
#from monicas.models.autoencoder_anomaly_tflite import AutoencoderAnomalyTFlite

def read_yaml(model_dir):
    files = os.listdir(model_dir)

    for file in files:
        if file.endswith('.yaml'):
            file = os.path.join(model_dir, file)
            with open(file, "r") as yaml_file:
                config = yaml.safe_load(yaml_file)

    return config

#def load_model(config):
#    model_type = config['model_type']

#    if model_type == 'AutoencoderAnomally':
#        model = AutoencoderAnomalyTFlite(
#            id = config['id'],
#            model_path=config['autoencoderAnomaly_path']
#        )

#    elif model_type == 'Classifier':
#        model = EfficientnetModelTFlite(
#            id=config['id'],
#            model_path=config['clasifier_path'],
#            activation_last_layer=config['activation_last_layer'],
#            num_classes=config['num_clases']
#        )

#    elif model_type == 'Regressor':
#        model = EfficientnetModelTFlite(
#            id=config['id'],
#            model_path=config['pure_regressor_path'],
#            activation_last_layer=config['activation_last_layer']
#        )

#    elif model_type == 'FoamSegmentation':
#        model = FoamSegmentationTFLite(
#            path_image=path_image, input_size=config['input_shape'],
#            model_path=config['seg_model_path'], roi_mask_path=config['mask_path'],
#            normalize=config['normalize'], mask_th=config['mask_threshold'],
#            percent_jump=config['percent_jump']
#        )

#    elif model_type == 'AutoencoderRegressor':
#        model = AutoencoderToRegressionTFlite(
#            id=config['id'],
#            autoencoder_path=config['autoencoder_path'],
#            regressor_path=config['regressor_path'],
#            parameters=config['selected_features']
#        )

#    else:
#        print('Unknown model type')
#        return 'NULL'

#    return model

#def read_and_adapt_image(path_image):
#    image_proces = cv2.imread(path_image)
#    image_proces = image_proces[:, :, [2, 1, 0]]

#    return image_proces

#def crop_image(image, bbox):
#    y1, x1, h, w = bbox
#    y2 = y1 + h
#    x2 = x1 + w

#    image_crop = image[y1:y2, x1:x2]

#    return image_crop

#def resize_image(image, input_shape):
#    resized_image = cv2.resize(image, input_shape)
#    resized_image = np.expand_dims(resized_image, axis=0)
#    resized_image = resized_image.astype(np.float32)
#    return resized_image

#def normalize_image(image):
#    return image/255

#def process_image(bbox, input_shape, normalize, path_image):
#    image = read_and_adapt_image(path_image=path_image)

#    if bbox is not None:
#        image = crop_image(image, bbox)

#    if input_shape is not None:
#        image = resize_image(image=image, input_shape=input_shape)

#    if normalize:
#        image = normalize_image(image)

#    return image

def one_hot(vector, num_clases):
    encod = np.zeros(num_clases, dtype=np.int8)
    pos_max = np.argmax(vector)
    encod[pos_max] = 1

    return encod, pos_max

def get_params_predictions(dict, model_list, path_models):
    for model in model_list:
        param = model[-2:]
        path_param_model = os.path.join(path_models, model)
        config = read_yaml(path_param_model)

        imageProcessor = ImageProcessor()

        image = process_image(bbox=config['bbox'], input_shape=config['input_shape'], normalize=config['normalize'], path_image=path_image)

        model = load_model(config)
        prediction = model.predict(data=image)

        if param == 'NI':
            if config['num_clases'] == 2:
                fq_param_dict = {'deviation': 0,
                                 'name': 'TMOA',
                                 'percentage': 0,
                                 'value': round(prediction[0][0])+1}
            else:
                encod, pos_max = one_hot(vector=prediction, num_clases=config['num_clases'])
                fq_param_dict = {'deviation': 0,
                                 'name': 'TMOA',
                                 'percentage': 0,
                                 'value': pos_max+1}
        else:
            fq_param_dict = {'deviation':0,
                             'name': param,
                             'percentage':0,
                             'value': prediction[0][0]}

        dict['ESTIMATION'].append(fq_param_dict)

    return dict

path_models = r'C:\Users\aquacorp\Desktop\chiva_for_tflite' # Ubicacion de los modelos en la estacion
path_image = r'C:\Users\aquacorp\Desktop\Imagenes\Chiva_pozo_entrada\muestreo/20240815061249.tif' # Imagen de turno

prediction_type = True # Este flag va a controlar si se trata de una imagen que viene de la estacion o se manda como peticion
                       # para testear los modelos de la estacion

dict_param = {}
model_list = os.listdir(path_models)

if 'modelo_AN' in model_list:
    path_AN_model = os.path.join(path_models, 'modelo_AN')
    config = read_yaml(path_AN_model)

    image = process_image(bbox=config['bbox'], input_shape=config['input_shape'], normalize=config['normalize'], path_image=path_image)

    model = load_model(config=config)
    prediction = model.predict(data=image)
    mse = np.mean(np.square(image - prediction), axis=(1, 2, 3))

    if mse > config['anomaly_thresh']: # Caso de imagen anomala
        dict_param['STATUS'] = {'alert':'ANOMALY', 'status':'CORRECT'}

    else: # Caso de imagen normal
        model_list.remove('modelo_AN')
        dict_param['ESTIMATION'] = []
        dict_param = get_params_predictions(dict=dict_param, model_list=model_list, path_models=path_models)
        dict_param['STATUS'] = {'alert': 'NULL', 'status': 'CORRECT'}

else:
    dict_param['ESTIMATION'] = []
    dict_param = get_params_predictions(dict=dict_param, model_list=model_list, path_models=path_models)
    dict_param['STATUS'] = {'alert': 'NULL', 'status': 'CORRECT'}

print(dict_param)