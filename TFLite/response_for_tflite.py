import os
import numpy as np
import cv2
import yaml
from tqdm import tqdm
import tensorflow as tf

from monicas.models.autoencoder_to_regression_tflite import AutoencoderToRegressionTFlite
from monicas.models.foam_percentage_segmentation_tflite import FoamSegmentationTFLite
from monicas.models.EfficieNet_reg_and_classif_tflite import EfficientnetModelTFlite
from monicas.models.autoencoder_anomaly_tflite import AutoencoderAnomalyTFlite
import pandas as pd

def read_yaml(model_dir):
    files = os.listdir(model_dir)

    for file in files:
        if file.endswith('.yaml'):
            file = os.path.join(model_dir, file)
            with open(file, "r") as yaml_file:
                config = yaml.safe_load(yaml_file)

    return config

def load_model(config):
    model_type = config['model_type']

    if model_type == 'AutoencoderAnomally':
        model = AutoencoderAnomalyTFlite(
            id = config['id'],
            model_path=config['autoencoderAnomaly_path']
        )

    elif model_type == 'Classifier':
        model = EfficientnetModelTFlite(
            id=config['id'],
            model_path=config['clasifier_path'],
            num_classes=config['num_clases']
        )

    elif model_type == 'Regressor':
        model = EfficientnetModelTFlite(
            id=config['id'],
            model_path=config['pure_regressor_path'],
        )

    elif model_type == 'FoamSegmentation':
        model = FoamSegmentationTFLite(
            input_size=config['input_shape'],
            model_path=config['seg_model_path'], roi_mask_path=config['mask_path'],
            normalize=config['normalize'], mask_th=config['mask_threshold'],
            percent_jump=config['percent_jump']
        )

    elif model_type == 'AutoencoderRegressor':
        model = AutoencoderToRegressionTFlite(
            id=config['id'],
            autoencoder_path=config['autoencoder_path'],
            regressor_path=config['regressor_path'],
            parameters=config['selected_features']
        )

    else:
        print('Unknown model type')
        return 'NULL'

    return model
def read_and_adapt_image(path_image):
    image_proces = cv2.imread(path_image)
    image_proces = image_proces[:, :, [2, 1, 0]]

    return image_proces
def crop_image(image, bbox):
    y1, x1, h, w = bbox
    y2 = y1 + h
    x2 = x1 + w

    image_crop = image[y1:y2, x1:x2]
    return image_crop
def resize_image(image, input_shape):
    resized_image = cv2.resize(image, input_shape, interpolation=cv2.INTER_AREA)
    resized_image = np.expand_dims(resized_image, axis=0)
    resized_image = resized_image.astype(np.float32)

    return resized_image
def normalize_image(image):
    return image/255
def process_image(bbox, input_shape, normalize, path_image):
    image = read_and_adapt_image(path_image=path_image)

    if bbox is not None:
        image = crop_image(image, bbox)

    if input_shape is not None:
        image = resize_image(image=image, input_shape=input_shape)

    if normalize:
        image = normalize_image(image)

    return image
def one_hot(vector, num_clases):
    encod = np.zeros(num_clases, dtype=np.int8)
    pos_max = np.argmax(vector)
    encod[pos_max] = 1

    return encod, pos_max
def get_params_predictions(dict, model_list, path_models, path_image):
    for model in model_list:
        param = model[-2:]
        path_param_model = os.path.join(path_models, model)
        config = read_yaml(path_param_model)

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

#def get_dif_promedios(img_path, bbox, input_shape): # Esta función sirve para lo de las diferencias del resize
    #    img = cv2.imread(img_path)
    #if bbox is not None:
    #    img_crop_raul = crop_image(img, bbox)
    #    img_crop_tf = tf.image.crop_to_bounding_box(img, bbox[0], bbox[1], bbox[2], bbox[3])
    #    img_crop = crop_image(img, bbox)

    #    resta_crop = abs(sum(sum(sum(img_crop_raul - img_crop_tf))))
    #else:
    #    img_crop = img
#    resta_crop = 0.1

#    cv2_resize = cv2.resize(img_crop, input_shape, interpolation=cv2.INTER_AREA)
#    tf_resize = tf.image.resize(img_crop, input_shape, method='bilinear', antialias=True)
#    tf_resize = tf_resize.numpy()

#    resta_size = cv2_resize - tf_resize#[:,:,[2,1,0]]



#    return [sum(sum(sum(abs(resta_size)))), resta_crop]

path_models = r'C:\Users\aquacorp\Desktop\modelos_tflite\simancas_cv2_worker'
path_imgs = r'C:\Users\aquacorp\Desktop\Imagenes\Simancas\muestreo'

imgs_test_names = os.listdir(path_imgs)
imgs_test_path = []

for img_name in imgs_test_names:
    path_img = os.path.join(path_imgs, img_name)
    imgs_test_path.append(path_img)

ejecutado_una_vez = False
num_fila = 0
for img_path in tqdm(imgs_test_path, desc="Procesando imágenes"):
    #print(img_path[-18:])
    dict_param = {}
    model_list = os.listdir(path_models)

    if 'modelo_AN' in model_list:
        path_AN_model = os.path.join(path_models, 'modelo_AN')
        config = read_yaml(path_AN_model)

        image = process_image(bbox=config['bbox'], input_shape=config['input_shape'], normalize=config['normalize'], path_image=img_path)

        model = load_model(config=config)
        prediction = model.predict(data=image)
        mse = np.mean(np.square(image - prediction), axis=(1, 2, 3))

        if mse > config['anomaly_thresh']: # Caso de imagen anomala
            dict_param['STATUS'] = {'alert':'ANOMALY', 'status':'CORRECT'}
            print('anomala')

        else: # Caso de imagen normal
            model_list.remove('modelo_AN')
            dict_param['ESTIMATION'] = []
            dict_param = get_params_predictions(dict=dict_param, model_list=model_list, path_models=path_models, path_image=img_path)
            dict_param['STATUS'] = {'alert': 'NULL', 'status': 'CORRECT'}

    else:
        dict_param['ESTIMATION'] = []
        dict_param = get_params_predictions(dict=dict_param, model_list=model_list, path_models=path_models, path_image=img_path)
        dict_param['STATUS'] = {'alert': 'NULL', 'status': 'CORRECT'}


    if not ejecutado_una_vez:
        columnas = []
        columnas.append('img_name')
        # Esto se va a utilizar para lo de meter la media de los pixeles de la resta entre cv2resize y tfimageresize
        #columnas.append('dif_resize')

        for dicts in dict_param['ESTIMATION']:
            columnas.append(dicts['name'])
        tflite_results = pd.DataFrame([[None] * len(columnas) for _ in range(len(imgs_test_path))], columns=columnas)
        ejecutado_una_vez = True

    for dict in dict_param['ESTIMATION']:
        column = dict['name']
        tflite_results.loc[num_fila, column] = dict['value']

    tflite_results.loc[num_fila, 'img_name'] = img_path[-18:]

    #dif_resize, dif_crop = get_dif_promedios(img_path=img_path, bbox=config['bbox'], input_shape=config['input_shape'])
    #tflite_results.loc[num_fila, 'dif_resize'] = dif_resize
    num_fila += 1

tflite_results.to_csv(r'C:\Users\aquacorp\Desktop\modelos_tflite\simancas_cv2_worker\NotInver_TFLite_cv2_muestreo_Simancas.csv', index=False)
#tflite_results.to_csv(r'C:\Users\aquacorp\Desktop\Imagenes\DonHierro\prueba.csv', index=False)
#print(dict_param)