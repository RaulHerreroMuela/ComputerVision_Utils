import os

import numpy as np
import cv2
import pandas as pd
import yaml
from tqdm import tqdm
import base64

from monicas.models.autoencoder_to_regression_tflite import AutoencoderToRegressionTFlite
from monicas.models.foam_percentage_segmentation_tflite import FoamSegmentationTFLite
from monicas.models.EfficieNet_reg_and_classif_tflite import EfficientnetModelTFlite
from monicas.models.autoencoder_anomaly_tflite import AutoencoderAnomalyTFlite
from monicas.models.autoencoder_dimReducer_tflite import AutoencoderDimReducerTFlite

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
    resized_image = cv2.resize(image, input_shape)
    resized_image = np.expand_dims(resized_image, axis=0)
    resized_image = resized_image.astype(np.float32)
    return resized_image

def normalize_image(image):
    return image/255

def pre_infer_process(bbox, input_shape, normalize, path_image):
    image = read_and_adapt_image(path_image=path_image)

    if bbox is not None:
        image = crop_image(image, config['bbox'])

    if input_shape is not None:
        image = resize_image(image=image, input_shape=tuple(config['input_shape']))

    if normalize:
        image = normalize_image(image)

    return image

def one_hot(vector, num_clases):
    encod = np.zeros(num_clases, dtype=np.int8)
    pos_max = np.argmax(vector)
    encod[pos_max] = 1

    return encod, pos_max

#yaml_path = r'C:\Users\aquacorp\Desktop\modelos a convertir\TF\autoencoder_anomalias\plantilla_yamml_autoencoderAnomally.yaml'
#yaml_path = r'C:\Users\aquacorp\Desktop\modelos a convertir\TF\autoencoder_to_regresion\plantilla_yamml_autoencoderRegressor.yaml'
#yaml_path = r'C:\Users\aquacorp\Desktop\modelos a convertir\TF\clasificador_2clases\plantilla_yamml_classif.yaml'
#yaml_path = r'C:\Users\aquacorp\Desktop\modelos a convertir\TF\clasificador_3clases\plantilla_yamml_classif.yaml'
#yaml_path = r'C:\Users\aquacorp\Desktop\modelos a convertir\TF\regresion\plantilla_yamml_regressor.yaml'
yaml_path = r'C:\Users\aquacorp\Desktop\modelos_tflite\chiva_for_tflite\modelo_DQ\config_yamml_autoencoderRegressor_DQ.yaml'
#yaml_path = r'C:\Users\aquacorp\Desktop\modelos a convertir\TF\segmentacion\plantilla_yamml_FoamSegmentation.yaml'
#yaml_path = r'C:\Users\aquacorp\Desktop\modelos_tflite\SantFeliu_for_TFLite\modelo_AU\config_yamml_AutoencoderDimReducer.yaml'

# Cargar la configuración desde el archivo YAML
with open(yaml_path, "r") as file:
    config = yaml.safe_load(file)

prediction_type = config['model_type'] # 'AutoencoderAnomally', 'AutoencoderRegressor', 'Regressor', 'FoamSegmentation', 'classifier'

if prediction_type == 'AutoencoderRegressor':
    path_image = r'C:\Users\aquacorp\Desktop\aquagit-ia-code\aqua_project\images\20241024074734.tif'

    model = AutoencoderToRegressionTFlite(
        id=config['id'],
        autoencoder_path=config['autoencoder_path'],
        regressor_path=config['regressor_path'],
        parameters=config['selected_features']
    )

    image = pre_infer_process(bbox=config['bbox'], input_shape=tuple(config['input_shape']), normalize=config['normalize'], path_image=path_image)
    output = model.predict(data=image)

    print(output)

elif prediction_type == 'AutoencoderDimReducer':
    path_images = r'C:\Users\aquacorp\Desktop\Imagenes\SantFeliu_entrada_sin_retorno\muestreo'
    img_features = []
    #columnas = ['img_name'] + [str(i) for i in range(128)]
    #features = pd.DataFrame(columns=columnas)
    #features = pd.DataFrame([[None] * len(columnas) for _ in range(len(os.listdir(path_images)))], columns=columnas)

    model = AutoencoderDimReducerTFlite(
        id=config['id'],
        autoencoder_path=config['autoencoder_path']
    )

    num_fila = 0
    for img in tqdm(os.listdir(path_images), desc='Procesando imagenes', total=len(os.listdir(path_images))):
        path_img = os.path.join(path_images, img)
        image = pre_infer_process(bbox=config['bbox'], input_shape=tuple(config['input_shape']), normalize=config['normalize'], path_image=path_img)

        output = model.predict(data=image)

        img_features.append([img] + output.flatten().tolist())

        if num_fila==0:
            print(output.shape)

        num_fila += 1
        #row = {'img_name': img}
        #row.update({str(i): output[i] for i in range(128)})
        #features = pd.concat([features, pd.DataFrame([row])], ignore_index=True)

    num_feat = int(len(img_features[0]) - 1)
    print(num_feat)
    columns = ['img_name'] + [f'{i}' for i in range(num_feat)]
    df = pd.DataFrame(img_features, columns=columns)
    print('END')
    df.to_csv(r'C:\Users\aquacorp\Desktop\modelos_tflite\SantFeliu_for_TFLite/features_TFLite.csv', index=False)

elif prediction_type == 'FoamSegmentation':
    path_image = r'C:\Users\aquacorp\Desktop\aquagit-ia-code\aqua_project\images\20241013132206.tif'

    segmentator_model = FoamSegmentationTFLite(path_image=path_image, input_size=config['input_shape'],
                                               model_path=config['seg_model_path'], roi_mask_path=config['mask_path'],
                                               normalize=config['normalize'], mask_th=config['mask_threshold'],
                                               percent_jump=config['percent_jump'])

    #foam_level = segmentator_model.get_foam_percentage()
    foam_level = segmentator_model.predict()
    print(foam_level)

elif prediction_type == 'Regressor': # No funciona bien
    path_image = r'C:\Users\aquacorp\Desktop\Imagenes\DonHierro\muestreo\20240421204233.jpg'

    model = EfficientnetModelTFlite(
        id=config['id'],
        model_path=config['pure_regressor_path'],
        activation_last_layer=config['activation_last_layer']
    )
    image = pre_infer_process(bbox=config['bbox'], input_shape=tuple(config['input_shape']),
                              normalize=config['normalize'], path_image=path_image)
    output = model.predict(data=image)

    print(output)

elif prediction_type == 'Classifier':
    #path_image = r'C:\Users\aquacorp\Desktop\Imagenes\SantFeliu_entrada_sin_retorno\muestreo\20240616065621.tif'
    path_image = r'C:\Users\aquacorp\Desktop\Imagenes\Simancas\muestreo\20240216162416.jpg'

    model = EfficientnetModelTFlite(
        id=config['id'],
        model_path=config['clasifier_path'],
        activation_last_layer=config['activation_last_layer'],
        num_classes=config['num_clases']
    )

    image = pre_infer_process(bbox=config['bbox'], input_shape=tuple(config['input_shape']),
                              normalize=config['normalize'], path_image=path_image)
    output = model.predict(data=image)

    if config['num_clases'] > 2:
        encoding_class, pre_class = one_hot(vector=output, num_clases=config['num_clases'])
        print(encoding_class)
        print('TMOA: ' + str(pre_class + 1))

    else:
        if output > 0.5:
            print('TMOA: 2')
        else:
            print('TMOA: 1')

elif prediction_type == 'AutoencoderAnomally':
    path_image = r'C:\Users\aquacorp\Desktop\aquagit-ia-code\aqua_project\images\velilla_costra/20240926111246.jpg'
    model = AutoencoderAnomalyTFlite(
        id = config['id'],
        model_path=config['autoencoderAnomaly_path']
    )

    image = pre_infer_process(bbox=config['bbox'], input_shape=tuple(config['input_shape']),
                              normalize=config['normalize'], path_image=path_image)

    output = model.predict(data=image)
    mse = np.mean(np.square(image - output), axis=(1, 2, 3))
    print(mse)
