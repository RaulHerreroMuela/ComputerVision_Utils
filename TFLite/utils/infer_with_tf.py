import numpy as np
import cv2
import yaml
import tensorflow as tf
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
    resized_image = resized_image.astype(np.float32)# / 255.0
    return resized_image

def normalize_image(image):
    return image/255

def pre_infer_process(bbox, input_shape, normalize, path_image):
    image = read_and_adapt_image(path_image=path_image)

    if bbox is not None:
        image = crop_image(image, bbox)

    if input_shape is not None:
        image = resize_image(image=image, input_shape=input_shape)

    if normalize:
        image = normalize_image(image)

    return image

path_image = r'C:\Users\aquacorp\Desktop\Imagenes\DonHierro\muestreo\20240421204233.jpg'
bbox = [486, 825, 1373, 706]
input_shape = (224, 224)
normalize = True


model = tf.keras.models.load_model(r'C:\Users\aquacorp\Desktop\modelos a convertir\TF\regresion\donhierro_e_regresion_imagenes_DQEfficientnetModel0.h5')
image = pre_infer_process(bbox=bbox, input_shape=input_shape,
                          normalize=normalize, path_image=path_image)

output = model.predict(image)
print(output)