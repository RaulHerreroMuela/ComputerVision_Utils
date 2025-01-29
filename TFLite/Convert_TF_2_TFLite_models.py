import tensorflow as tf

from monicas import CSVMaitre

model_h5 = True # True para convertir modelos en formato .h5.
autoencoder_pre_reg = False # True tambien que el anterior si el modelo .h5 a convertir es el del autoencoder preregresion y no el de regresion o segmentacion.
                   # El resto de modelos pasan por el else, que los convierte a formato .h5
model_save = False
def h5_to_tflite_convert(model, save_path_tflite):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(save_path_tflite, 'wb') as f:
        f.write(tflite_model)

def saved_model_to_tflite_convert(saved_model_dir, save_path_tflite):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    with open(save_path_tflite, "wb") as f:
        f.write(tflite_model)

save_path = r'C:\Users\aquacorp\Desktop\chiva_for_tflite\modelo_TU\regressorTU.tflite' # Parametro a modificar

if model_h5:
    if autoencoder_pre_reg:
        keras_model = tf.keras.models.load_model(r'C:\Users\aquacorp\Desktop\chiva_for_tflite\modelo_SS/model.h5')
        encoder = tf.keras.Model(
            inputs=keras_model.input,
            outputs=keras_model.get_layer('bottleneck').output
        )
        h5_to_tflite_convert(encoder, save_path)
    else:
        keras_model = tf.keras.models.load_model(r'C:\Users\aquacorp\Desktop\chiva_for_tflite\modelo_TU/model_Turbidez.h5') # Parametro a modificar
        h5_to_tflite_convert(keras_model, save_path)

elif model_save:
    saved_model_dir = r'C:\Users\aquacorp\Desktop\chiva_for_tflite\modelo_AN\classiAutoencoder'
    saved_model_to_tflite_convert(saved_model_dir=saved_model_dir, save_path_tflite=save_path)

else: # Aquí vamos a poner la opción de convertir a .h5 los modelos que estan con los archivos checkpoint, ckpt... y luego habrá que ejecutar otra vez el script
        # para convertirlo de .h5 a .tflite
    path_yamml = r'C:\Users\aquacorp\Desktop\modelos a convertir\TF\regresion/donhierro_e_regresion_imagenes_DQ.yaml'
    maitre = CSVMaitre.from_file(path_yamml, ignore_errors=False, only_predict=True)

    maitre.model.export()
