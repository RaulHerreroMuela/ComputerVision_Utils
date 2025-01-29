import os
import tensorflow as tf
from monicas import CSVMaitre

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


path_models = r'C:\Users\aquacorp\Desktop\modelos_tflite\simancas_cv2_worker'

models_list = os.listdir(path_models)
while(len(models_list) != 0):
    model_path = os.path.join(path_models, models_list[0])
    list_files = os.listdir(model_path)

    for file in list_files:
        if file.endswith(".h5"):
            h5_path = os.path.join(model_path, file)
            keras_model = tf.keras.models.load_model(h5_path)
            save_path = os.path.join(model_path, (file[:-3] + '.tflite'))
            h5_to_tflite_convert(model=keras_model, save_path_tflite=save_path)


        elif file.endswith(".ckpt.index"):
            for sub_file in list_files:
                if sub_file.endswith(".yaml"):
                    yaml_file = os.path.join(model_path, sub_file)
                    maitre = CSVMaitre.from_file(yaml_file, ignore_errors=False, only_predict=True)
                    maitre.model.export()

            new_file_list = os.listdir(model_path)
            for new_subfile in new_file_list:
                if os.path.isdir(os.path.join(model_path, new_subfile)):
                    save_path = os.path.join(model_path, (file[:-11] + '.tflite'))
                    saved_model_to_tflite_convert(saved_model_dir=os.path.join(model_path, new_subfile),
                                                  save_path_tflite=save_path)

    models_list.remove(models_list[0])