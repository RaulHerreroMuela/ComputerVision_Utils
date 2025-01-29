import base64
import cv2
import numpy as np
from tensorflow import lite

class FoamSegmentationTFLite:
    def __init__(self, roi_mask_path: str = None, input_size=(128, 128), model_path=None,
                 normalize=False, umbrales=None, mask_th=0.5, percent_jump = 10):
        self.input_size = input_size
        self.interpreter = lite.Interpreter(model_path=model_path)
        self.normalize = normalize
        self.umbrales = umbrales
        self.mask_th = mask_th
        self.percent_jump = percent_jump

        self.foam_percentage = None
        self.foam_percentage_round = None
        self.foam_level = None

        if roi_mask_path is None:
            self.resized_roi_mask = np.zeros(self.input_size, dtype=bool)
        else:
            # Load ROI mask
            mask_array = cv2.imread(roi_mask_path, cv2.IMREAD_GRAYSCALE)/255
            self.resized_roi_mask = np.round(mask_array).astype(np.uint8)
            self.resized_roi_mask = cv2.resize(self.resized_roi_mask, input_size)

        
    def predict(self, data):
        # Predict foam mask
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.interpreter.set_tensor(input_details[0]['index'], data)
        self.interpreter.invoke()

        predicted_mask = self.interpreter.get_tensor(output_details[0]['index'])

        foam_mask = (predicted_mask > self.mask_th).astype(np.uint8)
        foam_mask = np.squeeze(foam_mask, axis=0)
        foam_mask = np.squeeze(foam_mask, axis=2)
        roi_pixels = np.sum(self.resized_roi_mask == 1)
        sum_water_foam_masks = foam_mask + self.resized_roi_mask
        
        if roi_pixels == 0:
            print("ROI incorrecta")
            self.foam_percentage = 0
        else:
            foam_pixels = np.sum(sum_water_foam_masks == 2)
            self.foam_percentage = (foam_pixels / roi_pixels) * 100
            limits = list(range(0, 101, self.percent_jump))

            self.foam_percentage_round = [[min(limits, key=lambda x: abs(x-self.foam_percentage))]] # Metido en 2 listas para hacer funcionamiento equivalente al resto de parametros

        return self.foam_percentage_round
