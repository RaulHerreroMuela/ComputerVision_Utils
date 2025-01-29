import cv2
import numpy as np

class ImageProcessor:

    def __init__(self, config, path_img):
        self.config = config
        self.path_img = path_img
    @staticmethod
    def read_image(path_image):
        image = cv2.imread(path_image)
        if image is not None:
            return image[:, :, [2, 1, 0]]  # Cambia BGR a RGB
        else:
            raise FileNotFoundError(f"No se pudo encontrar la imagen en {path_image}")

    @staticmethod
    def crop(image, bbox):
        y1, x1, h, w = bbox
        return image[y1:y1 + h, x1:x1 + w]

    @staticmethod
    def resize(image, input_shape):
        return cv2.resize(image, input_shape)

    @staticmethod
    def normalize(image):
        return image / 255.0

    def process_image(self, path_image, bbox=None, input_shape=None, normalize=False):
        image = self.read_image(path_image)
        if bbox is not None:
            image = self.crop(image, bbox)
        if input_shape is not None:
            image = self.resize(image, input_shape)
        if normalize:
            image = self.normalize(image)
        return image