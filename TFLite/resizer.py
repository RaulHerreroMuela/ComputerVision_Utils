from monicas.workers import ResizerWorker
import tensorflow as tf
import cv2


class BasicResizerWorker(ResizerWorker):
    def __init__(self, size, method='bilinear', preserve_aspect_ratio=False, antialias=True, num_classes=None):
        """
        Constructor of CLAHEWorker.
        :param size: new size to which resize the input image
        :param num_classes: number of classes
        """
        super().__init__()
        self.size = tf.constant(size)
        self.method = method
        self.preserve_aspect_ration = preserve_aspect_ratio
        self.antialias = antialias
        self._logger.info(f"Set size to {self.size}, method to {self.method},preserve aspect ratio to"
                          f"{self.preserve_aspect_ration}, and antialias to {self.antialias}")
        self.num_classes = num_classes

    def transform(self, id_image, image, label=0):
        if self.num_classes is None:
            size = self.size
        else:
            size = self.size[label]
        #image_resized = tf.image.resize(image, size, self.method, self.preserve_aspect_ration, self.antialias)
        #return image_resized.numpy()
        size = tuple(int(dim) for dim in size.numpy())
        image_resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return image_resized