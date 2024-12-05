import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self, target_size=(224, 224), normalize=True):
        """
        Initializes the image preprocessor.
        :param target_size: Tuple indicating the target size (width, height).
        :param normalize: Whether to normalize pixel values to the range [0, 1].
        """
        self.target_size = target_size
        self.normalize = normalize

    def process(self, image_path):
        """
        Processes a single image.
        :param image_path: Path to the image file.
        :return: Processed image as a numpy array.
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Resize the image
        image = cv2.resize(image, self.target_size)

        # Normalize the image
        if self.normalize:
            image = image / 255.0

        return image
