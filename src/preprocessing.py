import cv2
import numpy as np


def preprocess_image(image):
    image = cv2.medianBlur(image, 3)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    image = clahe.apply(image)

    image = image.astype(np.float32) / 255.0

    return image