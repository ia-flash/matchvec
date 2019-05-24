"""SSD detection"""
import os
import cv2
import json
import numpy as np
import pandas as pd
from utils import timeit

DETECTION_MODEL = 'googlelenet_cars/'
DETECTION_THRESHOLD = 0.4
SWAPRB = False

with open(os.path.join('/model', DETECTION_MODEL, 'labels.json')) as json_data:
    CLASS_NAMES = json.load(json_data)


class ClassDetector():
    """SSD Mobilenet object detection

    DETECTION_MODEL: Detection model to use
    DETECTION_THRESHOLD: Detection threshold
    SWAPRB: Swap R and B chanels (usefull when opening using opencv) (Default: False)
    """

    @timeit
    def __init__(self):
        self.model = cv2.dnn.readNetFromCaffe(
                os.path.join(
                    '/model', DETECTION_MODEL, 'deploy.prototxt'),
                os.path.join(
                    '/model', DETECTION_MODEL, 'googlenet_finetune_web_car_iter_10000.caffemodel')
                )

    def prediction(self, image: np.ndarray) -> np.ndarray:
        """Inference

        Args:
            image: image to make inference
        Returns:
            result: Predictions form SSD Mobilenet
        """
        blob = cv2.dnn.blobFromImage(image, scalefactor=1, size=(224, 224), mean=(104, 117, 123), swapRB=SWAPRB, crop=False)
        self.model.setInput(blob)
        output = self.model.forward()
        idxs = np.argsort(output[0])[::-1][:5]
        prob = [float(output[0][i]) for i in idxs]
        pred = [CLASS_NAMES[str(i)] for i in idxs]
        return pred, prob

    def lot_prediction(self, image: np.ndarray, df: pd.DataFrame):
        final_pred: List = list()
        final_prob: List = list()
        for idx, val in df.iterrows():
            crop_image = image[int(val['y1']):int(val['y2']),int(val['x1']):int(val['x2'])]
            pred, prob = self.prediction(crop_image)
            final_pred.append(pred)
            final_prob.append(prob)
        return final_pred, final_prob

