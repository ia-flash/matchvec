"""Yolo detection"""
import os
import cv2
import json
import numpy as np
import pandas as pd
from typing import List
from utils import timeit

DETECTION_MODEL = 'yolo'
DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD'))
NMS_THRESHOLD = 0.4  # Non Maximum Supression threshold
SWAPRB = False
SCALE = 0.00392  # 1/255

with open(os.path.join(os.environ['BASE_MODEL_PATH'], DETECTION_MODEL, 'labels.json')) as json_data:
    CLASS_NAMES = json.load(json_data)


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = (
            [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()])
    return output_layers


def filter_yolo(chunk: np.ndarray) -> pd.DataFrame:
    """ Filter Yolo chunks

    Create a DataFrame from each chunk and then filter it with the DETECTION_THRESHOLD

    Args:
        chunk: A Yolo chunk

    Returns:
        df: The object detection predictions for the chunk
    """
    pred = np.argmax(chunk[:, 5:], axis=1)
    prob = np.max(chunk[:, 5:], axis=1)
    df = pd.DataFrame(
            np.concatenate(
                [chunk[:, :4], pred.reshape(-1, 1), prob.reshape(-1, 1)],
                axis=1
                ),
            columns=[
                'center_x', 'center_y', 'w', 'h', 'class_id', 'confidence'])
    df = df[df['confidence'] > DETECTION_THRESHOLD]
    return df


class Detector():
    """Yolo object detection

    Args:
        DETECTION_MODEL: Detection model to use
        DETECTION_THRESHOLD: Detection threshold
        NMS_THRESHOLD: Non Maximum Supression threshold (to remove overlapping boxes)
        SWAPRB: Swap R and B chanels (usefull when opening using opencv) (Default: False)
        SCALE: Yolo uses a normalisation factor different than 1 for each pixel
    """
    @timeit
    def __init__(self):
        self.model = cv2.dnn.readNetFromDarknet(
                os.path.join(os.environ['BASE_MODEL_PATH'], DETECTION_MODEL, 'yolov3.cfg'),
                os.path.join(os.environ['BASE_MODEL_PATH'], DETECTION_MODEL, 'yolov3.weights')
                )

    def prediction(self, image: np.ndarray) -> List[np.ndarray]:
        """ Inference

        Make inference

        Args:
            image: input image

        Returns:
            result: Yolo boxes from object detections
        """
        blob = cv2.dnn.blobFromImage(image, SCALE, (416, 416), (0, 0, 0),
                                     swapRB=SWAPRB, crop=False)
        self.model.setInput(blob)
        result = self.model.forward(get_output_layers(self.model))
        return result

    def create_df(self, result: List[np.ndarray],
                  image: np.ndarray) -> pd.DataFrame:
        """Filter predictions and create an output DataFrame

        Args:
            result: Result from prediction model
            image: Image where the inference has been made

        Returns:
            df: Onject detection predictions filtered
        """
        height, width = image.shape[:-1]
        df = pd.concat([filter_yolo(i) for i in result])
        df = df.assign(
                center_x=lambda x: (x['center_x'] * width),
                center_y=lambda x: (x['center_y'] * height),
                w=lambda x: (x['w'] * width),
                h=lambda x: (x['h'] * height),
                x1=lambda x: (x.center_x - (x.w / 2)).astype(int).clip(0),
                y1=lambda x: (x.center_y - (x.h / 2)).astype(int).clip(0),
                x2=lambda x: (x.x1 + x.w).astype(int),
                y2=lambda x: (x.y1 + x.h).astype(int),
                class_name=lambda x: (
                    x['class_id'].astype(int).astype(str).replace(CLASS_NAMES)),
                label=lambda x: (
                    x.class_name + ': ' + (
                        x['confidence'].astype(str).str.slice(stop=4)
                        )
                    )
                )
        cols = ['x1', 'y1', 'w', 'h']
        indices = cv2.dnn.NMSBoxes(
                df[cols].values.tolist(),
                df['confidence'].tolist(), DETECTION_THRESHOLD, NMS_THRESHOLD)
        if len(indices) > 0:
            df = df.iloc[indices.flatten()]
        return df
