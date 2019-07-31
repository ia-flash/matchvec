"""SSD detection"""
import os
import cv2
import json
import numpy as np
import pandas as pd
from matchvec.utils import timeit
from BaseModel import BaseModel

# DETECTION_MODEL = 'faster_rcnn_resnet101_coco_2018_01_28/'
DETECTION_MODEL = 'ssd_mobilenet_v2_coco_2018_03_29/'
DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD'))
SWAPRB = False

with open(os.path.join(os.environ['BASE_MODEL_PATH'], DETECTION_MODEL, 'labels.json')) as json_data:
    CLASS_NAMES = json.load(json_data)


class Detector(BaseModel):
    """SSD Mobilenet object detection

    DETECTION_MODEL: Detection model to use
    DETECTION_THRESHOLD: Detection threshold
    SWAPRB: Swap R and B chanels (usefull when opening using opencv) (Default: False)
    """

    @timeit
    def __init__(self):
        self.files = [ 'frozen_inference_graph.pb', 'config.pbtxt', 'labels.json']
        dst_path = os.path.join(
            os.environ['BASE_MODEL_PATH'], DETECTION_MODEL)
        src_path = DETECTION_MODEL

        self.download_model_folder(dst_path, src_path)

        self.model = cv2.dnn.readNetFromTensorflow(
                os.path.join(dst_path, 'frozen_inference_graph.pb'),
                os.path.join(dst_path, 'config.pbtxt')
        )

    def prediction(self, image: np.ndarray) -> np.ndarray:
        """Inference

        Args:
            image: image to make inference
        Returns:
            result: Predictions form SSD Mobilenet
        """
        self.model.setInput(
                cv2.dnn.blobFromImage(
                    image, size=(300, 300),
                    swapRB=SWAPRB,
                    crop=False)
                )
        cvOut = self.model.forward()
        result = cvOut[0, 0, :, :]
        return result

    def create_df(self, result: np.ndarray, image: np.ndarray) -> pd.DataFrame:
        """Filter predictions and create an output DataFrame

        Args:
            result: Result from prediction model
            image: Image where the inference has been made

        Returns:
            df: Object detection filtered predictions
        """
        height, width = image.shape[:-1]
        df = pd.DataFrame(
                result,
                columns=[
                    '_', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2'])
        df = df.assign(
                x1=lambda x: (x['x1'] * width).astype(int).clip(0),
                y1=lambda x: (x['y1'] * height).astype(int).clip(0),
                x2=lambda x: (x['x2'] * width).astype(int),
                y2=lambda x: (x['y2'] * height).astype(int),
                class_name=lambda x: (
                    x['class_id'].astype(int).astype(str).replace(CLASS_NAMES)),
                label=lambda x: (
                    x.class_name + ': ' + (
                        x['confidence'].astype(str).str.slice(stop=4)
                        )
                    )
                )
        df = df[df['confidence'] > DETECTION_THRESHOLD]
        return df
