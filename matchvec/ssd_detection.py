"""SSD detection"""
import os
import cv2
import json
import numpy as np
from typing import List, Union

from matchvec.utils import timeit
from matchvec.BaseModel import BaseModel

# DETECTION_MODEL = 'faster_rcnn_resnet101_coco_2018_01_28/'
DETECTION_MODEL = 'ssd_mobilenet_v2_coco_2018_03_29/'
DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD'))
SWAPRB = False

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

        with open(os.path.join(dst_path, 'labels.json')) as json_data:
            self.class_name = json.load(json_data)


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

    def create_df(self, result: np.ndarray, image: np.ndarray) -> List[dict]:
        """Filter predictions and create an output dictionary

        Args:
            result: Result from prediction model
            image: Image where the inference has been made

        Returns:
            df: Object detection filtered predictions
        """
        height, width = image.shape[:-1]
        df = []
        for row in result:
            confidence=row[2]
            if confidence > DETECTION_THRESHOLD:
                class_name = self.class_name[row[1].astype(int).astype(str)]
                x1 = (row[3]* width).astype(int).clip(0)
                y1 = (row[4]* height).astype(int).clip(0)
                x2 = (row[5]* width).astype(int)
                y2 = (row[6]* height).astype(int)
                label = class_name + ': ' + confidence.round(4).astype(str)
                df += [dict(x1=int(x1),
                            y1=int(y1),
                            x2=int(x2),
                            y2=int(y2),
                            class_name=class_name,
                            label=label,
                            confidence=float(confidence))]


        return df
