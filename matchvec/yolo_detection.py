"""Yolo detection"""
import os
import cv2
import json
import numpy as np
from typing import List
from matchvec.BaseModel import BaseModel

DETECTION_MODEL = 'yolo'
DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD'))
NMS_THRESHOLD = 0.4  # Non Maximum Supression threshold
SWAPRB = False
SCALE = 0.00392  # 1/255

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = (
            [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()])
    return output_layers


def filter_yolo(chunk: np.ndarray) -> List[dict]:
    """ Filter Yolo chunks

    Create a list of dictionary from each chunk and then filter it with the DETECTION_THRESHOLD

    Args:
        chunk: A Yolo chunk

    Returns:
        df: The object detection predictions for the chunk
    """
    pred = np.argmax(chunk[:, 5:], axis=1)
    prob = np.max(chunk[:, 5:], axis=1)
    df = []
    for i, row in enumerate(chunk):
        if prob[i] > DETECTION_THRESHOLD:
            df += [dict(center_x=row[0],
                        center_y=row[1],
                        w=row[2],
                        h=row[3],
                        class_id=pred[i],
                        confidence=prob[i])]
    return df


class Detector(BaseModel):
    """Yolo object detection

    Args:
        DETECTION_MODEL: Detection model to use
        DETECTION_THRESHOLD: Detection threshold
        NMS_THRESHOLD: Non Maximum Supression threshold (to remove overlapping boxes)
        SWAPRB: Swap R and B chanels (usefull when opening using opencv) (Default: False)
        SCALE: Yolo uses a normalisation factor different than 1 for each pixel
    """
    def __init__(self):
        self.files = ['yolov3.weights', 'yolov3.cfg','labels.json']
        dst_path = os.path.join(
            os.environ['BASE_MODEL_PATH'], DETECTION_MODEL)
        src_path = DETECTION_MODEL

        self.download_model_folder(dst_path, src_path)
        with open(os.path.join(dst_path, 'labels.json')) as json_data:
            self.class_name = json.load(json_data)

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
                  image: np.ndarray) -> List[dict]:
        """Filter predictions and create an output list of dictionary

        Args:
            result: Result from prediction model
            image: Image where the inference has been made

        Returns:
            df: Onject detection predictions filtered
        """
        height, width = image.shape[:-1]
        df = [filter_yolo(i) for i in result]
        df = [j for i in df for j in i] # reduce
        df_tmp = []
        for row in df:
            confidence = row['confidence']
            w = row['w'] * width
            h = row['h'] * height
            x1 = (row['center_x'] * width - (w / 2)).astype(int).clip(0)
            y1 = (row['center_y'] * height - (h / 2)).astype(int).clip(0)
            x2 = x1 + w
            y2 = y1 + h
            class_name = self.class_name[row['class_id'].astype(int).astype(str)]
            label = class_name + ': ' + confidence.round(4).astype(str)
            df_tmp += [dict(x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        w = w,
                        h = h,
                        class_name=class_name,
                        label=label,
                        confidence=float(confidence))]
        cols = ['x1', 'y1', 'w', 'h']
        indices = cv2.dnn.NMSBoxes(
                [[row[col] for col in cols] for row in df_tmp],
                [row['confidence'] for row in df_tmp], DETECTION_THRESHOLD, NMS_THRESHOLD)
        df = []
        if len(indices) > 0:
            for i, row in enumerate(df_tmp):
                if i in indices.flatten():
                    df += [row]
        return df
