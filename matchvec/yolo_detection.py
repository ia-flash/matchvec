"""Yolo detection"""
import os
import cv2
import json
import numpy as np
import pandas as pd
from utils import timeit

DETECTION_MODEL = 'yolo'
DETECTION_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4 # Non Maximum Supression threshold
SWAPRB = True
SCALE = 0.00392  # 1/255

with open(os.path.join('/model', DETECTION_MODEL, 'labels.json')) as json_data:
    CLASS_NAMES = json.load(json_data)


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = (
            [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()])
    return output_layers


def filter_yolo(chunk):
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


@timeit
class Detector():
    """Yolo object detection"""

    def __init__(self):
        self.model = cv2.dnn.readNetFromDarknet(
                os.path.join('/model', DETECTION_MODEL, 'yolov3.cfg'),
                os.path.join('/model', DETECTION_MODEL, 'yolov3.weights')
                )

    def prediction(self, image):
        """Inference"""
        blob = cv2.dnn.blobFromImage(image, SCALE, (416, 416), (0, 0, 0),
                                     swapRB=True, crop=False)
        self.model.setInput(blob)
        output = self.model.forward(get_output_layers(self.model))
        return output

    def create_df(self, result, image):
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
                    x['class_id'].apply(lambda x: CLASS_NAMES[str(int(x))])),
                label=lambda x: (
                    x.class_name + ': ' + (
                        x['confidence'].astype(str).str.slice(stop=4)
                        )
                    )
                )
        cols = ['x1', 'y1', 'w', 'h']
        #indices = cv2.dnn.NMSBoxes(boxes, confidences, THRESHOLD, NMS_THRESHOLD)
        indices = cv2.dnn.NMSBoxes(df[cols].values.tolist(), df['confidence'].tolist(), DETECTION_THRESHOLD, NMS_THRESHOLD)
        if len(indices) > 0:
            df = df.iloc[indices.flatten()]
        return df

