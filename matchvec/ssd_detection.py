"""SSD detection"""
import os
import cv2
import json
import pandas as pd

# DETECTION_MODEL = 'faster_rcnn_resnet101_coco_2018_01_28/'
DETECTION_MODEL = 'ssd_mobilenet_v2_coco_2018_03_29/'
DETECTION_THRESHOLD = 0.4
SWAPRB = True

with open(os.path.join('/model', DETECTION_MODEL, 'labels.json')) as json_data:
    CLASS_NAMES = json.load(json_data)


class Detector():
    """SSD Mobilenet object detection"""

    def __init__(self):
        self.model = cv2.dnn.readNetFromTensorflow(
                os.path.join(
                    '/model', DETECTION_MODEL, 'frozen_inference_graph.pb'),
                os.path.join(
                    '/model', DETECTION_MODEL, 'config.pbtxt')
        )

    def prediction(self, image):
        """Inference"""
        self.model.setInput(
                cv2.dnn.blobFromImage(
                    image, size=(300, 300),
                    swapRB=SWAPRB,
                    crop=False)
                )
        cvOut = self.model.forward()
        result = cvOut[0, 0, :, :]
        return result

    def create_df(self, result, image):
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
                    x['class_id'].apply(lambda y: CLASS_NAMES[str(int(y))]))
                )
        df = df[df['confidence'] > DETECTION_THRESHOLD]
        return df
