import os
import cv2
import logging
import numpy as np
import pandas as pd
from PIL import Image
from importlib import import_module
from itertools import combinations, product
from typing import List, Union
from matchvec.utils import timeit

assert os.environ['BACKEND'] in ['onnx','torch']

Detector = import_module('matchvec.' + os.getenv('DETECTION_MODEL') + '_detection').Detector
detector = Detector()

Classifier = import_module('matchvec.' + 'classification_' + os.getenv('BACKEND')).Classifier
classifier = Classifier()


level = logging.DEBUG
logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
        )
logger = logging.getLogger(__name__)


DETECTION_IOU_THRESHOLD = 0.9
DETECTION_SIZE_THRESHOLD = 0.01


def IoU(boxA: pd.Series, boxB: pd.Series) -> float:
    """ Calculate IoU

    Args:
        boxA: Bounding box A
        boxB: Bounding box B
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA['x1'], boxB['x1'])
    yA = max(boxA['y1'], boxB['y1'])
    xB = min(boxA['x2'], boxB['x2'])
    yB = min(boxA['y2'], boxB['y2'])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxA['surf_box'] + boxB['surf_box'] - interArea)

    # return the intersection over union value
    return iou


def filter_by_size(df: pd.DataFrame, image: np.ndarray) -> pd.DataFrame:
    """Filter box too small

    Args:
        df: Detected boxes
        image: Image used for detection

    Returns:
        df: Filtered boxes
    """
    height, width = image.shape[:-1]
    surf = width * height
    df = df.assign(
            mean_x=lambda x: x[['x1', 'x2']].mean(axis=1),
            mean_y=lambda x: x[['y1', 'y2']].mean(axis=1),
            dist=lambda x: (
                ((width/2 - x['mean_x']) ** 2 +
                    (height/2 - x['mean_y'])**2).pow(1./2)),
            surf_box=lambda x: (x['x2'] - x['x1']) * (x['y2'] - x['y1']),
            surf_ratio=lambda x: x['surf_box'] / surf
            )
    df = df[(df['surf_ratio'] > DETECTION_SIZE_THRESHOLD)]
    return df


def filter_by_iou(df: pd.DataFrame) -> pd.DataFrame:
    """Filter box of car and truck when IoU>DETECTION_IOU_THRESHOLD

    Args:
        df: Detected boxes

    Returns:
        df: Filtered boxes
    """
    df['surf_box'] = (df['x2'] - df['x1']) * (df['y2'] - df['y1'])
    df_class = df[df['class_name'].isin(['car', 'truck'])].groupby('class_name')
    prod_class = combinations(df_class, 2)
    id_to_drop: List = list()
    for (class_a, df_a), (class_b, df_b) in prod_class:
        for (id1, vec1), (id2, vec2) in product(df_a.iterrows(), df_b.iterrows()):
            iou = IoU(vec1, vec2)
            if iou > DETECTION_IOU_THRESHOLD:
                # print('drop truck')
                if class_a == 'truck':
                    id_to_drop += [id1]
                elif class_b == 'truck':
                    id_to_drop += [id2]
    # drop trucks overlapping car
    df = df.drop(id_to_drop)
    return df


@timeit
def predict_objects(img: np.ndarray) -> List[Union[str, float]]:
    """Object detection

    Args:
        img: Image to make inference

    Returns:
        result: Predictions
    """
    result = detector.prediction(img)
    df = detector.create_df(result, img)

    df = filter_by_size(df, img)

    df = filter_by_iou(df)

    cols = ['x1', 'y1', 'x2', 'y2', 'class_name', 'confidence', 'label']
    return df[cols].to_dict(orient='records')


@timeit
def predict_class(img: np.ndarray) -> List[Union[str, float]]:
    """Classficate image

    Args:
        img: Image to make inference

    Returns:
        result: Predictions
    """
    #cv2.imwrite('/app/img.jpg',img)

    result = detector.prediction(img)
    df = detector.create_df(result, img)

    # Filter by class
    df = df[(df['class_name'] == 'car') | (df['class_name'] == 'truck')]
    #Image.fromarray(img).convert('RGB').save('/app/debug/classif_input.jpg')
    selected_boxes = list(
            zip(
                [Image.fromarray(img)]*len(df),
                df[['x1', 'y1', 'x2', 'y2']].values.tolist()
                )
            )

    # Selected box
    if len(selected_boxes) > 0:

        pred, prob = classifier.prediction(selected_boxes)
        df = df.assign(
                pred=pred,
                prob=prob,
                label=lambda x: (
                    x['pred'].apply(lambda x: x[0]) +
                    ": " + (
                        x['prob'].apply(lambda x: x[0])
                        .astype(str).str.slice(stop=4)
                        )
                    )
                )
        cols = ['x1', 'y1', 'x2', 'y2', 'pred', 'prob', 'class_name',
                'confidence', 'label']
        return df[cols].to_dict(orient='records')

    else:
        return list()


if __name__ == '__main__':
    img = cv2.imread('clio-punto-megane.jpg')
    #img = cv2.imread('image.jpg')
    print(img.shape)
    #res = predict_objects(img)
    res = predict_class(img)
    print(res)
    #test_app()
    #test_app_multiple()
