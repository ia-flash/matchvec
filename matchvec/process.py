import os
import cv2
import logging
import numpy as np
from PIL import Image
from importlib import import_module
from itertools import combinations, product
from typing import List, Union
from matchvec.utils import timeit

assert os.environ['BACKEND'] in ['onnx','torch']

Detector = import_module('matchvec.' + os.getenv('DETECTION_MODEL') + '_detection').Detector
detector = Detector()

Classifier = import_module('matchvec.' + 'classification_' + os.getenv('BACKEND')).Classifier
classifier = Classifier(os.getenv('CLASSIFICATION_MODEL'))

if 'CLASSIFICATION_MODEL_PRIO' in os.environ:
    classifier_prio = Classifier(os.getenv('CLASSIFICATION_MODEL_PRIO'))

if 'ANONYM_MODEL' in os.environ:
    Detector_Anonym = import_module('matchvec.' + 'anonym_' + os.getenv('BACKEND')).Detector
    detector_anonym = Detector_Anonym()


logger = logging.getLogger(__name__)


DETECTION_IOU_THRESHOLD = float(os.getenv('DETECTION_IOU_THRESHOLD', 0.9))
DETECTION_SIZE_THRESHOLD = float(os.getenv('DETECTION_SIZE_THRESHOLD',0.002))


def IoU(boxA: dict, boxB: dict) -> float:
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
    surf_boxA = (boxA['x2'] - boxA['x1']) * (boxA['y2'] - boxA['y1'])
    surf_boxB = (boxB['x2'] - boxB['x1']) * (boxB['y2'] - boxB['y1'])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(surf_boxA + surf_boxB - interArea)

    # return the intersection over union value
    return iou


def filter_by_size(df: List[dict], image: np.ndarray) -> dict:
    """Filter box too small

    Args:
        df: Detected boxes
        image: Image used for detection

    Returns:
        df: Filtered boxes
    """
    height, width = image.shape[:-1]
    surf = width * height
    df_filtered = []
    for i, row in enumerate(df):
        mean_x = (row['x1'] + row['x2'])/ 2.
        mean_y = (row['y1'] + row['y2'])/ 2.
        dist = ((width/2 - mean_x) ** 2 + (height/2 - mean_y)**2)**(1./2)
        surf_box = (row['x2'] - row['x1']) * (row['y2'] - row['y1'])
        if surf_box >= DETECTION_SIZE_THRESHOLD * surf:
            df_filtered.append(df[i])

    return df_filtered


def filter_by_iou(df: dict) -> dict:
    """Filter box of car and truck when IoU>DETECTION_IOU_THRESHOLD
    If a car and a truck overlap, take in priority the car box!
    Args:
        df: Detected boxes

    Returns:
        df: Filtered boxes
    """
    df_class = {'car': [], 'truck':[] }
    for i, row in enumerate(df):
        df[i].update(dict(id=i))
        if row['class_name'] in ['car', 'truck']:
            df_class[row['class_name']].append(row)
    prod_class = combinations(df_class.items(), 2)
    for (class_a, df_a_group), (class_b, df_b_group) in prod_class:
        for df_a, df_b in product(df_a_group, df_b_group):
            iou = IoU(df_a, df_b)
            if iou > DETECTION_IOU_THRESHOLD:
                if class_a == 'truck':
                    df.pop(df_a['id'])
                elif class_b == 'truck':
                    df.pop(df_b['id'])
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

    return [dict((k, row[k]) for k in cols) for row in df]


@timeit
def predict_class(img: np.ndarray) -> List[Union[str, float]]:
    """Classficate image

    Args:
        img: Image to make inference

    Returns:
        result: Predictions
    """
    #cv2.imwrite('/app/img.jpg',img)
    logger.debug("Détection des objets")
    result = detector.prediction(img)
    df = detector.create_df(result, img)
    #print(df)

    # Filter by class
    df = [row for row in df if row['class_name'] in ['car', 'truck']]
    df = filter_by_size(df, img)
    df = filter_by_iou(df)

    #Image.fromarray(img).convert('RGB').save('/app/debug/classif_input.jpg')
    selected_boxes = list(
            zip(
                [Image.fromarray(img)]*len(df),
                [[row[col] for col in ['x1', 'y1', 'x2', 'y2']] for row in df]
                )
            )

    # Selected box
    res = list()
    cols = ['x1', 'y1', 'x2', 'y2', 'class_name','confidence']

    if len(selected_boxes) > 0:
        logger.debug("Classification de %d objets", len(selected_boxes))
        pred, prob = classifier.prediction(selected_boxes)
        for i, obj in enumerate([dict((k, row[k]) for k in cols) for row in df]):
            obj.update({
                            "brand_model_classif": {
                                "pred": pred[i],
                                "prob": prob[i],
                                "label": pred[i][0] + ": " + str(round(prob[i][0],4))
                                }
                            })
            res += [obj]

        if 'CLASSIFICATION_MODEL_PRIO' in os.environ:
            pred_prio, prob_prio = classifier_prio.prediction(selected_boxes)
            for i, obj in enumerate(res):
                obj.update({
                                "prio_classif": {
                                    "pred": pred_prio[i],
                                    "prob": prob_prio[i],
                                    "label": pred_prio[i][0] + ": " + str(round(prob_prio[i][0],4))
                                    }
                                })
                res[i] = obj
    return res


@timeit
def predict_anonym(img: np.ndarray) -> List[Union[str, float]]:
    """Take an image and return sensible parts

    Args:
        img: Image to make inference

    Returns:
        result: Predictions
    """
    result = detector_anonym.prediction(img)
    df = detector_anonym.create_df(result)

    df = detector_anonym.detect_band(df, img)
    if len(df) > 0:
        cols = ['x1', 'y1', 'x2', 'y2', 'class_name', 'confidence', 'label']
        return [dict((k, row[k]) for k in cols) for row in df]

    else:
        return None

if __name__ == '__main__':
    img = cv2.imread('tests/clio-peugeot.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Respect matchvec input format
    print(img.shape)
    #res = predict_objects(img)
    #res = predict_class(img)
    res = predict_anonym(img)
    print(res)
    #test_app()
    #test_app_multiple()
