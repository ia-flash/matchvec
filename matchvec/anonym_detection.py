"""Yolo detection"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
import mmcv
from mmdet.models import build_detector
from mmdet.apis import inference_detector, init_detector
from mmdet.core import get_classes

from typing import List
from matchvec.utils import timeit, logger


DETECTION_MODEL = os.getenv('DETECTION_MODEL')
DETECTION_THRESHOLD = 0.15 #float(os.getenv('DETECTION_THRESHOLD'))
logger.debug(DETECTION_MODEL)
logger.debug(DETECTION_THRESHOLD)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.debug(device)

# Model checkpoint at https://github.com/open-mmlab/mmdetection/blob/master/MODEL_ZOO.md
modele = dict(conf="retinanet_r50_fpn_1x_anonym",
           checkpoint="latest")


def cut_down(img, width, height):
    """ find black frame at the bottom
     Return y2
     """
    zeros = np.asarray(img)[:,
         (int(width/4), int(1 * width/3), int(width/2),(int(2*width/3))),
         :].sum(axis=1).sum(axis=1)

    row0 = np.nonzero(zeros == 0)
    if len(row0[0]) > 0:
        minumun = np.min(row0) - 1
        return width,  minumun
    else:
        return width, height

def det_bboxes(bboxes,
              labels,
              class_names=None,
              score_thr=0,
              ):
    """Save bboxes and class labels (with scores) on an image.
    Args:
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.

    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    to_save = []

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)

        to_save.append({'x1':bbox_int[0],'y1':bbox_int[1],
                        'x2':bbox_int[2],'y2':bbox_int[3],
                        'class_name':label_text,'confidence':bbox[-1]})

    return to_save


def save_result(result,
                dataset='coco',
                score_thr=0.3
                ):

    """Return list dict [{x1,x2,y1,y2,classe,score},...]
    Args:
        results:
        score_thr (float): Minimum score of bboxes to be shown.
    """
    class_names = get_classes(dataset)
    #class_names = ["person", "plate"]
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(result)

    return det_bboxes(
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr)


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
        config_file = os.path.join('/model', 'anonym_detection',
                f"{modele['conf']}.py")
        checkpoint_file = os.path.join(
                '/model',
                'anonym_detection/retinanet_r50_fpn_1x',
                f"{modele['checkpoint']}.pth")
        print(config_file, checkpoint_file)
        self.model = init_detector(config_file, checkpoint_file)

        #self.model.eval()
        #torch.cuda.set_device(0)
        #self.model.cuda(0)

    def prediction(self, image: np.ndarray) -> List[np.ndarray]:
        """ Inference

        Make inference

        Args:
            image: input image

        Returns:
            result: Yolo boxes from object detections
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = inference_detector(self.model, image)

        return result

    def batch_prediction(self, image: List[np.ndarray]) -> List[List[np.ndarray]]:
        """ Inference

        Make inference

        Args:
            image: input image

        Returns:
            result: Yolo boxes from object detections
        """
        result = []
        for one_image in image:
            one_result = self.prediction(one_image)
            result.append(one_result)
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
        res = save_result(result,
                    dataset='anonym',
                    score_thr=DETECTION_THRESHOLD)
        df = pd.DataFrame(res)
        if len(df) > 0:
            df['label'] = df['class_name'] + ' : ' + df['confidence'].astype(str).str.slice(stop=4)
        return df


    def batch_create_df(self, result: List[List[np.ndarray]],
                        image: List[np.ndarray]) -> List[pd.DataFrame]:
        """Filter predictions and create an output DataFrame

        Args:
            result: Result from prediction model
            image: Image where the inference has been made

        Returns:
            df: Onject detection predictions filtered
        """
        df = []
        for one_result, one_image in zip(result, image):
            one_df = self.create_df(one_result, one_image)
            df.append(one_df)

        return df

    def detect_band(self, df: pd.DataFrame,
                  image: np.ndarray) -> pd.DataFrame:
        """Filter predictions and create an output DataFrame

        Args:
            result: Result from prediction model
            image: Image where the inference has been made

        Returns:
            df: Onject detection predictions filtered
        """
        height, width = image.shape[:-1]
        new_width, new_height = cut_down(image, width, height)
        if (new_height != height):
            df_band = pd.DataFrame([{'x1': 0, 'y1': new_height, 'x2': width, 'y2': height, 'class_name': 'band', 'confidence': 1, 'label': 'band'}])
            df = pd.concat([df, df_band])
        return df

if __name__ == '__main__':
    detector = Detector()
