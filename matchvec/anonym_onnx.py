"""Yolo detection"""
import os
import cv2
import numpy as np
import pandas as pd
import onnxruntime

from typing import List
from matchvec.utils import timeit, logger
from matchvec.BaseModel import BaseModel


DETECTION_MODEL = os.getenv('ANONYM_MODEL')
DETECTION_THRESHOLD = 0.15 #float(os.getenv('DETECTION_THRESHOLD'))
logger.debug(DETECTION_MODEL)
logger.debug(DETECTION_THRESHOLD)

def imnormalize(img, mean, std, to_rgb=True):
    """Normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    img = img.copy().astype(np.float32)
    return imnormalize_(img, mean, std, to_rgb)



def imnormalize_(img, mean, std, to_rgb=True):
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img

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

def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        return [bboxes[labels == i, :] for i in range(num_classes)]

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
                score_thr=0.3
                ):

    """Return list dict [{x1,x2,y1,y2,classe,score},...]
    Args:
        results:
        score_thr (float): Minimum score of bboxes to be shown.
    """
    class_names = ('person', 'plate')
    bbox_result = bbox2result(result[0], result[1], len(class_names))
    #class_names = ["person", "plate"]
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]

    labels = np.concatenate(labels)
    bboxes = np.vstack(bbox_result)

    return det_bboxes(
        bboxes,
        labels,
        class_names=class_names,
        score_thr=score_thr)


class Detector(BaseModel):
    """
    Args:
        DETECTION_MODEL: Detection model to use
        DETECTION_THRESHOLD: Detection threshold
        NMS_THRESHOLD: Non Maximum Supression threshold (to remove overlapping boxes)
    """
    @timeit
    def __init__(self):
        self.files = ['latest_simplified.onnx']
        dst_path = os.path.join(
            os.environ['BASE_MODEL_PATH'], DETECTION_MODEL)
        src_path = DETECTION_MODEL
        self.download_model_folder(dst_path, src_path)

        self.session = onnxruntime.InferenceSession(os.path.join(dst_path, "latest_simplified.onnx"))
        self.output_name = self.session.get_outputs()[0].name
        self.input_name = self.session.get_inputs()[0].name

    def prediction(self, img: np.ndarray) -> List[np.ndarray]:
        """ Inference

        Make inference

        Args:
            img: input image

        Returns:
            result: boxes from object detections
        """
        # TODO : read mmdet config
        img = img.copy().astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        img = imnormalize(img, mean, std, to_rgb=False) # the image is in rgb
        img = np.expand_dims(np.moveaxis(img, -1, 0), 0).astype(np.float32)
        result = self.session.run(None, {'input' : img})
        result = [_[-1] for _ in result]
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

    def create_df(self, result: List[np.ndarray]) -> pd.DataFrame:
        """Filter predictions and create an output DataFrame

        Args:
            result: Result from prediction model

        Returns:
            df: Onject detection predictions filtered
        """
        res = save_result(result,
                    score_thr=DETECTION_THRESHOLD)

        df = pd.DataFrame(res)
        if len(df) > 0:
            df['label'] = df['class_name'] + ' : ' + df['confidence'].astype(str).str.slice(stop=4)
        return df


    def batch_create_df(self, result: List[List[np.ndarray]]) -> List[pd.DataFrame]:
        """Filter predictions and create an output DataFrame

        Args:
            result: Result from prediction model
            image: Image where the inference has been made

        Returns:
            df: Onject detection predictions filtered
        """
        df = []
        for one_result in result:
            one_df = self.create_df(one_result)
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
