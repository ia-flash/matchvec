"""Classification Marque Modèle"""
import cv2
import io
import os
import json
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Dict
from utils import timeit, logger
import onnxruntime
from PIL import Image
from BaseModel import BaseModel

CLASSIFICATION_MODEL = os.getenv('CLASSIFICATION_MODEL')

# Get label

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=1),axis=1))
    return  e_x / np.expand_dims(e_x.sum(axis=1),axis=1)



class Classifier(BaseModel):
    """Classifier for marque et modèle

    Classifies images using a pretrained model.
    """

    @timeit
    def __init__(self):
        self.files = [ 'classifcation_model', 'idx_to_class.json']
        dst_path = os.path.join(
            os.environ['BASE_MODEL_PATH'], CLASSIFICATION_MODEL)
        src_path = os.path.join('model', CLASSIFICATION_MODEL)

        self.download_model_folder(dst_path, src_path)

        self.session = onnxruntime.InferenceSession(dst_path,"classifcation_model.onnx"))
        self.output_name = self.session.get_outputs()[0].name
        self.input_name = self.session.get_inputs()[0].name

        filename = os.path.join(dst_path,  'idx_to_class.json')
        with open(filename) as json_data:
            self.all_categories = json.load(json_data)
            self.class_number = len(self.all_categories)


    def prediction(self, selected_boxes: Tuple[np.ndarray, List[float]]):
        """Inference in image

        1. Crops, normalize and transforms the image to tensor
        2. The image is forwarded to the resnet model
        3. The results are concatenated

        Args:
            selected_boxes: Contains a List of Tuple with the image and coordinates of the crop.

        Returns:
            (final_pred, final_prob): The result is two lists with the top 5 class prediction and the probabilities
        """

        X = list()
        ind = 0
        for img, boxes in selected_boxes:
            logger.debug('img')
            logger.debug(img.size)
            logger.debug('MIDE')
            logger.debug(img.mode)
            #detect_faces(np.array(img.crop(boxes)), ind)
            ind += 1
            #img = np.array(img.crop(boxes).resize((224, 224), Image.BILINEAR)).reshape(3, 224, 224).astype(np.float32)
            img = np.moveaxis(np.array(img.crop(boxes).resize((224, 224),
                Image.BILINEAR)), 2, 0).astype(np.float32)
            #img = np.array(Image.open(io.BytesIO(img.crop(boxes).resize((224, 224), Image.BILINEAR).tobytes())))
            #img = img.reshape(3, 224, 224).astype(np.float32)
            #logger.debug()
            img /= 255
            img -= np.array([0.485, 0.456, 0.406])[:, None, None]
            img /= np.array([0.229, 0.224, 0.225])[:, None, None]
            logger.debug(img.shape)
            logger.debug(type(img))

            logger.debug('box')
            logger.debug(boxes)
            X.append(img)

        res = self.session.run([self.output_name], {self.input_name: np.array(X)})
        #res = self.session.run([self.output_name], {self.input_name: np.expand_dims(img,axis=0)})

        norm_output = softmax(res[0])
        #e_x = np.exp(res[0] - np.expand_dims(np.max(res[0], axis=1),axis=1))
        #norm_output =  e_x / np.expand_dims(e_x.sum(axis=1),axis=1)
        logger.debug(norm_output)

        pred = np.argmax(norm_output, axis=1)
        prob = np.max(norm_output, axis=1)

        final_pred = list([[self.all_categories[str(i)]] for i in pred])
        final_prob = list([[float(i)] for i in prob])

        return final_pred, final_prob
