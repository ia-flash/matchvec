"""Classification Marque Modèle"""
import os
import json
import numpy as np
from typing import List, Tuple
from matchvec.utils import timeit
import onnxruntime
from PIL import Image
from matchvec.utils import timeit
from matchvec.BaseModel import BaseModel



# Get label
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    return e_x / np.expand_dims(e_x.sum(axis=1), axis=1)


class Classifier(BaseModel):
    """Classifier for marque et modèle

    Classifies images using a pretrained model.
    """

    @timeit
    def __init__(self, classification_model):
        self.files = ['classifcation_model_onnx1.5.onnx', 'idx_to_class.json']
        dst_path = os.path.join(
            os.environ['BASE_MODEL_PATH'], classification_model)
        src_path = classification_model

        self.download_model_folder(dst_path, src_path)

        self.session = onnxruntime.InferenceSession(os.path.join(dst_path, "classifcation_model_onnx1.5.onnx"))
        self.output_name = self.session.get_outputs()[0].name
        self.input_name = self.session.get_inputs()[0].name

        with open(os.path.join(dst_path, 'idx_to_class.json')) as json_data:
            self.class_name = json.load(json_data)

    def prediction(self, selected_boxes: Tuple[np.ndarray, List[float]]):
        """Inference in image

        1. Crops, normalize and transforms the image to tensor
        2. The image is forwarded to the resnet model
        3. The results are concatenated

        Args:
            selected_boxes: Contains a List of Tuple with the image and
            coordinates of the crop.

        Returns:
            (final_pred, final_prob): The result is two lists with the top 5
            class prediction and the probabilities
        """

        X = list()
        ind = 0
        for img, boxes in selected_boxes:
            #detect_faces(np.array(img.crop(boxes)), ind)
            ind += 1
            #img = np.array(img.crop(boxes).resize((224, 224), Image.BILINEAR)).reshape(3, 224, 224).astype(np.float32)
            img = np.moveaxis(np.array(img.crop(boxes).resize((224, 224),
                Image.BILINEAR)), 2, 0).astype(np.float32)
            #img = img.reshape(3, 224, 224).astype(np.float32)
            img /= 255
            img -= np.array([0.485, 0.456, 0.406])[:, None, None]
            img /= np.array([0.229, 0.224, 0.225])[:, None, None]

            X.append(img)
        res = self.session.run(None, {self.input_name: np.array(X)})

        norm_output = softmax(res[0])

        preds = np.argsort(-norm_output, axis=1)
        pred_top = [i[:3] for i in preds.tolist()]
        final_pred = [
                [self.class_name[str(x)] for x in pred]
                for pred in pred_top
                ]
        final_prob = [
                [norm_output[i].tolist()[x] for x in pred_top[i]]
                for i in range(len(pred_top))
                ]
        return final_pred, final_prob
