"""Classification Marque Modèle"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from collections import OrderedDict
from typing import List, Tuple, Dict
from utils import timeit

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CLASSIFICATION_MODEL = os.getenv('CLASSIFICATION_MODEL')

# Get label
filename = os.path.join('/model', CLASSIFICATION_MODEL,  'idx_to_class.json')
with open(filename) as json_data:
    all_categories = json.load(json_data)
    CLASS_NUMBER = len(all_categories)

checkpoint = torch.load(
        os.path.join('/model', CLASSIFICATION_MODEL, 'model_best.pth.tar'),
        map_location='cpu'
        )
state_dict = checkpoint['state_dict']

new_state_dict: Dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove 'module.' of dataparallel
    new_state_dict[name] = v


class DatasetList(torch.utils.data.Dataset):
    """ Datalist generator

    Args:
        samples: Samples to use for inference
        transform: Transformation to be done to samples
        target_transform: Transformation done to the targets
    """
    def __init__(self, samples: Tuple[np.ndarray, List[float]],
                 transform=None, target_transform=None):
        self.samples = samples
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in dataframe"))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, coords = self.samples[index]  # coords [x1,y1,x2,y2]
        args = (image, coords)

        if self.transform is not None:
            sample = self.transform(args)
        return sample

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return "Dataset size {}".format(self.__len__())


class Crop(object):
    """Rescale the image in a sample to a given size.

    Args:
        params: Tuple containing the sample and coordinates. The image is cropped using the coordiantes.
    """
    def __call__(self, params):
        sample, coords = params
        # [coords[1]: coords[3], coords[0]: coords[2]]
        sample = sample.crop(coords)
        return sample


class Classifier(object):
    """Classifier for marque et modèle

    Classifies images using a pretrained model.
    """

    @timeit
    def __init__(self):
        """TODO: to be defined1. """
        self.classification_model = models.__dict__['resnet18'](pretrained=True)
        self.classification_model.fc = nn.Linear(512, CLASS_NUMBER)
        self.classification_model.load_state_dict(new_state_dict)
        self.classification_model.eval()

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
        # Crop and resize
        crop = Crop()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([
            crop,
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            normalize,
        ])

        val_loader = torch.utils.data.DataLoader(
                DatasetList(selected_boxes, transform=preprocess),
                batch_size=256, shuffle=False)

        final_pred: List = list()
        final_prob: List = list()
        for inp in val_loader:
            output = self.classification_model(inp)

            softmax = nn.Softmax(dim=1)
            norm_output = softmax(output)

            probs, preds = norm_output.topk(5, 1, True, True)
            pred = preds.data.cpu().tolist()
            pred_class = [
                    [all_categories[str(x)] for x in pred[i]]
                    for i in range(len(pred))
                    ]
            prob = probs.data.cpu().tolist()
            final_pred.extend(pred_class)
            final_prob.extend(prob)
        return final_pred, final_prob
