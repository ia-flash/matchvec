from os import listdir, path, getenv
import json
import numpy as np
import cv2
import onnxruntime
from PIL import Image
import base64
from process import predict_class, predict_objects

#if path.isfile('/tmp/classification_model.onnx') != True:
if path.isfile('/tmp/classification_model.onnx') != True and False:
    import boto3
    print("Downloading..")
    s3 = boto3.resource('s3')
    myobject = s3.Object('iaflash', 'classifcation_model.onnx')
    myobject.download_file('/tmp/classifcation_model.onnx')
    print("Downloading ok")

def lambda_handler(event, context):

    res = list()
    data = event.get('body', None)
    if data:
        #  read encoded image
        imageString = base64.b64decode(data)

        #  convert binary data to numpy array
        nparr = np.frombuffer(imageString, np.uint8)

        #  let opencv decode image to correct format
        img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR);

        print("IMAGE", img)

        res.append(predict_class(img))

    return {
        'statusCode': 200,
        'body': json.dumps(res)
    }
