from os import listdir, path, getenv
import json
import numpy as np
import cv2
import onnxruntime
from PIL import Image
import base64
#from process import predict_class, predict_objects
<<<<<<< HEAD
=======

from cgi import parse_header, parse_multipart
from io import BytesIO
>>>>>>> 5e242d3... WIP: add test get route

<<<<<<< HEAD
#if path.isfile('/tmp/classification_model.onnx') != True and False:
if path.isfile('/tmp/classification_model.onnx') != True:
    import boto3
    print("Downloading..")
    s3 = boto3.resource('s3')
    myobject = s3.Object('iaflash', 'classifcation_model.onnx')
    myobject.download_file('/tmp/classifcation_model.onnx')
    print("Downloading ok")

=======
>>>>>>> 06cd91b... Fix errors for classification and detection
def lambda_handler(event, context):
    print("ENV", getenv('BACKEND'))
    print("ENV", getenv('DETECTION_THRESHOLD'))
    print("LISTDIR", listdir('/tmp'))

    res = list()
    return {
        'statusCode': 200,
        'body': json.dumps(res)
    }


def lambda_handler_classification(event, context):
    print("ENV", getenv('BACKEND'))
    print("ENV", getenv('DETECTION_THRESHOLD'))
    print("LISTDIR", listdir('/tmp'))

    res = list()
    data = event.get('body', None)

    c_type, c_data = parse_header(event['headers']['Content-Type'])
    assert c_type == 'multipart/form-data'
    decoded_string = base64.b64decode(event['body'])
    form_data = parse_multipart(BytesIO(decoded_string), c_data)

    #print(data)
    #post_data = base64.b64decode(event['body'])
    #headers, data_2 = post_data.split('\r\n', 1)

    if data:
        print(type(data))
        print(data)


        #print(post_data)
        #print(headers)
        #print(data_2)


        #nparr = np.frombuffer(data.read(), np.uint8)
        #img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        #print(img.shape)

        #  read encoded image
        #imageString = base64.b64decode(data)
        ##  convert binary data to numpy array
        #nparr = np.frombuffer(imageString, np.uint8)
        ##  let opencv decode image to correct format
        #img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
        #res.append(predict_class(img))

    return {
        'statusCode': 200,
        'body': json.dumps(res)
    }
