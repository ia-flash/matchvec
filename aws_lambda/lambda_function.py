from os import listdir, path, getenv
import json
import numpy as np
import cv2
import onnxruntime
from PIL import Image
import base64
from matchvec import predict_class, predict_objects
import io, re
import binascii

from requests_toolbelt.multipart import decoder
from urllib.request import urlopen

pattern = re.compile('(?<=form-data; name=").*?(?=")')


def lambda_handler_classification(event, context):
    print("ENV", getenv('BACKEND'))
    print("ENV", getenv('DETECTION_THRESHOLD'))
    print("LISTDIR", listdir('/tmp'))
    res = list()
    assert event.get('httpMethod') == 'POST'
    try :
        event['body'] = base64.b64decode(event['body'])
    except :
         return {
        'statusCode': 400,
        'body': json.dumps(res)
        }

    if event['path'] == '/predict' :
        infer_func = predict_class
    elif event['path'] == '/object_detection' :
        infer_func = predict_objects
    else:
         return {
        'statusCode': 404,
        'body': json.dumps(res)
        }

    content_type = event.get('headers', {"content-type" : ''}).get('content-type')
    if 'multipart/form-data' in content_type  :

        # convert to bytes if need
        if type(event['body']) is str:
            event['body'] = bytes(event['body'],'utf-8')

        multipart_data = decoder.MultipartDecoder(event['body'], content_type)
        for part in multipart_data.parts:
            #
            content_disposition = part.headers.get(b'Content-Disposition',b'').decode('utf-8')
            search_field = pattern.search(content_disposition)
            #import pdb; pdb.set_trace()
            if search_field :
                if search_field.group(0) == 'image' :
                    try:
                        img_io = io.BytesIO(part.content)
                        img_io.seek(0)
                        img = Image.open(img_io)
                        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
                        res.append(infer_func(img))
                    except Exception as e:
                        print(e)
                        res.append([])

                elif search_field.group(0) == 'url' :
                    try:
                        resp = urlopen(part.content.decode('utf-8'))
                        img = np.asarray(bytearray(resp.read()), dtype="uint8")
                        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                        res.append(infer_func(img))
                    except Exception as e:
                        print(e)
                        res.append([])
                else :
                    print('Bad field name in form-data')

    return {
        'statusCode': 200,
        'body': json.dumps(res)
        }
