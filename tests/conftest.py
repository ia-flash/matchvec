import pytest
import base64
import cv2
from requests_toolbelt.multipart.encoder import MultipartEncoder
import base64

path_clio4 = "tests/clio4.jpg"
path_bmw = "tests/bmw.png"
url_clio3 = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Renault_Clio_III_20090527_front.JPG/800px-Renault_Clio_III_20090527_front.JPG"

@pytest.fixture
def app():
    from  matchvec.app import app
    yield app

@pytest.fixture
def file_clio4():

    def _file_clio4(nb=2):
        data = dict(image=[])
        for i in range(nb):
            data['image'].append(( open(path_clio4, "rb"), 'test.jpg') )
        return data

    yield _file_clio4

@pytest.fixture
def url_clio():
    yield url_clio3

@pytest.fixture
def img_clio4():
    img = cv2.imread(path_clio4) # BGR
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    yield img

@pytest.fixture
def apigateway_event():

    """
    mp_encoder = MultipartEncoder(
        fields={'field0': open("tests/binary1.dat", "rb"),
            'field1': open("tests/binary2.dat", "rb")}
    )
    """

    mp_encoder = MultipartEncoder(
        fields={'image': ('filename',open(path_bmw, "rb"),'image/png'),
                'url' : url_clio3
                })
    body = mp_encoder.to_string()
    print('form-data is :')
    print(body[:100])
    body = base64.b64encode(body)
    print('It is encoded in base64')

    event = dict(httpMethod = 'POST',
                 path = '/predict',
                 headers = {'content-type': mp_encoder.content_type},
                 body = body)


    yield event
