import pytest
import base64
import cv2

path_clio4 = "tests/clio4.jpg"

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
def img_clio4():
    img = cv2.imread(path_clio4) # BGR
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    yield img

@pytest.fixture
def event_clio4():
    with open(path_clio4, "rb") as imageFile:
        img = base64.b64encode(imageFile.read())
    event = {
            'image': img
            }

    yield event
