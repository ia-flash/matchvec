import os
from matchvec.process import predict_class, predict_objects
from importlib import import_module
Classifier = import_module('matchvec.classification_' + os.getenv('BACKEND')).Classifier


def test_apidoc(app):
    with app.test_client() as c:
        print("Testing doc")
        response = c.get('/matchvec/docs')
        assert response.status == '308 PERMANENT REDIRECT'


def test_class(img_clio4):

    print('Testing image', img_clio4.shape)
    res = predict_class(img_clio4)

    assert type(res) == list
    assert any(['CLIO' in vehicule['label'] for vehicule in res]), 'There is no clio in first predictions'


def test_class_prio(img_clio4):
    """Test if 'CLASSIFICATION_MODEL_PRIO' in os.environ
    """
    if 'CLASSIFICATION_MODEL_PRIO' in os.environ:

        print('Testing image', img_clio4.shape)
        res = predict_class(img_clio4)

        assert type(res) == list
        assert any(['CLIO' in vehicule['label_prio'] for vehicule in res]), 'There is no clio in first predictions'
    else:
        print('!!!! Test not executed, add CLASSIFICATION_MODEL_PRIO path !!!!!')

        
def test_object(img_clio4):
    res = predict_objects(img_clio4)
    assert type(res) == list
