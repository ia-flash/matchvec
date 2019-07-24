import sys
import cv2
import unittest

sys.path.append('./matchvec')
from app import app
from process import predict_class, predict_objects
from classification import Classifier

class TestFileFail(unittest.TestCase):

    def test_apidoc(self):
        with app.test_client() as c:
            print("Testing doc")
            response = c.get('/matchvec/docs')
            self.assertEqual('308 PERMANENT REDIRECT', response.status)

    def test_class(self):
        img = cv2.imread('tests/clio4.jpg') # BGR
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)

        print('Testing image', img.shape)
        res = predict_class(img)

        self.assertIsInstance(res, list)
        assert any(['CLIO' in vehicule['label'] for vehicule in res]), 'There is no clio in first predictions'

    def test_object(self):
        img = cv2.imread('tests/clio4.jpg')
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        print('Testing image', img.shape)
        res = predict_objects(img)
        self.assertIsInstance(res, list)


    def test_export_classification_model(self):
        classifier = Classifier()
        classifier.export_model()


    #def test_request_prediction(self):
    #    with app.test_client() as c:
    #        #files = [
    #        #        ('image', open('tests/clio-peugeot.jpg', 'rb')),
    #        #        ]
    #        files = {'image': open('tests/clio-peugeot.jpg', 'rb')}
    #        resp = c.post(
    #            '/predict',
    #            #content_type='multipart/form-data',
    #            data=files
    #        )
    #        self.assertEqual(
    #            '200 OK',
    #            resp.status,
    #            )

if __name__ == '__main__':
    #unittest.main()
    tf = TestFileFail()
    tf.test_class()
