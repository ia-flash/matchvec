import sys
import cv2
import unittest

sys.path.append('./matchvec')
from app import app
from process import predict_class, predict_objects

class TestFileFail(unittest.TestCase):
    def test_apidoc(self):
        with app.test_client() as c:
            print("Testing doc")
            response = c.get('/api/doc')
            self.assertEqual('200 OK', response.status)

    def test_class(self):
        img = cv2.imread('tests/clio-peugeot.jpg')
        print('Testing image', img.shape)
        res = predict_class(img)
        self.assertIsInstance(res, list)

    def test_object(self):
        img = cv2.imread('tests/clio-peugeot.jpg')
        print('Testing image', img.shape)
        res = predict_objects(img)
        self.assertIsInstance(res, list)


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
    unittest.main()
