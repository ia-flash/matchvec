import os
import cv2
import numpy as np
from flask import Flask, send_from_directory, request, Blueprint, url_for
from flask_restplus import Resource, Api, reqparse, fields
from process import predict_class, predict_objects
from werkzeug.datastructures import FileStorage
from urllib.request import urlopen

app = Flask(__name__)
app.config.SWAGGER_UI_DOC_EXPANSION = 'list'
app.config.SWAGGER_UI_OPERATION_ID = True
app.config.SWAGGER_UI_REQUEST_DURATION = True


##########################
#  Documentation Sphinx  #
##########################

blueprint_doc = Blueprint('documentation', __name__,
                          static_folder='../docs/build/html/_static',
                          url_prefix='/matchvec/docs')


@blueprint_doc.route('/', defaults={'filename': 'index.html'})
@blueprint_doc.route('/<path:filename>')
def show_pages(filename):
    return send_from_directory('../docs/build/html', filename)


app.register_blueprint(blueprint_doc)

#################
#  API SWAGGER  #
#################


class Custom_API(Api):
    @property
    def specs_url(self):
        '''
        The Swagger specifications absolute url (ie. `swagger.json`)

        :rtype: str
        '''
        return url_for(self.endpoint('specs'), _external=False)


blueprint = Blueprint('api', __name__, url_prefix='/matchvec')
api = Custom_API(blueprint, doc='/swagger', version='1.0', title='IA Flash',
                 description='Classification marque et modèle')
app.register_blueprint(blueprint)


parser = reqparse.RequestParser()
parser.add_argument('url', type=str, location='form', help='Image URL in jpg format. URL must end with jpg.')
parser.add_argument('image', type=FileStorage, location='files', help='Image saved locally. Multiple images are allowed.')


ObjectDetectionOutput = api.model('ObjectDetectionOutput', {
    "x1": fields.Integer(description='X1', min=0, example=10),
    "y1": fields.Integer(description='Y1', min=0, example=10),
    "x2": fields.Integer(description='X2', min=0, example=200),
    "y2": fields.Integer(description='Y2', min=0, example=200),
    "class_name": fields.String(description='Matched model', example='car'),
    "confidence": fields.Float(description='Detection confidence', min=0, max=1, example=0.95),
    "label": fields.String(description='Label for visualization', example='car: 0.99'),
    })


ClassificationOutput = api.inherit('ClassificationOutput', ObjectDetectionOutput, {
    'pred': fields.List(fields.String(description='oiuioi', example=[0.5462563633918762, 0.07783588021993637, 0.047950416803359985, 0.041797831654548645, 0.03768396005034447])),
    'prob': fields.List(fields.Float(description='iii', example=["PEUGEOT 207", "CITROEN C3 PICASSO", "NISSAN MICRA", "CITROEN XSARA PICASSO", "RENAULT KANGOO"]))
})


@api.route('/object_detection')
class ObjectDetection(Resource):
    """Docstring for MyClass. """

    @api.expect(parser)
    @api.marshal_with(ObjectDetectionOutput, mask=None)
    def post(self):
        """Object detection

        Image can be loaded either by using an internet URL in the url field or
        by using a local stored image in the image field
        """
        images = request.files.getlist('image', None)
        url = request.form.get('url', None)
        res = list()
        if url:
            try:
                resp = urlopen(url)
                img = np.asarray(bytearray(resp.read()), dtype="uint8")
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                res.append(predict_objects(img))
            except Exception as e:
                print(url)
                print(e)
        if images:
            for i in range(len(images)):
                nparr = np.frombuffer(images[i].read(), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                res.append(predict_objects(img))
        return res


@api.route('/predict')
class ClassPrediction(Resource):
    """Predict vehicule class"""

    @api.expect(parser)
    @api.marshal_with(ClassificationOutput, mask=None)
    def post(self):
        """Brand and model classifcation

        Image can be loaded either by using an internet URL in the url field or
        by using a local stored image in the image field
        """
        images = request.files.getlist('image')
        url = request.form.get('url', None)

        res = list()
        if url:
            try:
                resp = urlopen(url)
                img = np.asarray(bytearray(resp.read()), dtype="uint8")
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                res.append(predict_class(img))
            except Exception as e:
                print(url)
                print(e)
        if images:
            for i in range(len(images)):
                nparr = np.frombuffer(images[i].read(), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
                res.append(predict_class(img))
        return res


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=bool(os.getenv('DEBUG')))
