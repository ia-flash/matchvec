import os
import cv2
import numpy as np
import json
from flask import Flask, send_from_directory, request, Blueprint, url_for
from flask_restplus import Resource, Api, reqparse
from process import predict_class, predict_objects
from werkzeug.datastructures import FileStorage
from urllib.request import urlopen
from flask_jwt_extended import (
    JWTManager, jwt_required, create_access_token,
    get_jwt_identity
)


app = Flask(__name__)
app.config.SWAGGER_UI_DOC_EXPANSION = 'list'
app.config.SWAGGER_UI_OPERATION_ID = True
app.config.SWAGGER_UI_REQUEST_DURATION = True


# Setup the Flask-JWT-Extended extension
app.config['JWT_SECRET_KEY'] = 'super-secret'  # Change this!
jwt = JWTManager(app)

# Provide a method to create access tokens. The create_access_token()
# function is used to actually generate the token, and you can return
# it to the caller however you choose.
@app.route('/api/login', methods=['POST'])
def login():
    #if not request.is_json:
    #    return json.dumps({"msg": "Missing JSON in request"})

    username = request.form.get('username', None)
    password = request.form.get('password', None)
    if not username:
        return json.dumps({"msg": "Missing username parameter"})
    if not password:
        return json.dumps({"msg": "Missing password parameter"})

    if username != 'test' or password != 'test':
        return json.dumps({"msg": "Bad username or password"})

    # Identity can be any data that is json serializable
    access_token = create_access_token(identity=username)
    return json.dumps(dict(access_token=access_token))


# Protect a view with jwt_required, which requires a valid access token
# in the request to access.
@app.route('/api/protected', methods=['GET'])
@jwt_required
def protected():
    # Access the identity of the current user with get_jwt_identity
    current_user = get_jwt_identity()
    print(current_user)
    return json.dumps(dict(logged_in_as=current_user))


##########################
#  Documentation Sphinx  #
##########################

blueprint_doc = Blueprint('documentation', __name__,
                          static_folder='../docs/build/html/_static',
                          url_prefix='/docs')


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

blueprint = Blueprint('api', __name__, url_prefix='/api')
api = Custom_API(blueprint, doc='/doc', version='1.0', title='IA Flash',
          description='Classification marque et modèle')
app.register_blueprint(blueprint)


parser = reqparse.RequestParser()
parser.add_argument('url', type=str, location='form')
parser.add_argument('image', type=FileStorage, location='files')


@api.route('/object_detection')
class ObjectDetection(Resource):
    """Docstring for MyClass. """

    @api.expect(parser)
    def post(self):
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
    def post(self):
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
