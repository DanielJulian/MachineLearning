import cv2
from flask import Flask, request, render_template, jsonify, session
from tensorflow import Graph, Session
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
from Flask.DigitPrediction.functions.ocr import perform_ocr
from random import randint

app = Flask(__name__)

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/static/media'
ALLOWED_EXTENSIONS = set(['png', 'jpg'])
app.secret_key = '12i54m3a4n826ol'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def index(name=None):
    return render_template('index.html', name=name)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload_file', methods=['POST'])
def upload_file():
    session['counter'] = 0
    # session['cropped_filenames'] = ""

    file = request.files['attachment']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        session['image_filename'] = filename
        file.save(app.config['UPLOAD_FOLDER'] + "/" + filename)
        img = cv2.imread(app.config['UPLOAD_FOLDER'] + "/" + filename)
        return jsonify({'result': 'ok', 'image_url': app.config['UPLOAD_FOLDER'] + "/" + filename, 'height': img.shape[0], 'width': img.shape[1]})
    return jsonify({'result': 'Invalid File. Only .jpg and .png allowed'})


@app.route('/save_cropped_image', methods=['POST'])
def save_cropped_image():
    session['counter'] = session.get('counter') + 1
    counter = session.get('counter')
    # Obtengo los parametros
    x1 = request.form['x1']
    x2 = request.form['x2']
    y1 = request.form['y1']
    y2 = request.form['y2']
    # Abro la imagen
    im = cv2.imread(app.config['UPLOAD_FOLDER'] + "/" + session.get('image_filename'), 0)

    # Recorto el area de interes que se seleccionó
    im_cropped = im[int(y1):int(y2), int(x1):int(x2)]
    # La guardo.
    randomint = str(randint(0, 900000))
    cv2.imwrite(app.config['UPLOAD_FOLDER'] + '/cropped_' + str(counter) + "_" + randomint + '.jpg', im_cropped)
    # session['cropped_filenames'] = session.get('cropped_filenames') + (',cropped_' + str(counter) + '.jpg')
    # Preproceso la Imagen, e identifico el/los numero/s utilizando la red neuronal
    digits = perform_ocr(im_cropped, model_dict)
    return jsonify({'digits': digits, 'counter': counter, 'image_url': app.config['UPLOAD_FOLDER'] + '/cropped_' + str(counter) + "_" + randomint + '.jpg'})


def load_saved_model():
    path_modelos = ['/functions/mnist_model.h5', '/functions/mnist_model_2.h5', '/functions/mnist_model_3.h5']
    global model_dict
    model_dict = {}
    for index, model_path in enumerate(path_modelos):
        graph = Graph()
        with graph.as_default():
            session = Session()
            with session.as_default():
                model = load_model(os.path.dirname(os.path.abspath(__file__)) + model_path)
                model_dict[index] = dict(graph=graph, session=session, model=model)


# Borro todo el contenido de la carpeta media
fileList = os.listdir(os.path.dirname(os.path.abspath(__file__)) + "/static/media")
for fileName in fileList:
    os.remove(os.path.dirname(os.path.abspath(__file__)) + "/static/media/" + fileName)

load_saved_model()
app.run(host='0.0.0.0')

