import json
import os
from threading import Thread

from flask import abort, Flask, jsonify, render_template, \
                  request, Response, make_response

import modelling


MODEL_NAME = "10k.model"  # Replace this with you model filename
DATA_FILE = "data/first10k.csv"  # Replace this with your data filename

app = Flask(__name__)
_model = modelling.load_model(MODEL_NAME)
_is_training = False


def _train_new_model(data_file, model_name):
    global _model
    global _is_training

    _is_training = True
    data = modelling.load_data(data_file)
    model = modelling.train_model(data)
    modelling.save_model(model, model_name)
    _model = model  # swap new model into use
    _is_training = False


@app.route("/")
def index():
    return "Use the API"


@app.route("/echo", methods=['GET', 'POST'])
def echo():
    print('Content-Type:: {}'.format(request.headers['Content-Type']))
    print('Request: {}'.format(request))
    print('Request Payload: \n{}'.format(request.json))
    return json.dumps(request.json)


@app.route('/api/predict', methods=['POST'])
def predict():
    global _model

    if request.headers['Content-Type'] == 'application/json':
        article = request.json
        article_text = article['article_text']
        del article['article_text']
        prediction = modelling.predict(_model, article_text)
        article['negative_score'] = prediction[0]
        article['positive_score'] = prediction[1]
        return json.dumps(article)
    else:
        abort(400)  # bad request


@app.route('/api/train', methods=['POST'])
def train():
    if _is_training:
        return json.dumps({'error': 'Training already in progress'})
    else:
        thread = Thread(target=_train_new_model, args=(DATA_FILE, MODEL_NAME))
        thread.start()
        return json.dumps({'ok': 'Training started'})


if __name__ == "__main__":
    app.run(debug=False)
