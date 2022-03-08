# import library
from crypt import methods
from pyexpat import model
from flask import Flask, render_template, request
from flask_restful import Resource, Api
from flask_cors import CORS
import os
import librosa
import numpy as np
import librosa.feature
import pickle
from models.knn import KNN
# init object flask
app = Flask(__name__)

# init object flask restfull
api = Api(app)

# init cors
CORS(app)

res = {}
model = pickle.load(open("models/model-vokal-a.pkl", 'rb'))


@app.route("/")
def landing():
    return render_template("/index.html")

@app.route("/data", methods=["GET", "POST"])
def coba():
    if request.method == "GET":
        return res
    if request.method == 'POST':
        save_path = os.path.join("audio/", "temp.wav")
        request.files['audio_data'].save(save_path)
        data = prediction()
        return data

def prediction():
    global model
    y, sr = librosa.load("audio/temp.wav")
    mfcc = np.array(getMFCC(y))
    new_mfcc = np.array(mfcc)
    X = np.reshape(new_mfcc,(1, new_mfcc.size))
    res = model.predict(X)[0]
    return res[1]

def getMFCC(f):
    mfcc = librosa.feature.mfcc(y=f, n_mfcc = 13)
    return [np.ndarray.flatten(mfcc)][0]

if __name__ == "__main__":
    app.run(debug=True, port = int(os.environ.get('PORT', 5000)))
    app.run()
