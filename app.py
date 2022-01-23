# import library
from pyexpat import model
from flask import Flask, render_template, request
from flask_restful import Resource, Api
from flask_cors import CORS
import os
import librosa
import numpy as np
import librosa.feature
import pickle
from model import Proses
# init object flask
app = Flask(__name__)

# init object flask restfull
api = Api(app)

# init cors
CORS(app)

res = {}
model = pickle.load(open("model.pkl", 'rb'))


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
        res["result"] = result(data)
        return res

def prediction():
    global model
    y, sr = librosa.load("audio/temp.wav")
    mfcc = np.array(getMFCC(y))
    new_mfcc = np.array(mfcc)
    X = np.reshape(new_mfcc,(1, new_mfcc.size))
    res = model.predict(X)[0]
    return res

def result(data):
    if(data == 0):
        return "Kumara"
    elif(data == 1):
        return "Vincky"
    elif(data == 2):
        return "Poke"
    elif(data == 3):
        return "Sinta"
    else:
        return "Tidak Dikenali"

def getMFCC(f):
    mfcc = librosa.feature.mfcc(y=f, n_mfcc = 13)
    return [np.ndarray.flatten(mfcc)][0]

if __name__ == "__main__":
    app.run(debug=True, port = int(os.environ.get('PORT', 5000)))