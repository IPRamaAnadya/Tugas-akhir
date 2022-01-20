#Import Library
from crypt import methods
from logging import debug
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
import os
import librosa
import numpy as np
import pandas as pd
import librosa.feature
import joblib
from model import Proses

# init object flask
app = Flask(__name__)

# init object flask restfull
api = Api(app)

# init cors
CORS(app)

#init var
identitas = {
            "nama":"I Putu Rama Anadya",
            "umur": 21
        }
secIdentitas = {}

# load model
model = joblib.load("model.sav")

@app.route("/")
def main():
    response = {"msg": "Hello World"}
    return response

@app.route("/api", methods = ['GET', 'POST'])
def second():
    if request.method == "GET":
        return identitas
    if request.method == "POST":
        nama = request.args.get("nama")
        umur = request.args.get("umur")
        identitas["nama"]= nama
        identitas["umur"]= umur
        response = {"msg":"Success"}
        return response

@app.route("/data", methods=["GET", "POST"])
def coba():
    if request.method == "GET":
        data = prediction()
        response = {"msg": result(data)}
        return response
    if request.method == 'POST':
        save_path = os.path.join("audio/", "temp.wav")
        request.files['music_file'].save(save_path)
        data = prediction()
        response = {"msg": result(data)}
        return response

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

def text(data):
    nama = "namaku "+ data
    return nama


if __name__ == "__main__":
    app.run(debug=True, port = int(os.environ.get('PORT', 5000)))
