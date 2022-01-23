import numpy as np
import librosa
import librosa.feature
import librosa.display
import pandas as pd
from fastdtw import fastdtw
import pickle

class Proses:
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels
        self.treshold = self.get_dict()
    
    def get_dict(self):
        label = []
        for i in self.labels:
            if i not in label:
                label.append(i)
            else:
                pass
        return { x : [] for x in sorted(label)}
        
    def train(self):
        for key in self.treshold:
            q3 = []
            for i in range(len(self.samples)):
                temp = []
                for j in range(len(self.samples)):
                    if(self.labels[i]==key and self.labels[j]== key):
                        distance, path = fastdtw(self.samples[i],self.samples[j])
                        temp.append(distance)
                try:
                    q3.append(max(temp))
                except:
                    pass
            self.treshold[key].append(int(sum(q3)/len(q3)))
        print(self.treshold)
    
    def predict(self, data):
        res = []
        for d in data:
            dict_dist = self.get_dict()
            for key in self.treshold:
                temp = []
                for i in range(len(self.samples)):
                    if(self.labels[i]==key):
                        distance, path = fastdtw(d, self.samples[i])
                        temp.append(int(distance))
                    else:
                        pass
                dict_dist[key].append(int(sum(temp)/len(temp)))
            sort_orders = sorted(dict_dist.items(), key=lambda x: x[1])
            if(sort_orders[0][1][0] < self.treshold[sort_orders[0][0]][0]):
                res.append(sort_orders[0][0])
            else:
                res.append("unknown")
        return res
    
def getMFCC(f):
    mfcc = librosa.feature.mfcc(y=f, n_mfcc = 13)
    return [np.ndarray.flatten(mfcc)][0]

df_fitur = pd.read_csv("data.csv")
fitur = df_fitur.stack().groupby(level=0).apply(list).tolist()

df_label = df_fitur.pop("label")
label = df_label.values.tolist()

model = Proses(fitur, label)
model.train()

def testing(path):
    global model
    try:
        y, sr = librosa.load(path)
        mfcc = np.array(getMFCC(y))
        new_mfcc = np.array(mfcc)
        X = np.reshape(new_mfcc,(1, new_mfcc.size))
        res = model.predict(X)
        return res
    except Exception as e: print(e)
res = testing("sample/vincky0.wav")
print(res if(res != "unknown") else "Suara tidak dikenal")

filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))

def testing2(path):
    global loaded_model
    try:
        y, sr = librosa.load(path)
        mfcc = np.array(getMFCC(y))
        new_mfcc = np.array(mfcc)
        X = np.reshape(new_mfcc,(1, new_mfcc.size))
        res = loaded_model.predict(X)
        return res
    except Exception as e: print(e)
res = testing2("sample/vincky0.wav")
print(res if(res != "unknown") else "Suara tidak dikenal")

