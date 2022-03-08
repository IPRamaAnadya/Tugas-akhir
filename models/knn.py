from fastdtw import fastdtw

class KNN(object):
    def __init__(self, k=1):
        self.k = k
    def train(self, x, y):
        self.x = x
        self.y = y
    def predict(self, x):
        result = []
        for i in x:
            dist = []
            for j in range(len(self.x)):
                distance, path = fastdtw(i, self.x[j])
                dist.append([distance, self.y[j]])
            dist.sort(key=lambda row: (row[0]), reverse=False)
            result.append(dist[0])
        return result