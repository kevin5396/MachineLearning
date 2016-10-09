import numpy as np
class LinearRegression:
    def __init__(self, trainData, label):
        self.model = np.zeros((trainData.shape[1] + 1))
        self.iter_time = 0
        self.train(trainData, label)
    def train(self, trainingData, label):
        data = extendOnes(trainingData)
        pseudo = np.dot(np.linalg.inv(np.dot(data.transpose(), data)), data.transpose())
        self.model = np.dot(pseudo, label)

    def modelParameter(self):
        return self.model

    def _classify(self, data):
        result = np.dot(data, self.model)
        result[result>=0] = 1
        result[result<0] = -1
        return result

    def classify(self, data):
        data = extendOnes(data)

        result = np.dot(data, self.model)
        result[result>=0] = 1
        result[result<0] = -1
        return result
