import numpy as np
import matplotlib.pyplot as plt

plt.xlim(-1, 1)
plt.ylim(-1, 1)
def graph(formula, style='r'):
    a, b, c = formula
    x = [-1, 1]
    y = [(a*xi+c)/(-b) for xi in x]
    plt.plot(x, y, style)

def extendOnes(data):
    """
    add padding ones for w0
    """
    ext = np.ones((data.shape[0], 1))
    return np.append(data, ext, 1)

def target_result(target, data):
    data = extendOnes(data)
    result = np.dot(data, target)
    result[result>=0] = 1
    result[result<0] = -1
    return result

def generate_points(N):
    return np.random.uniform(-1, 1, (N,2))

def generate_sample(N):
    data = generate_points(N)
    #generate two points for the target function
    linePs = generate_points(2)
    p = np.polyfit(linePs[:,0], linePs[:,1], 1)
    target_function = np.insert(p, 1, -1)

    dataExt = extendOnes(data)

    Y = np.dot(dataExt, target_function)
    Y[Y>=0] = 1
    Y[Y<0] = -1

    return (data, Y, target_function)

class PerceptronLearning:
    def __init__(self, trainData, label):
        self.model = np.zeros((trainData.shape[1] + 1))
        self.iter_time = 0
        self.train(trainData, label)
    def train(self, trainingData, label):
        data = extendOnes(trainingData)
        misclass = trainingData.shape[0]
        iter_time = 0
        while True:
            iter_time += 1
            pred = self._classify(data)
            comp = np.where(pred * label < 0)[0]
            misclass = len(comp)
            if len(comp) != 0:
                choice = np.random.choice(comp)
                self.model += label[choice] * data[choice].transpose()
            else:
                break
        #print("train completed\n iteration times: ", iter_time)
        self.iter_time = iter_time

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
