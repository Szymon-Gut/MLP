import numpy as np
from sklearn.metrics import f1_score
from activation_functions import Softmax

class Loss:
    def calculate(self, y_true, y_pred):
        pass

    def derivative(self, y_true, y_pred):
        pass

class Mse(Loss):
    def calculate(self, y_true, y_pred):
        y_pred = np.reshape(y_pred, newshape=y_true.shape)
        return np.mean(np.square(y_true - y_pred))
    def derivative(self, y_true, y_pred):
        return y_pred - y_true
    
class Cross_entropy(Loss):
    def calculate(self, y_true, y_pred):
        y_pred = np.reshape(y_pred, newshape=y_true.shape)
        eps = 0.00000001
        return np.mean(
            np.where(y_true == 1, -np.log(y_pred + eps), -np.log(1 - y_pred + eps)))

    def derivative(self, y_true, y_pred):
        return (1 - y_true) / (1 - y_pred) - y_true / y_pred
    

class F_score(Loss):
    def calculate(self, y_true, y_pred):
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_true, axis=1)
        return f1_score(y_true, y_pred, average='micro')
    def derivative(self, y_true, y_pred):
        pass

