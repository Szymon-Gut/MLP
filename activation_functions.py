import numpy as np


class ActivationFunction:
    def calculate(self, x):
        pass

    def derivative(self, x):
        pass

    def error(self, error, derivative):
        pass


class Relu(ActivationFunction):

    def calculate(self, x):
        return np.where(x > 0, x, 0)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)

    def error(self, error, derivative):
        return np.multiply(error, derivative)


class LeakyRelu(ActivationFunction):
    def calculate(self, x):
        return max(0.1 * x, x)

    def derivative(self, x, alpha=0.01):
        return alpha if x < 0 else 1

    def error(self, error, derivative):
        return np.multiply(error, derivative)


class Tanh(ActivationFunction):

    def calculate(self, x):
        counter = np.exp(x) - np.exp(-x)
        denominator = np.exp(x) + np.exp(-x)
        return counter / denominator

    def derivative(self, x):
        return 1 - np.square(self.calculate(x))

    def error(self, error, derivative):
        return np.multiply(error, derivative)


class Sigmoid(ActivationFunction):

    def calculate(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return np.exp(-x) / np.square((1 + np.exp(-x)))

    def error(self, error, derivative):
        return np.multiply(error, derivative)


class Linear(ActivationFunction):

    def calculate(self, x):
        return x

    def derivative(self, x):
        return np.ones(shape=(x.shape[-1], 1))

    def error(self, error, derivative):
        return np.multiply(error, derivative)


class Softmax(ActivationFunction):

    def calculate(self, x):
        if x.shape[1] == 1:
            return np.exp(x) / np.sum(np.exp(x))
        if x.shape[1] > 1:
            return np.exp(x) / (np.sum(np.exp(x), axis=1).reshape(
                (x.shape[0], 1)) @ np.ones((1, x.shape[1])))
  
    def derivative(self, x): 
        s = self.calculate(x).T
        a = np.eye(s.shape[-1])
        temp1 = np.zeros((s.shape[0], s.shape[1], s.shape[1]), dtype=np.float32)
        temp2 = np.zeros((s.shape[0], s.shape[1], s.shape[1]), dtype=np.float32)
        temp1 = np.einsum('ij,jk->ijk', s, a)
        temp2 = np.einsum('ij,ik->ijk', s, s)
        return (temp1 - temp2).T

    def error(self, error, derivative):
        derivative = derivative.reshape(derivative.shape[0], derivative.shape[1])
        return (derivative @ error).reshape((-1, 1))
