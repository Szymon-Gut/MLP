import numpy as np

from layer import Layer
from activation_functions import Softmax
from metrics import Mse, Cross_entropy


class NN:
    def __init__(self, input_shape, neurons_num, activations, seed=123):
        self.batches = None
        self.y_test = None
        self.x_test = None
        self.y_train = None
        self.x_train = None
        self.delta_weights = None
        self.input_shape = input_shape
        self.layers_num = len(neurons_num)
        self.neurons_num = neurons_num
        self.activations = activations
        self.history = {'train': [], 'test': []}
        np.random.seed(seed)
        self._build()

    def _build(self):
        self.layers = []

        layer = Layer(shape=(self.input_shape[1], self.neurons_num[0]),
                      activation=self.activations[0])
        self.layers.append(layer)

        for i in range(1, self.layers_num):
            layer = Layer(
                shape=(self.layers[i - 1].shape[1], self.neurons_num[i]),
                activation=self.activations[i])
            self.layers.append(layer)

    def calculate_errors(self, y_true):

        errors = []
        last_error = self.loss.derivative(y_true, self.layers[-1].output)
        last_derivative = self.layers[-1].activation.derivative(
            self.layers[-1].weighted_input
        )
        error = self.layers[-1].activation.error(last_error, last_derivative)
        errors.append(error)
        for i in range(self.layers_num - 2, -1, -1):
            layer = self.layers[i]
            derivative = layer.activation.derivative(
                layer.weighted_input
            )
            errors.append(np.multiply(
                derivative, (self.layers[i + 1].weights @ errors[-1])))

        errors.reverse()
        return errors

    def propagate_backwards(self, y_true, x):
        delta = {'weights': [], 'biases': []}

        errors = self.calculate_errors(y_true)
        for i in range(self.layers_num - 1, 0, -1):
            delta['weights'].insert(0,
                                    -self.layers[i - 1].output @ errors[i].T)
            delta['biases'].insert(0, -errors[i])

        delta['weights'].insert(0,
                                -x @ errors[0].T)
        delta['biases'].insert(0, -errors[0])
        return delta

    def propagate_backwards_with_regularization(self, y_true, x, lambd=0.05):
        delta = {'weights': [], 'biases': []}

        errors = self.calculate_errors(y_true)

        for i in range(self.layers_num - 1, 0, -1):
            l1 = lambd * np.sign(self.layers[i].weights)
            delta['weights'].insert(0,
                                    -self.layers[i - 1].output @ errors[i].T - l1)
            delta['biases'].insert(0, -errors[i])

        l1 = lambd * np.sign(self.layers[0].weights)
        delta['weights'].insert(0,
                               -x @ errors[0].T - l1)
        delta['biases'].insert(0, -errors[0])
        return delta
    
    def convert_to_numpy_array(self, x_train, y_train, x_test, y_test):
        if x_test is None or y_test is None:
            return np.reshape(np.array(x_train), (-1, self.input_shape[1])), \
                   np.reshape(np.array(y_train), (-1, y_train.shape[1])), None, None
        else:
            return np.reshape(np.array(x_train), (-1, self.input_shape[1])), \
                   np.reshape(np.array(y_train), (-1, y_train.shape[1])), \
                   np.reshape(np.array(x_test), (-1, self.input_shape[1])), \
                   np.reshape(np.array(y_test), (-1, y_test.shape[1]))

    def update_layers(self):
        for i in range(self.layers_num):
            self.layers[i].update(self.delta_weights['weights'][i],
                                  self.delta_weights['biases'][i])

    def initialize_dict(self):
        d = {'weights': [], 'biases': []}
        for layer in self.layers:
            d['weights'].append(np.zeros(shape=layer.weights.shape))
            d['biases'].append(np.zeros(shape=layer.biases.shape))

        return d

    def sum_dicts(self, dict1, dict2, dict1_multiplier=1, dict2_multiplier=1):

        d_sum = self.initialize_dict()
        for i in range(self.layers_num):
            w_shape = dict1['weights'][i].shape
            d_sum['weights'][i] = dict1_multiplier * dict1['weights'][
                i] + dict2_multiplier * dict2['weights'][i].reshape(w_shape)
            d_sum['biases'][i] = dict1_multiplier * dict1['biases'][
                i] + dict2_multiplier * dict2['biases'][i]
        return d_sum

    def print_results(self, epoch, metric_and_loss):
        print(f'Epoch number {epoch}/{self.n_epochs}')
        metric_name = self.metric.__class__.__name__
        if metric_and_loss:
            print(f'Loss on training set: '
                f'{round(self.loss.calculate(self.y_train, self.predict(self.x_train)), 2)}',
                end=' ')
        print(f'{metric_name} on training set: '
              f'{round(self.metric.calculate(self.y_train, self.predict(self.x_train)), 2)}', end=' ')
        if self.x_test is not None:
            if metric_and_loss:
                print(f', loss on test set: '
                    f'{round(self.loss.calculate(self.y_test, self.predict(self.x_test)), 2)}',
                    end=' ')
            print(f'{metric_name} on test set: '
                  f'{round(self.metric.calculate(self.y_test, self.predict(self.x_test)), 2)}')
            self.history['train'].append(round(self.metric.calculate(self.y_train, self.predict(self.x_train)), 2))
            self.history['test'].append(round(self.metric.calculate(self.y_test, self.predict(self.x_test)), 2))

    def generate_batches(self):
        np.random.shuffle(self.indices)
        indices_permutation = np.split(self.indices,
                                       [i * self.batch_size for i in
                                        range(1, self.n // self.batch_size)])
        return indices_permutation

    def fit(self, x_train, y_train, batch_size, n_epochs, learning_rate=0.003,
            x_test=None, y_test=None, loss=None, metric=None, verbose_step=10, stop_treshold=3, regularization_rate = 0, stop_action = True, metric_and_loss = False):

        self.x_train, self.y_train, self.x_test, self.y_test = self.convert_to_numpy_array(
            x_train, y_train, x_test, y_test)
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n = self.y_train.shape[0]
        self.indices = np.arange(self.n)
        self.loss = loss
        self.metric = metric

        epoch = 1
        while epoch <= self.n_epochs:
            batches = self.generate_batches()

            for batch in batches:
                self.delta_weights = self.initialize_dict()
                for j in range(self.batch_size):
                    idx = batch[j]
                    x = x_train[idx]
                    y = y_train[idx]
                    self.propagate_forward(np.reshape(x, (-1, 1)))

                    delta = self.propagate_backwards_with_regularization(
                        y_true=np.reshape(y, (-1, 1)),
                        x=np.reshape(x, (-1, 1)),
                        lambd=regularization_rate)

                    self.delta_weights = self.sum_dicts(
                        dict1=self.delta_weights, dict2=delta,
                        dict2_multiplier=self.learning_rate/self.batch_size)

                self.update_layers()

            if epoch % verbose_step == 0:
                self.print_results(epoch, metric_and_loss)
            epoch += 1
            
            if self.early_stop(epoch, verbose_step, stop_treshold, stop_action):
                break
            
    def early_stop(self, epoch, verbose_step, stop_treshold, stop_action, patience = 20):
        if not stop_action:
            return False
        
        if epoch  > verbose_step * (patience+1):
            num_epochs_without_improvement = 0
            loss_before = self.history['test'][-patience]
            metric_name = self.metric.__class__.__name__
            if metric_name == "Mse":
                for i in range(-1, -patience, -1):
                    current_loss = self.history['test'][i]
                    if current_loss >= loss_before + stop_treshold:
                        num_epochs_without_improvement+=1
            elif metric_name == "F_score":
                for i in range(-1, -patience, -1):
                    current_loss = self.history['test'][i]
                    if current_loss <= loss_before + stop_treshold:
                        num_epochs_without_improvement+=1
            if num_epochs_without_improvement == patience-1:
                return True
            else:
                return False
                
    def propagate_forward(self, x):
        for i in range(0, self.layers_num):
            x = self.layers[i].calculate(x)
            x = self.layers[i].activate(x)

        return x

    def predict(self, x):
        n = x.shape[0]
        for i in range(0, self.layers_num):
            x = x @ self.layers[i].weights + np.ones(shape=(n, 1)) @ self.layers[i].biases.reshape((1, -1))
            x = self.layers[i].activation.calculate(x)
        return x
