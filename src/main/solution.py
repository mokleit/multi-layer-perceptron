import pickle
import numpy as np
import gzip
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import ParameterGrid

def one_hot(y, n_classes=10):
    return np.eye(n_classes)[y]

def load_mnist():
    data_file = gzip.open("mnist.pkl.gz", "rb")
    train_data, val_data, test_data = pickle.load(data_file, encoding="latin1")
    data_file.close()

    train_inputs = [np.reshape(x, (784, 1)) for x in train_data[0]]
    train_results = [one_hot(y, 10) for y in train_data[1]]
    train_data = np.array(train_inputs).reshape(-1, 784), np.array(train_results).reshape(-1, 10)

    val_inputs = [np.reshape(x, (784, 1)) for x in val_data[0]]
    val_results = [one_hot(y, 10) for y in val_data[1]]
    val_data = np.array(val_inputs).reshape(-1, 784), np.array(val_results).reshape(-1, 10)

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = list(zip(test_inputs, test_data[1]))

    return train_data, val_data, test_data

# train_data_, val_data_, test_data_ = load_mnist()

def weight_initialization(init, m, n):
    if init == 'zero':
        w = np.zeros((m, n))
    elif init == 'normal':
        w = np.random.normal(0, 1, (m,n))
    elif init == 'glorot':
        d = np.sqrt(6/(m+n))
        w = np.random.uniform(-d, d, (m,n))
    else:
        print('Unknown initialization method. Falling back to glorot init method.')
        d = np.sqrt(6/(m+n))
        w = np.random.uniform(-d, d, (m,n))
    return w


class NN(object):
    def __init__(self,
                 hidden_dims=(784, 256),
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=64,
                 seed=None,
                 activation="relu",
                 data=None
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = 'glorot'
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon
        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}
        self.train_losses = {'zero': [], 'normal': [], 'glorot': []}
        self.valid_losses = {'zero': [], 'normal': [], 'glorot': []}


        if data is None:
            # for testing, do NOT remove or modify
            self.train, self.valid, self.test = (
                (np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400))),
                (np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400))),
                (np.random.rand(400, 784), one_hot(np.random.randint(0, 10, 400)))
        )
        else:
            self.train, self.valid, self.test = data

    def initialize_weights(self, dims):        
        if self.seed is not None:
            np.random.seed(self.seed)
        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            self.weights[f"W{layer_n}"] = weight_initialization(self.init_method, all_dims[layer_n-1], all_dims[layer_n])
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

    def relu(self, x, grad=False):
        if grad:
            x = np.heaviside(x, 1/2)
        else:
            x = np.maximum(0, x)
        return x

    def sigmoid(self, x, grad=False):
        sig = 1/(1+np.exp(-x))
        if grad:
            x = sig*(1-sig)
        else:
            x = sig
        return x

    def tanh(self, x, grad=False):
        tan = np.tanh(x)
        if grad:
            x = 1 - tan**2
        else:
            x = tan
        return x

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            a = self.relu(x, grad)
        elif self.activation_str == "sigmoid":
            a = self.sigmoid(x, grad)
        elif self.activation_str == "tanh":
            a = self.tanh(x, grad)
        else:
            raise Exception("invalid")
        return a

    def softmax(self, x):
        if x.ndim == 2:
            exponent = np.exp(x-np.max(x, axis=1)[:, np.newaxis])
            soft_max = exponent / exponent.sum(axis=1)[:, np.newaxis]
        else:
            exponent = np.exp(x - np.max(x))
            soft_max = exponent / exponent.sum()
        return soft_max

    def forward(self, x):
        cache = {"Z0": x}
        # cache is a dictionnary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        for layer in range(1, self.n_hidden + 2):
            cache[f"A{layer}"] = np.dot(cache[f"Z{layer-1}"], self.weights[f"W{layer}"]) + self.weights[f"b{layer}"]
            cache[f"Z{layer}"] = self.softmax(cache[f"A{layer}"]) if layer == (self.n_hidden + 1) else self.activation(cache[f"A{layer}"])
        return cache

    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}
        # grads is a dictionnary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        grads[f"dA{self.n_hidden+1}"] = output - labels
        grads[f"dW{self.n_hidden+1}"] = np.dot(np.array(cache[f"Z{self.n_hidden}"].T), grads[f"dA{self.n_hidden+1}"])/self.batch_size
        grads[f"db{self.n_hidden+1}"] = np.sum(grads[f"dA{self.n_hidden+1}"], axis=0, keepdims=True)/self.batch_size
        for layer_n in range(self.n_hidden, 0, -1):
            grads[f"dZ{layer_n}"] = np.dot(grads[f"dA{layer_n+1}"], np.array(self.weights[f"W{layer_n+1}"].T))
            grads[f"dA{layer_n}"] = np.multiply(self.activation(cache[f"A{layer_n}"], True), grads[f"dZ{layer_n}"])
            grads[f"dW{layer_n}"] = np.dot(np.array(cache[f"Z{layer_n-1}"].T), grads[f"dA{layer_n}"])/self.batch_size
            grads[f"db{layer_n}"] = np.sum(grads[f"dA{layer_n}"], axis=0, keepdims=True)/self.batch_size
        return grads

    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            self.weights[f"W{layer}"] -= self.lr*grads[f"dW{layer}"]
            self.weights[f"b{layer}"] -= self.lr*grads[f"db{layer}"]

    # def one_hot(self, y, n_classes=None):
    #     n_classes = n_classes or self.n_classes
    #     return np.eye(n_classes)[y]

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        cost = -np.sum(labels*np.log(prediction))/prediction.shape[0]
        return cost

    def compute_loss_and_accuracy(self, X, y):
        one_y = y
        y = np.argmax(y, axis=1)  # Change y to integers
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        array_validation_loss = []
        array_train_loss = []
        X_train, y_train = self.train
        y_onehot = y_train
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        print('Init method', self.init_method)
        for epoch in range(n_epochs):
            print('Epoch', epoch)
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                cache = self.forward(minibatchX)
                grads = self.backward(cache, minibatchY)
                self.update(grads)

            X_train, y_train = self.train
            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            X_valid, y_valid = self.valid
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)
            self.train_losses[self.init_method].append(train_loss)
            self.valid_losses[self.init_method].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss, test_accuracy

mnist = load_mnist()
epochs = 10
hidden_dims = [(784,100)]
# hidden_dims = [(500,400), (784,256), (784,480)]
lr = [0.1]
# lr = [7e-4, 7e-2]
activation_str = ['relu']
# activation_str = ['relu', 'sigmoid', 'tanh']
batch_size = [32]
# batch_size = [32, 64]

for (i, hidden) in enumerate(hidden_dims):
    for (j, rate) in enumerate(lr):
        for (k, activation) in enumerate(activation_str):
            for (l, batch) in enumerate(batch_size):
                model = NN(data=mnist, hidden_dims=hidden, lr=rate, activation=activation, batch_size=batch)
                start = time.process_time()
                train_logs = model.train_loop(epochs)
                end = time.process_time()
                print('Training for ', epochs, 'epochs took ', end-start, ' seconds')
                print('hidden', hidden,'rate', rate, 'activation', activation, 'batch_size', batch)
                print('Train accuracy mean', np.mean(model.train_logs['train_accuracy']))
                print(model.train_logs['train_accuracy'])
                print('Validation accuracy mean', np.mean(model.train_logs['validation_accuracy']))
                print(model.train_logs['validation_accuracy'])


# model.init_method = 'zero'
# epochs = 1
# x = np.arange(1, epochs+1)
# print('Train losses', model.init_method)
# print(model.train_losses[model.init_method])
# print('Valid losses', model.init_method)
# print(model.valid_losses[model.init_method])
# plt.figure()
# plt.title('Loss for ' + model.init_method + ' initialization method')
# plt.xlabel('Epoch')
# plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
# plt.ylabel('Loss')
# plt.plot(x, model.train_losses[model.init_method], label='Training')
# plt.plot(x, model.valid_losses[model.init_method], label='Validation')
# plt.legend()
# plt.show()
# init_methods = ['zero', 'normal', 'glorot']
# epochs = 1
# x = np.arange(1, epochs+1)
# for i in range(0, len(init_methods)):
#     model.init_method = init_methods[i]
#     model.train_loop(epochs)
#     print('Train losses', model.init_method)
#     print(model.train_losses[model.init_method])
#     print('Valid losses', model.init_method)
#     print(model.valid_losses[model.init_method])
    # plt.figure(i)
    # plt.title('Training loss for ' + model.init_method + ' initialization methods')
    # plt.xlabel('Epoch')
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    # plt.ylabel('Loss')
    # plt.plot(x, model.train_losses[model.init_method], label='Training')
    # plt.plot(x, model.valid_losses[model.init_method], label='Validation')
    # plt.legend()
    # plt.show()




