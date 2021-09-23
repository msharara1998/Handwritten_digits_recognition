# This code is based upon the code written by Michael Nielson in his book
# Neural Networks and Deep Learning
import numpy as np
from matplotlib import pyplot
import time


class ANN:
    """This is the primitive network class"""

    def __init__(self, layers_nodes):
        """Initialization of the network with random weights and biases.
        A list argument determines the number of nodes in each layer."""
        self.num_layers = len(layers_nodes)
        np.random.seed(0)
        self.weights = [np.random.randn(next_layer, layer) for layer, next_layer in
                        zip(layers_nodes[:-1], layers_nodes[1:])]
        self.biases = [np.zeros((layer, 1)) for layer in layers_nodes[1:]]

    def forward_propagation(self, x, cache=False):
        """Feed forward the network using the input x. If cache is set to
        True, the outputs z's and activations of each layer are cached.
        Particularly, this is used in backward propagation to calculate the
        gradients"""
        activation = x
        if not cache:
            for w, b in zip(self.weights, self.biases):
                activation = self.sigmoid(np.dot(w, activation) + b)
            return activation
        else:
            activations = [activation]
            zs = []
            for w, b in zip(self.weights, self.biases):
                zs.append(np.dot(w, activations[-1]) + b)
                activations.append(self.sigmoid(zs[-1]))
            return zs, activations

    def backward_propagation(self, y, zs, activations):
        """Propagate backwards through the network to calculate the local
        gradient delta corresponding to each layer and returns a list of
        them"""
        deltas = [(activations[-1] - y) * self.sigmoid_prime(zs[-1])]
        for i in range(2, self.num_layers):
            deltas.append((self.weights[self.num_layers - i].transpose() @ deltas[-1]) * self.sigmoid_prime(zs[-i]))
        return list(reversed(deltas))

    def mini_batch_gradient_descent(self, batch, learning_rate):
        """Performs a gradient descent step over a batch of the training
        examples given that they are split into mini-batches."""
        m = batch[0].shape[1]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]
        zs, activations = self.forward_propagation(batch[0], cache=True)
        cost_over_batch = self.compute_cost(activations[-1], batch[1])
        deltas = self.backward_propagation(batch[1], zs, activations)
        delta_w = [d @ a.transpose() + dw for a, dw, d in zip(activations, delta_w, deltas)]
        delta_b = [db + np.sum(d) for db, d in zip(delta_b, deltas)]
        self.weights = [w - (learning_rate / m) * dw for w, dw in zip(self.weights, delta_w)]
        self.biases = [b - (learning_rate / m) * db for b, db in zip(self.biases, delta_b)]
        return np.sum(cost_over_batch) / m

    def train_network(self, training_data, epochs, batch_size, learning_rate, test_data):
        """Trains the network over several epochs and prints the network stats"""
        start = time.process_time()  # set the timer
        costs = []
        accuracies = []
        for epoch in range(epochs):
            np.random.seed(1)
            np.random.shuffle(training_data)
            batches = self.vectors_to_matrices(training_data, batch_size)
            cost_over_batches = 0
            for batch in batches:
                cost_over_batches += self.mini_batch_gradient_descent(batch, learning_rate)
            costs.append(cost_over_batches / len(batches))
            accuracy = self.evaluate_network(test_data) / len(test_data)
            accuracies.append(accuracy)
            print("epoch {0}: cost: {1:.3f} -- accuracy: {2:.3f} %".format(str(epoch + 1).ljust(3),
                                                                           cost_over_batches / len(batches),
                                                                           100 * accuracy))
        self.plot_stats(costs, accuracies, epochs)
        print(time.process_time() - start)  # print training time

    def vectors_to_matrices(self, training_data, batch_size):
        """Stacks the vectors together in a matrix to form a batch according
        to the batch size"""
        m = batch_size
        n = len(training_data)
        batches = [training_data[k:k + m] for k in range(0, n, m)]
        new_batches = []
        for batch in batches:
            batch = (np.array([np.reshape(v[0], (784,)) for v in batch]).T,
                     np.array([np.reshape(v[1], (10,)) for v in batch]).T)
            new_batches.append(batch)
        return new_batches

    def compute_cost(self, al, y):
        """Compute the cost function for a single batch"""
        return 0.5 * np.sum((al - y) ** 2, axis=0)

    def sigmoid(self, z):
        """using broadcasting, numpy will apply the function to the whole
        matrix/numpy array."""
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        """Compute the derivative of the sigmoid function"""
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def evaluate_network(self, test_data):
        results = [(np.argmax(self.forward_propagation(x)), y) for x, y in test_data]
        return sum(int(x == y) for (x, y) in results)

    def plot_stats(self, costs, accuracies, epochs):
        plot1 = pyplot.figure(1)
        pyplot.xticks(range(0, epochs, 2))
        pyplot.ylabel('COST')
        pyplot.xlabel('EPOCHS')
        pyplot.plot(range(epochs), costs)
        plot2 = pyplot.figure(2)
        pyplot.xticks(range(0, epochs, 2))
        pyplot.ylabel('ACCURACY')
        pyplot.xlabel('EPOCHS')
        pyplot.plot(range(epochs), accuracies)
        pyplot.show()


def data_preprocessing(training_data, test_data):
    """This function processes the data to be convenient with the Artificial
    Neural Network"""
    X_train = np.reshape(training_data[0], (60000, 784, 1))
    y_train = np.reshape(training_data[1], (60000, 1))
    X_test = np.reshape(test_data[0], (10000, 784, 1))
    y_test = np.reshape(test_data[1], (10000, 1))
    training_data = [(x / 256, vectorize(y)) for x, y in zip(X_train, y_train)]
    test_data = [(x / 256, y) for x, y in zip(X_test, y_test)]

    return training_data, test_data


def vectorize(y):
    """Transforms a digit betweeen 0 and 9 to one hot-encoded vector"""
    v = np.zeros((10, 1))
    v[y] = 1.0
    return v
