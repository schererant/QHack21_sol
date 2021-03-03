#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def classify_data(X_train, Y_train, X_test):
    """Develop and train your very own variational quantum classifier.

    Use the provided training data to train your classifier. The code you write
    for this challenge should be completely contained within this function
    between the # QHACK # comment markers. The number of qubits, choice of
    variational ansatz, cost function, and optimization method are all to be
    developed by you in this function.

    Args:
        X_train (np.ndarray): An array of floats of size (250, 3) to be used as training data.
        Y_train (np.ndarray): An array of size (250,) which are the categorical labels
            associated to the training data. The categories are labeled by -1, 0, and 1.
        X_test (np.ndarray): An array of floats of (50, 3) to serve as testing data.

    Returns:
        str: The predicted categories of X_test, converted from a list of ints to a
            comma-separated string.
    """

    # Use this array to make a prediction for the labels of the data in X_test
    predictions = []

    # QHACK #
    dev = qml.device("default.qubit", wires=3)

    def layer(W):
        qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
        qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
        qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

    def statepreparation(x):
        x /= np.linalg.norm(x)
        qml.templates.embeddings.AngleEmbedding(x, wires=[0, 1, 2])

    @qml.qnode(dev)
    def circuit(weights, x):
        statepreparation(x)

        for W in weights:
            layer(W)

        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))]

    def variational_classifier(var, x):
        weights = var[0]
        bias = var[1]
        circ_out = circuit(weights, x) + bias
        softmax_out = np.exp(circ_out) / np.sum(np.exp(circ_out))
        return softmax_out

    def cross_entropy(labels, predictions, epsilon=1e-12):
        """
        Computes cross entropy between labels (encoded as one-hot vectors)
        and predictions.
        Input: predictions (N, k) ndarray
               labels (N, k) ndarray
        Returns: scalar
        """
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = -np.sum(labels * np.log(predictions + 1e-09)) / N
        return ce

    def run_var_class(var, X):
        return np.array([variational_classifier(var, x) for x in X])

    def predict_to_labels(prediction):
        pred_labels = np.argmax(prediction, axis=1)
        pred_labels_onehot = np.zeros((pred_labels.size, 3))
        pred_labels_onehot[np.arange(pred_labels.size), pred_labels] = 1
        return pred_labels_onehot

    def predict_to_format(prediction):
        return np.argmax(prediction, axis=1) - 1

    def cost(var, X, Y):
        predictions = np.array([variational_classifier(var, x) for x in X])
        return cross_entropy(Y, predictions)

    def accuracy(labels, predictions):
        correct = (labels.shape[0] - (np.abs(labels - predictions).sum() / 2)) / labels.shape[0]
        return correct

    # initialize
    np.random.seed(0)
    num_qubits = 3
    num_layers = 2
    var_init = (0.01 * np.random.randn(num_layers, num_qubits, 3), np.array([0.0, 0.0, 0.0]))

    opt = qml.optimize.NesterovMomentumOptimizer(stepsize=2)

    def transform_labels(labels):
        labels += 1
        labels_onehot = np.zeros((labels.size, 3))
        labels_onehot[np.arange(labels.size), labels] = 1
        return labels_onehot

    Y_train_onehot = transform_labels(Y_train)

    var = var_init
    for it in range(7):
        var = opt.step(lambda v: cost(v, X_train, Y_train_onehot), var)

        # Compute accuracy
        pred_noformat = run_var_class(var, X_train)
        pred_labels = predict_to_labels(pred_noformat)
        acc = accuracy(Y_train_onehot, pred_labels)
        predictions = predict_to_format(pred_noformat)

        print(
            "Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.7f} ".format(
                it + 1, cost(var, X_train, Y_train_onehot), acc
            )
        )
        print('Y_train:', Y_train[:10] - 1)
        print('pred:', predictions[:10])

    pred_noformat = run_var_class(var, X_test)
    predictions = predict_to_format(pred_noformat)
    # QHACK #

    return array_to_concatenated_string(predictions)


def array_to_concatenated_string(array):
    """DO NOT MODIFY THIS FUNCTION.

    Turns an array of integers into a concatenated string of integers
    separated by commas. (Inverse of concatenated_string_to_array).
    """
    return ",".join(str(x) for x in array)


def concatenated_string_to_array(string):
    """DO NOT MODIFY THIS FUNCTION.

    Turns a concatenated string of integers separated by commas into
    an array of integers. (Inverse of array_to_concatenated_string).
    """
    return np.array([int(x) for x in string.split(",")])


def parse_input(giant_string):
    """DO NOT MODIFY THIS FUNCTION.

    Parse the input data into 3 arrays: the training data, training labels,
    and testing data.

    Dimensions of the input data are:
      - X_train: (250, 3)
      - Y_train: (250,)
      - X_test:  (50, 3)
    """
    X_train_part, Y_train_part, X_test_part = giant_string.split("XXX")

    X_train_row_strings = X_train_part.split("S")
    X_train_rows = [[float(x) for x in row.split(",")] for row in X_train_row_strings]
    X_train = np.array(X_train_rows)

    Y_train = concatenated_string_to_array(Y_train_part)

    X_test_row_strings = X_test_part.split("S")
    X_test_rows = [[float(x) for x in row.split(",")] for row in X_test_row_strings]
    X_test = np.array(X_test_rows)

    return X_train, Y_train, X_test


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    X_train, Y_train, X_test = parse_input(sys.stdin.read())
    output_string = classify_data(X_train, Y_train, X_test)
    print(f"{output_string}")
