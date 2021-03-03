#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)


def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.

    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.

    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.

    Args:
        params (np.ndarray): Input parameters, of dimension 6

    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = np.zeros(6)

    # QHACK #

    # Get gradients
    s_param_g = 6

    def calc_gradient(circuit, params):
        unit_v = np.zeros_like(params)
        gradient_ = np.zeros_like(params)
        add_calc = np.zeros_like(params)
        substr_cal = np.zeros_like(params)
        for ii in np.ndenumerate(unit_v):
            ii = ii[0]
            unit_v[ii] = 1
            add_calc[ii] = circuit(params + s_param_g * unit_v)
            substr_cal[ii] = circuit(params - s_param_g * unit_v)
            gradient_[ii] = (add_calc[ii] - substr_cal[ii]) / (2 * np.sin(s_param_g))
            unit_v[ii] = 0
        return gradient_

    gradient = calc_gradient(qnode, params)

    # Get FS Metric
    def fs_metric(circuit, params):
        F_array = np.zeros((len(params), len(params)))
        base_state = circuit(params)
        unit_1 = np.zeros_like(params)
        unit_2 = np.zeros_like(params)

        def fubini_help(added_units):
            return np.power(np.abs((base_state.conjugate() @ circuit(params + (added_units) * (math.pi / 2)))), 2)

        for ii in range(len(params)):
            for jj in range(len(params)):
                unit_1[ii] = 1
                unit_2[jj] = 1
                first = -fubini_help(unit_1 + unit_2)
                second = fubini_help(unit_1 - unit_2)
                third = fubini_help(-unit_1 + unit_2)
                fourth = -fubini_help(-unit_1 - unit_2)
                unit_1[ii] = 0
                unit_2[jj] = 0
                F_array[ii, jj] = (first + second + third + fourth) / 8
        return F_array

    F_array = fs_metric(qnode_state, params)

    # Combine to get natural gradient definition

    def nat_grad(F_array, gradient):
        F_array_inv = np.linalg.inv(F_array)
        return F_array_inv @ gradient

    natural_grad = nat_grad(F_array, gradient)
    # QHACK #

    return natural_grad


def non_parametrized_layer():
    """A layer of fixed quantum gates.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")
