#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np
import remote_cirq

API_KEY = ""
sim = remote_cirq.RemoteSimulator(API_KEY)

# DO NOT MODIFY any of these parameters
WIRES = 2
LAYERS = 5
NUM_PARAMETERS = LAYERS * WIRES * 3


def optimize_circuit(params, floq=False):
    """Minimize the variational circuit and return its minimum value.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers. You should create a device and convert the
    variational_circuit function into an executable QNode. Next, you should minimize the variational
    circuit using gradient-based optimization to update the input params. Return the optimized value
    of the QNode as a single floating-point number.

    Args:
        params (np.ndarray): Input parameters to be optimized, of dimension 30

    Returns:
        float: the value of the optimized QNode
    """

    optimal_value = 0.0

    # QHACK #

    # Initialize the device
    if floq:
        dev = qml.device("cirq.simulator",
                         wires=WIRES,
                         simulator=sim,
                         analytic=False)

    else:
        # Instantiate the QNode
        dev = qml.device("default.qubit", wires=WIRES)


    # Use TPU's

    circuit = qml.QNode(variational_circuit, dev)
    # Minimize the circuit
    def parameter_shift_term(circuit, params, i, shift):
        shifted = params.copy()
        shifted[i] += shift
        forward = circuit(shifted)

        shifted = params.copy()
        shifted[i] -= shift
        backward = circuit(shifted)

        return (forward - backward) / (2 * np.sin(shift))

    def comp_gradient(circuit, params):
        gradient = np.zeros([len(params)], dtype=np.float64)
        for i in range(len(params)):
            gradient[i] = parameter_shift_term(circuit, params, i, 2)
        return gradient

    # dcircuit = qml.grad(circuit)
    # gradients = dcircuit(params)
    # print(gradients)
    # gradients = comp_gradient(circuit, params)
    # print(gradients)

    step_size = 0.1
    for i in range(200):
        # gradients = dcircuit(params)[0]
        gradients = comp_gradient(circuit, params)
        params -= step_size * gradients
        optimal_value = circuit(params)
        print('Iterations:', i, 'Optimal value:', optimal_value, end="\r")
    # print('')
    # step_size = 0.5
    # for i in range(25):
    #     gradients = comp_gradient(circuit, params)
    #     params -= step_size * gradients
    #     optimal_value = circuit(params)
    #
    # step_size = 0.1
    # for i in range(25):
    #     gradients = comp_gradient(circuit, params)
    #     params -= step_size*gradients
    #     optimal_value = circuit(params)

    # QHACK #

    # Return the value of the minimized QNode
    return optimal_value


def variational_circuit(params):
    """
    # DO NOT MODIFY anything in this function! It is used to judge your solution.

    This is a template variational quantum circuit containing a fixed layout of gates with variable
    parameters. To be used as a QNode, it must either be wrapped with the @qml.qnode decorator or
    converted using the qml.QNode function (as shown above).

    The output of this circuit is the expectation value of a Hamiltonian. An unknown Hamiltonian
    will be used to judge your solution.

    Args:
        params (np.ndarray): An array of optimizable parameters of shape (30,)
    """
    parameters = params.reshape((LAYERS, WIRES, 3))
    qml.templates.StronglyEntanglingLayers(parameters, wires=range(WIRES))
    return qml.expval(qml.Hermitian(hamiltonian, wires=[0, 1]))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process Hamiltonian data
    hamiltonian = sys.stdin.read()
    hamiltonian = hamiltonian.split(",")
    hamiltonian = np.array(hamiltonian, float).reshape((2 ** WIRES, 2 ** WIRES))

    # Generate random initial parameters
    np.random.seed(1967)
    initial_params = np.random.random(NUM_PARAMETERS)

    minimized_circuit = optimize_circuit(initial_params)
    print(f"{minimized_circuit:.6f}")
