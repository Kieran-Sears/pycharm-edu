from __future__ import print_function
import argparse
import torch
import copy as obj


class Model(object):
    def __init__(self, nodes, connections, biases, gradient):
        self.nodes = nodes
        self.connections = connections
        self.biases = biases
        self.gradient = gradient


class Connections(object):
    def __init__(self, weight_ih, weight_hh, weight_ho):
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.weight_ho = weight_ho


class Nodes(object):
    def __init__(self, input_layer, hidden_layers, output_layer):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer


class Biases(object):
    def __init__(self, hidden_biases, output_biases):
        self.hidden_biases = hidden_biases
        self.output_biases = output_biases


def initialise_network(input_size, hidden_size, output_size):
    weight_ih = torch.zeros(input_size, hidden_size[0])
    weight_hh = torch.zeros(calculate_hidden_connections_shape(hidden_size))
    weight_ho = torch.zeros(hidden_size[len(hidden_size) - 1], output_size)
    connections = Connections(weight_ih, weight_hh, weight_ho)

    input_layer = torch.FloatTensor(input_size).uniform_(0, 1)
    hidden_layers = init_hidden_layers(hidden_size)
    output_layer = torch.empty(output_size)
    nodes = Nodes(input_layer, hidden_layers, output_layer)

    hidden_biases = torch.zeros(calculate_hidden_biases_shape(hidden_size))
    output_biases = torch.zeros(hidden_size[len(hidden_size) - 1], output_size)
    biases = Biases(hidden_biases, output_biases)

    gradient_ih_00 = 0.0

    model = Model(nodes, connections, biases, gradient_ih_00)
    model = probsToClasses(computeOutputs(model))
    return model


def init_hidden_layers(hidden_size):
    hidden_layers = []
    for i in range(len(hidden_size)):
        hidden_layers.append(torch.FloatTensor(hidden_size[i]).uniform_(-9, 9))
    return hidden_layers


def calculate_hidden_connections_shape(hidden_size):
    shape = []
    for layer in range(len(hidden_size) - 1):
        layer_count = 0
        for node_0 in range(hidden_size[layer]):
            for node_1 in range(hidden_size[layer + 1]):
                layer_count += 1
        shape.append(layer_count)
    return shape


def calculate_hidden_biases_shape(hidden_size):
    shape = []
    for i in range(len(hidden_size)):
        bias_count = 0
        for j in range(hidden_size[i]):
            bias_count += 1
        shape.append(bias_count)
    return shape


def update_nodes(hidden_layer, previous_layer, weights, biases):
    new_nodes = torch.empty(hidden_layer.size)
    for j in range(hidden_layer.numel()):
        for jj in range(previous_layer.numel()):
            new_nodes[j] += weights[jj][j] * previous_layer[jj]
        new_nodes[j] += biases[j]
    return new_nodes


def tanh(tensor_1d):
    ret = []
    for elem in range(tensor_1d.numel()):
        ret.append(torch.tanh(tensor_1d[elem]))
    return torch.FloatTensor(ret)


def computeOutputs(model):

    input_nodes = tanh(update_nodes(model.nodes.input_layer, model.nodes.hidden_layers[0], model.connections.weight_ih, model.biases.hidden_biases[0]))

    hidden_nodes = []
    for layer in range(1, len(model.nodes.hidden_layers)):
        hidden_nodes.append(tanh(update_nodes(
            model.nodes.hidden_layers[layer],
            model.nodes.hidden_layers[layer -1],
            model.connections.weight_hh[layer -1],
            model.biases.hidden_biases[layer])))

    output_nodes = torch.nn.Softmax(update_nodes(
        model.nodes.output_layer,
        model.nodes.hidden_layers[len(model.nodes.hidden_layers)],
        model.connections.weight_ho,
        model.biases.output_biases))

    model = obj.copy()
    return


def probsToClasses(inputs):
    result = torch.zeros(inputs.size)
    idx = maxIndex(inputs)
    result[idx] = 1.0
    return result


def maxIndex(inputs):
    maxIdx = 0
    maxVal = inputs[0]
    for i in range(inputs.size):
        if inputs[i] > maxVal:
            maxVal = inputs[i]
            maxIdx = i

    return maxIdx


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=int, dest="input", help="number of inputs")
parser.add_argument('--hidden', nargs='+', type=int, dest='hidden', help='number of hidden layers')
parser.add_argument('--output', type=int, dest="output", help="number of outputs")
parser.add_argument('--items', type=int, dest="items", help="size of dataset")
parser.add_argument('--seed', type=int, dest="seed", help="random generator seed")
args = parser.parse_args()

print("Found application input params: " + str(args))

initialise_network(args.input, args.hidden, args.output)
