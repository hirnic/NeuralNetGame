import numpy as np
import random
from mpmath import mp
import torch
import torch.nn as nn
import torch.optim as optim

mp.dps = 100  # set number of digits

input_dim = 2
output_dim = 1


class Net:
    def __init__(self, layers, node_value):
        self.layers = layers  # list of Layer objects
        self.node_value = node_value  # int value for nodes

    def total_nodes(self):
        return sum([layer.nodes for layer in self.layers])

    def total_layers(self):
        return len(self.layers)


class Layer:
    def __init__(self, nodes):
        self.nodes = nodes  # int number of nodes in the layer

    def __repr__(self):
        return f"Layer({self.nodes})"


default_network = Net([Layer(1)], node_value=1)


class NeuralNetwork(nn.Module):
    def __init__(self, network):  # network is a Net object
        super(NeuralNetwork, self).__init__()
        layers = [nn.Linear(input_dim, network.node_value * network.layers[0].nodes), nn.ReLU()]
        for i in range(1, len(network.layers)):
            layers.append(nn.Linear(network.node_value * network.layers[i - 1].nodes,
                                    network.node_value * network.layers[i].nodes))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(network.node_value * network.layers[-1].nodes, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_model(number, round_number, network=default_network, memory=5):
    if round_number == 0:
        return None
    X_train = np.array([[int(number[i - 1]), int(number[i])] for i in range(3, mp.dps)])
    Y_train = np.array([int(number[i + 1]) for i in range(3, mp.dps)])

    for _ in range(min(memory, round_number)):
        X_train = np.concatenate((X_train, X_train))
        Y_train = np.concatenate((Y_train, Y_train))

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)

    model = NeuralNetwork(network)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(X_train[:100])
    return predictions.numpy()


def format_prediction(predictions):
    output = ""
    if predictions is None:
        for _ in range(mp.dps):
            output += str(random.randint(0, 9))
    else:
        for y in predictions:
            output += str(round(y[0]))
    print("Length of output: ", len(output))
    return output
