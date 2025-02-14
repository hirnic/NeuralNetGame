import numpy as np
import random
from mpmath import mp
import torch
import torch.nn as nn
import torch.optim as optim

mp.dps = 100  # set number of digits

input_dim = 2
output_dim = 1


# Create the network structure
class Net:
    def __init__(self, layers, node_value):
        self.layers = layers
        self.node_value = node_value

    def total_nodes(self):
        return sum([layer.nodes for layer in self.layers])

    def total_layers(self):
        return len(self.layers)


class Layer:
    def __init__(self, nodes):
        self.nodes = nodes

    def __repr__(self):
        return f"Layer({self.nodes})"


default_network = Net([Layer(1)], node_value=1)


class NeuralNetwork(nn.Module):
    def __init__(self, network):
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


# Q-learning variables
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # exploration rate

# Action space
digit_range = list(range(10))

# Initialize the neural network
model = NeuralNetwork(default_network)

# Q-table initialization (a simple array for each (pair of digits) -> possible next digits)
q_table = np.zeros((100, 100, 10))  # (X1, X2) -> next digit (0-9)


def get_action(state):
    """ Epsilon-greedy policy """
    if random.uniform(0, 1) < epsilon:
        # Exploration: Choose a random action
        return random.choice(digit_range)
    else:
        # Exploitation: Choose the best action (the one with the highest Q value)
        x1, x2 = state
        return np.argmax(q_table[x1][x2])


def update_q_table(state, action, reward, next_state):
    """ Update Q-value using Q-learning formula """
    x1, x2 = state
    next_x1, next_x2 = next_state
    max_future_q = np.max(q_table[next_x1][next_x2])  # max Q value for the next state
    current_q = q_table[x1][x2][action]  # current Q value for (state, action)

    # Q-learning formula
    new_q = current_q + alpha * (reward + gamma * max_future_q - current_q)
    q_table[x1][x2][action] = new_q


def train_model(number, round_number, memory=5):
    if round_number == 0:
        return None

    X_train = np.array([[int(number[i - 1]), int(number[i])] for i in range(3, mp.dps)])
    Y_train = np.array([int(number[i + 1]) for i in range(3, mp.dps)])

    # Training with Q-learning loop
    for episode in range(50):
        state = (int(number[episode % (mp.dps - 2)]), int(number[(episode + 1) % (mp.dps - 2)]))
        action = get_action(state)

        # Take the action (predict next digit) and receive reward
        predicted_digit = action
        next_state = (int(number[episode + 1]), int(number[episode + 2]))  # next pair of digits
        correct_digit = Y_train[episode]

        # Reward +1 if the prediction is correct, else -1
        reward = 1 if predicted_digit == correct_digit else -1

        # Update Q-table
        update_q_table(state, action, reward, next_state)

        # Optionally, we could use the neural network to guide the exploration/exploitation
        model_input = torch.tensor(state, dtype=torch.float32).view(1, -1)
        model_output = model(model_input)

    # Return the Q-table to inspect learned values
    return q_table


# Example usage of the RL-based model training
number = str(mp.pi)[2:]  # Digits of pi
train_model(number, round_number=100)
