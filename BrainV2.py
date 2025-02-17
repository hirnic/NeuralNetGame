# In this script we are making a reinforcement learning model whose training set stores up to N data points. When the
# model somehow does better than the worst round in the training set, the worst round in the training set is replaced by
# the better one. The performance rating for a round is  S = \frac{points acheived}{points possible}.
# The input data will now be the entire string of numbers for the round and the output data will be the player moves.

import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset  # For mini-batch training


input_dim = 6
output_dim = 1
batch_size = 10
max_data_points = 50  # maximum number of data points to store in the training set. More can be purchased.


class Layer:
    def __init__(self, nodes):
        self.nodes = nodes  # int number of nodes in the layer

    def __repr__(self):
        return f"Layer({self.nodes})"


class Net:
    def __init__(self, layers, node_value):
        self.layers = layers  # list of Layer objects
        self.node_value = node_value  # int value for nodes
        self.exploration_rate = 0.1  # exploration rate for epsilon-greedy strategy... maybe?
        self.memory = 50  # max_data_points integer
        self.training_set = []  # list of TrialData objects

    def total_nodes(self):
        return sum([layer.nodes for layer in self.layers])

    def total_layers(self):
        return len(self.layers)


default_network = Net([Layer(50)], node_value=1)


# Here we have the class for the trial data.
class TrialData:
    def __init__(self, on_screen, next_move):
        self.on_screen = on_screen  # List of lists [lane, x_position, velocity, price, player lane], max determined by
        # limitations of neural network. Maximum can be increased by purchasing more space in the shop up to 5.
        self.next_move = next_move  # Int
        self.score = 0  # Int represents the number of points gained due to this action


full_data_set = []
# # List of TrialData objects
# for _ in range(max_data_points):
#     input = np.array([random.randint(0,9) for _ in range(input_dim)])
#     output = np.array([input[0]])
#     delta = random.uniform(0,1)
#     # Here we select some random entries in the output and change them
#     while delta > 0.1:
#         output[random.randint(0,output_dim-1)] = random.randint(0,9)
#         delta -= 0.1
#     trial_data = TrialData(input, output)
#     trial_data.score = (np.linalg.norm(input[::6] - output)+0.1)**(-1)
#     # trial_data.score = 10
#     full_data_set.append(trial_data)

# X_train = np.array([data.number for data in full_data_set])
# Y_train = np.array([data.moves for data in full_data_set])


class NeuralNetwork(nn.Module):
    def __init__(self, network):  # network is a Net object
        super(NeuralNetwork, self).__init__()
        layers = [nn.Linear(input_dim, network.node_value * network.layers[0].nodes), nn.Softplus()]
        for i in range(1, len(network.layers)):
            layers.append(nn.Linear(network.node_value * network.layers[i - 1].nodes,
                                    network.node_value * network.layers[i].nodes))
            layers.append(nn.Softplus())  # Replace ReLU with Softplus
        layers.append(nn.Linear(network.node_value * network.layers[-1].nodes, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_model(network=default_network):
    # global X_train, Y_train
    X_train = np.array([data.on_screen for data in network.training_set])
    Y_train = np.array([data.next_move.item() if isinstance(data.next_move, torch.Tensor) else data.next_move
                        for data in network.training_set], dtype=np.float32)
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    Y_train_tensor = torch.from_numpy(Y_train).float().view(-1, 1)

    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = NeuralNetwork(network)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(50):
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

    return model


# score_per_round = []
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# score_per_round = []  # Initialize the score list
#
# plt.ion()  # Turn on interactive mode
# fig, ax = plt.subplots()
# ax.set_xlabel("Rounds")
# ax.set_ylabel("Score")
# ax.set_title("Learning Progress")
# line, = ax.plot([], [], 'b-', label="Score per Round")  # Blue line for individual scores
# avg_line, = ax.plot([], [], 'r-', label="Average Score")  # Red line for average score
# ax.legend()  # Add a legend to indicate the two lines
#
# for _ in range(1000):
#     epsilon = random.uniform(0, 1)
#     number = np.array([random.randint(0, 9) for _ in range(input_dim)])
#     moves = train_model(number)
#
#     if epsilon < 0.1:
#         delta = random.uniform(0, 1)
#         while delta > 0.1:
#             moves[random.randint(0, output_dim - 1)] = random.randint(0, 9)
#             delta -= 0.05
#
#     data = TrialData(number, moves)
#     score = (np.linalg.norm(number[::6] - moves) + 0.1) ** (-1)
#     data.score = score
#
#     score_per_round.append(score)  # Store the score
#
#     worst_performance = data
#     worst_rounds = sorted(full_data_set, key=lambda x: x.score)[:10]
#     if data.score > worst_rounds[0].score:
#         if len(full_data_set) < max_data_points:
#             full_data_set.append(data)
#         else:
#             full_data_set[0] = data
#
#     # Compute cumulative average
#     cumulative_average = [np.mean(score_per_round[:i+1]) for i in range(len(score_per_round))]
#
#     # Update the plot
#     line.set_xdata(range(len(score_per_round)))
#     line.set_ydata(score_per_round)
#     avg_line.set_xdata(range(len(score_per_round)))
#     avg_line.set_ydata(cumulative_average)  # Set the red line to the cumulative average
#     ax.relim()
#     ax.autoscale_view()
#     plt.draw()
#     plt.pause(0.01)  # Pause to allow the plot to update
#
#     print("Round: ", str(_ + 1))
#     print("Input Number:    ", data.number)
#     if epsilon < 0.1:
#         print("Predicted Moves: ", np.round(data.moves).astype(int) % 10, " (Exploration)")
#     else:
#         print("Predicted Moves: ", np.round(data.moves).astype(int) % 10)
#
# plt.ioff()  # Turn off interactive mode when done
# plt.show()  # Display the final plot
