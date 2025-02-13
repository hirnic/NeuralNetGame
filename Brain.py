import numpy as np
import random
from mpmath import mp
from tensorflow import keras
from tensorflow.keras import layers
mp.dps = 100  # set number of digits

input_dim = 2
output_dim = 1


class Net:
    def __init__(self, layers):
        self.layers = layers  #list of Layer objects

    def total_nodes(self):
        return sum([layer.nodes for layer in self.layers])

    def total_layers(self):
        return len(self.layers)

class Layer:
    def __init__(self, nodes):
        self.nodes = nodes  #int number of nodes in the layer

    def __repr__(self):
        return f"Layer({self.nodes})"


default_network = Net([Layer(1)])

# Train the model
def train_model(number, round_number, node_value, network=default_network, memory=5):
    if round_number == 0:
        return None
    X_train = np.array([[int(number[i - 1]), int(number[i])] for i in range(3, mp.dps)])
    Y_train = np.array([int(number[i + 1]) for i in range(3, mp.dps)])

    for _ in range(min(memory, round_number)):
        X_train = np.concatenate((X_train, X_train))
        Y_train = np.concatenate((Y_train, Y_train))

    # Define the neural network model
    layer_list = [layers.Dense(node_value * network.layers[0].nodes, activation="relu", input_shape=(input_dim,))]
    for i in range(1, len(network.layers)):
        layer_list.append(layers.Dense(node_value * network.layers[i].nodes, activation="relu"))
    layer_list.append(layers.Dense(output_dim))
    model = keras.Sequential(layer_list)

    # Compile the model
    model.compile(optimizer="adam", loss="huber_loss", metrics=["mae"])
    model.fit(X_train, Y_train, epochs=50, batch_size=min(memory, round_number)*100, validation_split=0.1)
    return model.predict(X_train[:100])

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
