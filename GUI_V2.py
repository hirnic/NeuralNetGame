# This is the graphical user interface the player will be using.

import BrainV2
import tkinter as tk
import random
import numpy as np
import torch
# import time

# Set up the window
root = tk.Tk()
root.title("Catch the Collectibles")
root.geometry("600x600")  # Updated geometry

screen_width = 600
screen_height = 500

# Canvas for game
canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg="white")
canvas.pack()

# Number of lanes (10 horizontal lanes)
num_lanes = 10
lane_height = screen_height // num_lanes  # Height of each lane

# Game variables
game_speed = 1  # Game milliseconds per real millisecond
player_width = 30
player_height = 30
player_velocity = 1 * game_speed  # pixels per millisecond?
player_lane = 5  # Start at the middle lane (lane 5)
player_moves = []  # List containing the next move
player_network = BrainV2.Net([BrainV2.Layer(1)], 1)  # Initialize the player network

collectibles = []
collectible_width = 20
collectible_height = 20
collectible_speed = 10  # pixels per movement
generation_rate = int(333 / game_speed)  # milliseconds per generation
movement_rate = int(60 / game_speed)  # milliseconds per movement
delay = int((screen_width * movement_rate / collectible_speed) / game_speed)  # milliseconds
score = 0

model = None

game_active = False
network_active = False

all_frames = {"layer0": tk.Frame(root),
              "layer1": tk.Frame(root),
              "layer2": tk.Frame(root),
              "layer3": tk.Frame(root),
              "layer4": tk.Frame(root),
              }

all_labels = {"Loading Screen": tk.Label(root, text="Loading...", font=("Helvetica", 16)),
              "lane0": tk.Label(root, text="0", font=("Helvetica", 16)),
              "lane1": tk.Label(root, text="1", font=("Helvetica", 16)),
              "lane2": tk.Label(root, text="2", font=("Helvetica", 16)),
              "lane3": tk.Label(root, text="3", font=("Helvetica", 16)),
              "lane4": tk.Label(root, text="4", font=("Helvetica", 16)),
              "lane5": tk.Label(root, text="5", font=("Helvetica", 16)),
              "lane6": tk.Label(root, text="6", font=("Helvetica", 16)),
              "lane7": tk.Label(root, text="7", font=("Helvetica", 16)),
              "lane8": tk.Label(root, text="8", font=("Helvetica", 16)),
              "lane9": tk.Label(root, text="9", font=("Helvetica", 16)),
              "score_label": tk.Label(root, text="Score: 0", font=("Helvetica", 16)),
              "move_label": tk.Label(root, text="Move: 0", font=("Helvetica", 12)),
              "generation_label": tk.Label(root, text="Generation: 0", font=("Helvetica", 12)),

              "money_label": tk.Label(root, text="Money: 0", font=("Helvetica", 16)),
              "nodes_label": tk.Label(root, text="Nodes: 1", font=("Helvetica", 16)),
              "layers_label": tk.Label(root, text="Layers: 1", font=("Helvetica", 16)),
              "rounds_label": tk.Label(root, text="Rounds played: 0", font=("Helvetica", 16)),
              }

all_buttons = {
    "play_button": tk.Button(root, text="Play", font=("Helvetica", 20)),
    "quit_button": tk.Button(root, text="Quit", font=("Helvetica", 20)),

    "supervise_button": tk.Button(root, text="Supervise Network", font=("Helvetica", 20)),
    "autotrain_button": tk.Button(root, text="Autotrain Network", font=("Helvetica", 20)),

    "slow_button": tk.Button(root, text="Slow", font=("Helvetica", 20)),
    "medium_button": tk.Button(root, text="Medium", font=("Helvetica", 20)),
    "fast_button": tk.Button(root, text="Fast", font=("Helvetica", 20)),
    "hyper_button": tk.Button(root, text="Hyper", font=("Helvetica", 20)),

    "purchase_node": tk.Button(root, text="Purchase Node", font=("Helvetica", 20)),
    "purchase_layer": tk.Button(root, text="Purchase Layer", font=("Helvetica", 20)),

    "add_layer": tk.Button(root, text="+", font=("Helvetica", 20)),
    "remove_layer": tk.Button(root, text="-", font=("Helvetica", 20)),
    "layer0_add": tk.Button(root, text="+", font=("Helvetica", 20)),
    "layer0_remove": tk.Button(root, text="-", font=("Helvetica", 20)),
    "layer1_add": tk.Button(root, text="+", font=("Helvetica", 20)),
    "layer1_remove": tk.Button(root, text="-", font=("Helvetica", 20)),
    "layer2_add": tk.Button(root, text="+", font=("Helvetica", 20)),
    "layer2_remove": tk.Button(root, text="-", font=("Helvetica", 20)),
    "layer3_add": tk.Button(root, text="+", font=("Helvetica", 20)),
    "layer3_remove": tk.Button(root, text="-", font=("Helvetica", 20)),
    "layer4_add": tk.Button(root, text="+", font=("Helvetica", 20)),
    "layer4_remove": tk.Button(root, text="-", font=("Helvetica", 20)),
}

current_game_data = []


# Here we have the class for the trial data.
class TrialData:
    def __init__(self, on_screen, next_move):
        self.on_screen = on_screen  # List of lists [lane, x_position, velocity, price, player lane], max determined by
        # limitations of neural network. Maximum can be increased by purchasing more space in the shop up to 5.
        self.next_move = next_move  # Int
        self.score = 0  # Int represents the number of points gained due to this action


class Player:
    def __init__(self):
        self.x1 = 50
        self.x2 = 50 + player_width
        self.y1 = player_lane * lane_height
        self.y2 = player_lane * lane_height + player_height
        self.velocity = player_velocity
        self.lane = 5
        self.width = player_width
        self.height = player_height
        self.money = 0
        self.nodes = 1
        self.layers = 1
        self.id = None

    def auto_move(self, lane):
        lane_diff = lane - self.lane
        self.lane = lane  # Move directly to a specified lane
        self.y1 = self.lane * lane_height
        self.y2 = self.lane * lane_height + self.height
        canvas.coords(self.id, self.x1, self.y1, self.x2, self.y2)
        canvas.update()
        # for i in range(1, 13):
        #     self.y1 = (self.lane - lane_diff * (1 - i / 12)) * lane_height
        #     self.y2 = (self.lane - lane_diff * (1 - i / 12)) * lane_height + self.height
        #     canvas.coords(self.id, self.x1, self.y1, self.x2, self.y2)
        #     canvas.update()

    def manual_move(self, event):
        lane_diff = 0
        if event.keysym == 'Up' and self.lane > 0:
            lane_diff = -1
            self.lane -= 1  # Move up one lane
        elif event.keysym == 'Down' and self.lane < num_lanes - 1:
            lane_diff = 1
            self.lane += 1  # Move down one lane
        elif event.keysym in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            lane_diff = int(event.keysym) - self.lane
            self.lane = int(event.keysym)  # Move directly to a specified lane
        self.y1 = self.lane * lane_height
        self.y2 = self.lane * lane_height + self.height
        canvas.coords(self.id, self.x1, self.y1, self.x2, self.y2)
        canvas.update()
        # for i in range(1, 7):
        #     self.y1 = (self.lane - lane_diff * (1 - i / 6)) * lane_height
        #     self.y2 = (self.lane - lane_diff * (1 - i / 6)) * lane_height + self.height
        #     canvas.coords(self.id, self.x1, self.y1, self.x2, self.y2)
        #     canvas.update()

    def place_avatar(self):
        self.id = canvas.create_rectangle(50, 5 * lane_height, 50 + player_width,
                                          player_lane * lane_height + player_height, fill="blue")


player = Player()


# Here we have a class for collectibles. It is just a canvas rectangle, but it also has a price associated with it.
class Collectible:
    def __init__(self, price, lane, velocity):
        self.price = price
        self.lane = lane
        self.velocity = velocity  # float
        self.x1 = 600
        self.x2 = 600 + collectible_width
        y1 = lane * lane_height + 5
        self.y1 = y1
        self.y2 = y1 + collectible_height

        def rgb_to_hex(r, g, b):
            return f'#{r:02x}{g:02x}{b:02x}'

        self.id = canvas.create_rectangle(600, y1, 600 + collectible_width, y1 + collectible_height,
                                          fill=rgb_to_hex(2 * price ** 2, int(255 / (2 * price ** 2)), 0))

    def move(self):
        self.x1 -= collectible_speed * self.velocity
        self.x2 -= collectible_speed * self.velocity
        canvas.move(self.id, (-1) * collectible_speed * self.velocity, 0)


# This holds all the information about the screen and contains methods for converting to Pytorch-ready data
class Screen:
    def __init__(self, screen_shot):
        self.on_screen = screen_shot  # Usually collectibles

    def to_tensor(self):
        # If the number of collectibles on the screen is less than the maximum, we do this
        fake_block = Collectible(0, -1, -1)
        fake_block.x1 = -1
        fake_block.x2 = -1

        # This converts the collectible objects to a tensor
        tensor = torch.tensor([[block.lane, block.x1, block.velocity, block.price] for block in self.on_screen],
                              dtype=torch.float32)

        # This ensures there are exactly 12 entries in tensor
        if len(self.on_screen) > 12:
            tensor = tensor[:12]
        elif len(self.on_screen) < 12:
            for _ in range(12 - len(self.on_screen)):
                tensor = torch.cat((tensor, torch.tensor([[0, -1, -1, -1]], dtype=torch.float32)))

        return tensor


def clear_GUI():
    canvas.delete("all")
    for frame in all_frames:
        all_frames[frame].place_forget()
    for label in all_labels:
        all_labels[label].place_forget()
    for button in all_buttons:
        all_buttons[button].place_forget()
    # Clear the collectibles
    for collectible in collectibles:
        canvas.delete(collectible.id)
    collectibles.clear()
    # Clear the player
    if player is not None:
        canvas.delete(player.id)
        player.id = None


def end_game():
    if collectibles == []:
        global game_active
        game_active = False
        canvas.delete("all")
        player.id = None
        for data_point in current_game_data:
            player_network.training_set = sorted(player_network.training_set, key=lambda x: x.score)
            if len(player_network.training_set) < player_network.memory:
                player_network.training_set.append(data_point)
            elif data_point.score > player_network.training_set[0].score:
                player_network.training_set[0] = data_point
        root.unbind("<Up>")
        root.unbind("<Down>")
        for i in range(10):
            root.unbind(str(i))
        build_screen()
    else:
        canvas.after(50, end_game)


# This is the function that generates the collectibles.
def generate_collectible(optional=0):
    all_labels["generation_label"].configure(text="Generation: " + str(optional))
    price = random.randint(1, 10)
    if price == 10:
        pass
    elif price > 7:
        price = 5
    else:
        price = 1
    collectible = Collectible(price, random.randint(0, num_lanes - 1), random.uniform(1, 3))
    collectibles.append(collectible)
    canvas.update()


# This handles collision detection
def detect_collision(collectible):
    global score
    if ((player.x1 < collectible.x1 < player.x2 or player.x1 < collectible.x2 < player.x2)
            and player.lane == collectible.lane):
        player.money += collectible.price
        canvas.delete(collectible.id)
        collectibles.remove(collectible)
        score += collectible.price
        all_labels["score_label"].configure(text="Score: " + str(score))


# This is the function that moves the player and tracks all the data.
def move_player(optional=0):
    all_labels["move_label"].configure(text="Move: " + str(optional))
    # We need to convert the screen data to the input data for the neural network.
    # The input data is of the form [lane, x_position, velocity, price]
    # Keep in mind, the output data is of the form [next move], where next move is an int (0, 9)
    screen_shot = Screen(collectibles)
    print(screen_shot)
    on_screen = sorted(collectibles, key=lambda block: block.x1)[0]
    on_screen = np.array(
        [on_screen.lane, on_screen.x1, on_screen.velocity, on_screen.price])
    if network_active:
        epsilon = random.uniform(0, 1)
        if epsilon < 0.1:  # Exploration
            next_move = [random.randint(0, 9)]
        else:  # Exploitation
            new_input_tensor = torch.tensor(on_screen, dtype=torch.float32)
            with torch.no_grad():
                next_move = model(new_input_tensor)
        player.auto_move(int(np.round(next_move[0])) % 10)
    else:
        next_move = np.array([player.lane])

    trial_data = TrialData(on_screen, next_move)

    if network_active:
        for collectible in collectibles:
            if int(np.round(next_move[0])) % 10 == collectible.lane and (collectible.x1 < 100 or collectible.x2 > 50):
                trial_data.score += collectible.price
    else:
        for collectible in collectibles:
            if player.lane == collectible.lane and (collectible.x1 < 100 or collectible.x2 > 50):
                trial_data.score += collectible.price
    print("Score: ", trial_data.score)

    current_game_data.append(trial_data)


# This is the function that moves the collectibles.
def move_collectibles():
    for collectible in collectibles:
        collectible.move()
        if collectible.x1 < 0:
            collectibles.remove(collectible)
            canvas.delete(collectible.id)
        detect_collision(collectible)
    move_player()
    canvas.update()
    if game_active:
        canvas.after(movement_rate, move_collectibles)


def run_game():
    # Get the game screen started
    clear_GUI()
    player.place_avatar()

    # Display the lanes at the left hand side of the screen
    for i in range(10):
        all_labels["lane" + str(i)].place(x=10, y=i * 50)

    # Display the labels for score, move, and generation
    all_labels["move_label"].place(relx=0.01, rely=0.9, relwidth=0.2, relheight=0.05)
    all_labels["move_label"].configure(text="Move: 0")
    all_labels["generation_label"].place(relx=0.21, rely=0.9, relwidth=0.2, relheight=0.05)
    all_labels["generation_label"].configure(text="Generation: 0")
    all_labels["score_label"].configure(text="Score: " + str(score))
    all_labels["score_label"].place(relx=0.4, rely=0.9, relwidth=0.2, relheight=0.05)
    all_buttons["quit_button"].place(relx=0.895, rely=0.935, relwidth=0.1, relheight=0.06)
    all_buttons["quit_button"].configure(command=lambda: build_screen())

    # Begin main loop.
    # Make blocks
    for n in range(100):
        canvas.after(generation_rate * n, lambda x=n: generate_collectible(x))
    # Move blocks, player, and check for collisions
    canvas.after(10, move_collectibles)
    # End game
    canvas.after(delay, end_game)


def start_game():
    print("The game speed is " + str(game_speed))
    clear_GUI()
    all_labels["Loading Screen"].place(relx=0.4, rely=0.4, relwidth=0.2, relheight=0.2)
    root.update()

    global score
    score = 0

    global game_active
    game_active = True

    global current_game_data
    current_game_data = []

    # Train the neural network for the level
    if network_active:
        global model
        model = BrainV2.train_model(player_network)
        model.eval()
    else:
        root.bind("<Up>", player.manual_move)
        root.bind("<Down>", player.manual_move)
        for i in range(10):
            root.bind(str(i), player.manual_move)

    # Start running the game
    run_game()


def purchase_node():
    if player.money >= 50 > player.nodes:
        player.money -= 50
        player.nodes += 1
        all_labels["money_label"].config(text="Money: " + str(player.money))
        all_labels["nodes_label"].config(text="Nodes: " + str(player.nodes))
        if player.money < 50:
            all_buttons["purchase_node"].place_forget()
        build_screen()


def purchase_layer():
    if player.money >= 500 and player.layers < 5:
        player.money -= 500
        player.layers += 1
        all_labels["money_label"].config(text="Money: " + str(player.money))
        all_labels["layers_label"].config(text="Layers: " + str(player.layers))
        if player.money < 500:
            all_buttons["purchase_layer"].place_forget()
        build_screen()


# Define a function that adds a layer to the neural network when the add layer button is clicked. It should also
# add the add and remove buttons for the new layer.
def add_layer():
    if player_network.total_layers() < player.layers and player_network.total_nodes() < player.nodes:
        player_network.layers.append(BrainV2.Layer(1))
        all_frames["layer" + str((len(player_network.layers) - 1))].place(
            relx=0.10 + 0.11 * (len(player_network.layers) - 1), rely=0.2, relwidth=0.09, relheight=0.5)
        for j in range(player_network.layers[(len(player_network.layers) - 1)].nodes):
            tk.Label(
                all_frames["layer" + str((len(player_network.layers) - 1))],
                text="O",
                font=("Helvetica", 16)).place(
                relx=0.1,
                rely=0.5 - 0.05 * player_network.layers[(len(player_network.layers) - 1)].nodes + j * 0.1,
                relwidth=0.8,
                relheight=0.1)
        all_buttons["layer" + str((len(player_network.layers) - 1)) + "_add"].place(
            relx=0.12 + 0.11 * (len(player_network.layers) - 1), rely=0.71, relwidth=0.05, relheight=0.05)
    build_screen()


# Define a function that removes a layer from the neural network when the remove layer button is clicked. It should
# also remove the add and remove buttons for the removed layer.
def remove_layer():
    if len(player_network.layers) > 1:
        player_network.layers.pop()
        for widget in all_frames["layer" + str(len(player_network.layers))].winfo_children():
            widget.destroy()
        all_frames["layer" + str(len(player_network.layers))].place_forget()
        all_buttons["layer" + str(len(player_network.layers)) + "_add"].place_forget()
        all_buttons["layer" + str(len(player_network.layers)) + "_remove"].place_forget()
    build_screen()


# Define a function that adds a node to a layer when the add node button is clicked.
def add_node(layer):
    if (player_network.layers[int(layer)].nodes < 10
            and player_network.total_nodes() < player.nodes):
        player_network.layers[int(layer)].nodes += 1
        for widget in all_frames["layer" + layer].winfo_children():
            widget.destroy()
        all_frames["layer" + layer].place(relx=0.10 + 0.11 * int(layer), rely=0.2, relwidth=0.09, relheight=0.5)
        for j in range(player_network.layers[int(layer)].nodes):
            tk.Label(
                all_frames["layer" + layer],
                text="O",
                font=("Helvetica", 16)).place(
                relx=0.1,
                rely=0.5 - 0.05 * player_network.layers[int(layer)].nodes + j * 0.1,
                relwidth=0.8,
                relheight=0.1)
    build_screen()


# Define a function that removes a node from a layer when the remove node button is clicked.
def remove_node(layer):
    if player_network.layers[int(layer)].nodes > 1:
        player_network.layers[int(layer)].nodes -= 1
        for widget in all_frames["layer" + layer].winfo_children():
            widget.destroy()
        all_frames["layer" + layer].place(relx=0.10 + 0.11 * int(layer), rely=0.2, relwidth=0.09, relheight=0.5)
        for j in range(player_network.layers[int(layer)].nodes):
            tk.Label(
                all_frames["layer" + layer],
                text="O",
                font=("Helvetica", 16)).place(
                relx=0.1,
                rely=0.5 - 0.05 * player_network.layers[int(layer)].nodes + j * 0.1,
                relwidth=0.8,
                relheight=0.1)
        build_screen()


# This displays the screen where the player can visually build the neural network.
def build_screen():
    # Clear the canvas
    clear_GUI()
    global game_active
    game_active = False

    global network_active
    network_active = True

    global game_speed, generation_rate, movement_rate, player_velocity, delay
    game_speed = 1
    delay = int((screen_width * movement_rate / collectible_speed) / game_speed)
    generation_rate = int(333 * (1 / game_speed))
    movement_rate = int(60 * (1 / game_speed))
    player_velocity = 1 * game_speed

    # Display the labels for nodes, layers, rounds, and money at the top of the screen
    all_labels["nodes_label"].configure(text="Nodes: " + str(player.nodes))
    all_labels["nodes_label"].place(relx=0.01, rely=0.01, relwidth=0.19, relheight=0.05)
    all_labels["layers_label"].configure(text="Layers: " + str(player.layers))
    all_labels["layers_label"].place(relx=0.21, rely=0.01, relwidth=0.19, relheight=0.05)
    all_labels["money_label"].configure(text="Money: " + str(player.money))
    all_labels["money_label"].place(relx=0.41, rely=0.01, relwidth=0.24, relheight=0.05)
    # all_labels["rounds_label"].configure(text="Rounds played: " + str(level.rounds_played))
    # all_labels["rounds_label"].place(relx=0.66, rely=0.01, relwidth=0.33, relheight=0.05)

    # Display the visual representation of the neural net hidden layers.
    for i in range(len(player_network.layers)):
        all_frames["layer" + str(i)].place(relx=0.10 + 0.11 * i, rely=0.2, relwidth=0.09, relheight=0.5)
        for j in range(player_network.layers[i].nodes):
            tk.Label(
                all_frames["layer" + str(i)],
                text="O",
                font=("Helvetica", 16)).place(
                relx=0.1,
                rely=0.5 - 0.05 * player_network.layers[i].nodes + j * 0.1,
                relwidth=0.8,
                relheight=0.1)

    # Place the add layer button and remove layer button to the left of the first layer
    if player_network.total_layers() < player.layers and player_network.total_nodes() < player.nodes:
        all_buttons["add_layer"].place(relx=0.01, rely=0.40, relwidth=0.05, relheight=0.05)
    if len(player_network.layers) > 1:
        all_buttons["remove_layer"].place(relx=0.01, rely=0.45, relwidth=0.05, relheight=0.05)
    all_buttons["remove_layer"].configure(command=remove_layer)
    all_buttons["add_layer"].configure(command=add_layer)

    # Place the add and remove buttons for each layer
    for i in range(len(player_network.layers)):
        if player_network.layers[i].nodes < 10 and player_network.total_nodes() < player.nodes:
            all_buttons["layer" + str(i) + "_add"].place(
                relx=0.12 + 0.11 * i, rely=0.71, relwidth=0.05, relheight=0.05)
        if player_network.layers[i].nodes > 1:
            all_buttons["layer" + str(i) + "_remove"].place(
                relx=0.12 + 0.11 * i, rely=0.76, relwidth=0.05, relheight=0.05)
    for i in range(5):
        all_buttons["layer" + str(i) + "_add"].configure(command=lambda x=i: add_node(str(x)))
        all_buttons["layer" + str(i) + "_remove"].configure(command=lambda x=i: remove_node(str(x)))

    # Display the purchase node button
    if player.money >= 50 > player.nodes:
        all_buttons["purchase_node"].place(relx=0.66, rely=0.25, relwidth=0.33, relheight=0.1)
    all_buttons["purchase_node"].configure(command=purchase_node)

    # Display the purchase layer button
    if player.money >= 500 and player.layers < 5:
        all_buttons["purchase_layer"].place(relx=0.66, rely=0.40, relwidth=0.33, relheight=0.1)
    all_buttons["purchase_layer"].configure(command=purchase_layer)

    # Display the play button
    all_buttons["play_button"].place(relx=0.895, rely=0.935, relwidth=0.1, relheight=0.06)
    all_buttons["play_button"].configure(command=lambda: start_game())
    # Display the quit button
    all_buttons["quit_button"].place(relx=0.01, rely=0.935, relwidth=0.1, relheight=0.06)
    all_buttons["quit_button"].configure(command=main_menu)


# This displays the screen where the player can control the rectangle to catch the collectibles.
def supervise_screen():
    clear_GUI()
    global network_active
    network_active = False

    def set_speed(speed):
        global game_speed, generation_rate, movement_rate, player_velocity, delay
        game_speed = speed
        delay = int((screen_width * movement_rate / collectible_speed) / game_speed)
        generation_rate = int(333 * (1 / game_speed))
        movement_rate = int(60 * (1 / game_speed))
        player_velocity = 1 * game_speed
        start_game()

    # Display the speed buttons
    all_buttons["slow_button"].place(relx=0.2, rely=0.05, relwidth=0.6, relheight=0.19)
    all_buttons["slow_button"].configure(command=lambda: set_speed(1 / 3))
    all_buttons["medium_button"].place(relx=0.2, rely=0.25, relwidth=0.6, relheight=0.19)
    all_buttons["medium_button"].configure(command=lambda: set_speed(2 / 3))
    all_buttons["fast_button"].place(relx=0.2, rely=0.45, relwidth=0.6, relheight=0.19)
    all_buttons["fast_button"].configure(command=lambda: set_speed(1))
    all_buttons["hyper_button"].place(relx=0.2, rely=0.65, relwidth=0.6, relheight=0.19)
    all_buttons["hyper_button"].configure(command=lambda: set_speed(10 / 3))
    all_buttons["quit_button"].place(relx=0.01, rely=0.935, relwidth=0.1, relheight=0.06)
    all_buttons["quit_button"].configure(command=main_menu)
    all_buttons["play_button"].place_forget()


# This is the entry point of the game. It displays the main menu.
def main_menu():
    # Clear the canvas
    clear_GUI()

    # Display the main menu
    all_buttons["supervise_button"].place(relx=0.05, rely=0.05, relwidth=0.5, relheight=0.05)
    all_buttons["supervise_button"].configure(command=supervise_screen)
    all_buttons["autotrain_button"].place(relx=0.05, rely=0.15, relwidth=0.5, relheight=0.05)
    all_buttons["autotrain_button"].configure(command=build_screen)


main_menu()

# Start the Tkinter event loop
root.mainloop()
