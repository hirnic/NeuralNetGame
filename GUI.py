# This is the graphical user interface the player will be using.

import Brain
import tkinter as tk
import random
from mpmath import mp

mp.dps = 100  # set number of digits

# Set up the window
root = tk.Tk()
root.title("Catch the Collectibles")
root.geometry("600x600")  # Updated geometry

# Canvas for game
canvas = tk.Canvas(root, width=600, height=500, bg="white")
canvas.pack()

# Number of lanes (10 horizontal lanes)
num_lanes = 10
lane_height = 500 // num_lanes  # Height of each lane

# Game variables
player_width = 30
player_height = 30
player_lane = 5  # Start at the middle lane (lane 5)
player_moves = ""  # String of digits to store the player moves
player_network = Brain.Net([Brain.Layer(1)], 5)  # Initialize the player network
number = ""

collectibles = []
collectible_width = 20
collectible_height = 20
collectible_speed = 10
generation_rate = 333
score = 0

game_active = False

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

              "money_label": tk.Label(root, text="Money: 0", font=("Helvetica", 16)),
              "nodes_label": tk.Label(root, text="Nodes: 1", font=("Helvetica", 16)),
              "layers_label": tk.Label(root, text="Layers: 1", font=("Helvetica", 16)),
              "rounds_label": tk.Label(root, text="Rounds played: 0", font=("Helvetica", 16)),
              }

all_buttons = {
    "play_button": tk.Button(root, text="Play", font=("Helvetica", 20)),
    "quit_button": tk.Button(root, text="Quit", font=("Helvetica", 20)),

    "levels_button": tk.Button(root, text="Levels", font=("Helvetica", 20)),
    "level0": tk.Button(root, text="Level 0", font=("Helvetica", 20)),
    "level1": tk.Button(root, text="Level 1", font=("Helvetica", 20)),
    "level2": tk.Button(root, text="Level 2", font=("Helvetica", 20)),
    "level3": tk.Button(root, text="Level 3", font=("Helvetica", 20)),
    "level4": tk.Button(root, text="Level 4", font=("Helvetica", 20)),
    "level5": tk.Button(root, text="Level 5", font=("Helvetica", 20)),

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


class Level:
    def __init__(self, seed, rounds_played):
        self.seed = seed
        self.rounds_played = rounds_played


current_level = Level("r7", 0)


class Player:
    def __init__(self):
        self.x1 = 50
        self.x2 = 50 + player_width
        self.y1 = player_lane * lane_height
        self.y2 = player_lane * lane_height + player_height
        self.lane = 5
        self.width = player_width
        self.height = player_height
        self.money = 0
        self.nodes = 50
        self.layers = 5
        self.id = None

    def auto_move(self, lane):
        lane_diff = int(lane) - self.lane
        self.lane = int(lane)  # Move directly to a specified lane
        for i in range(1, 13):
            self.y1 = (self.lane - lane_diff * (1 - i / 12)) * lane_height
            self.y2 = (self.lane - lane_diff * (1 - i / 12)) * lane_height + self.height
            canvas.coords(self.id, self.x1, self.y1, self.x2, self.y2)
            canvas.update()
            canvas.after(10)

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
        for i in range(1, 7):
            self.y1 = (self.lane - lane_diff * (1 - i / 6)) * lane_height
            self.y2 = (self.lane - lane_diff * (1 - i / 6)) * lane_height + self.height
            canvas.coords(self.id, self.x1, self.y1, self.x2, self.y2)
            canvas.update()
            canvas.after(10)

    def place_avatar(self):
        self.id = canvas.create_rectangle(50, 5 * lane_height, 50 + player_width,
                                          player_lane * lane_height + player_height, fill="blue")

    def check_collision(self, collectible):
        if (self.x1 < collectible.x1 < self.x2 or self.x1 < collectible.x2 < self.x2) and self.lane == collectible.lane:
            return True
        return False


player = Player()


# Here we have a class for collectibles. It is just a canvas rectangle, but it also has a price associated with it.
class Collectible:
    def __init__(self, price, lane):
        y1 = lane * lane_height + 5
        self.price = price
        self.lane = lane
        self.x1 = 600
        self.x2 = 600 + collectible_width
        self.y1 = y1
        self.y2 = y1 + collectible_height

        def rgb_to_hex(r, g, b):
            return f'#{r:02x}{g:02x}{b:02x}'

        self.id = canvas.create_rectangle(600, y1, 600 + collectible_width, y1 + collectible_height,
                                          fill=rgb_to_hex(2 * price ** 2, int(255 / (2 * price ** 2)), 0))

    def move(self):
        self.x1 -= collectible_speed
        self.x2 -= collectible_speed
        canvas.move(self.id, -collectible_speed, 0)


def clearGUI():
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


def move_player(lane):
    delay = int(600 / collectible_speed * 60 - generation_rate)
    if game_active:
        canvas.after(delay, lambda: player.auto_move(player_moves[int(lane)]))


n = 4


# Function to create new collectibles
def create_collectible():
    global n
    # lane = random.randint(0, 9)
    lane = int(number[n])
    price_tag = random.randint(1, 100)
    if price_tag < 75:
        collectibles.append(Collectible(1, lane))
    elif price_tag < 95:
        collectibles.append(Collectible(5, lane))
    else:
        collectibles.append(Collectible(10, lane))
    if n < 100 and game_active:
        canvas.after(generation_rate, create_collectible)
    elif game_active:
        delay = 2 * int(600 / collectible_speed * 60)
        canvas.after(delay, lambda: build_screen(current_level))
        player.money += score
        current_level.rounds_played += 1
    n += 1
    move_player(n - 5)


# Function to move collectibles
def move_collectibles():
    global score
    for collectible in collectibles[:]:
        if game_active:
            collectible.move()  # Move left
        else:
            canvas.delete(collectible.id)
            collectibles.remove(collectible)

        # Check if the collectible is off screen
        if collectible.x2 < 0:
            canvas.delete(collectible.id)
            collectibles.remove(collectible)

        # Check for collision with player
        if player.check_collision(collectible):
            score += collectible.price
            canvas.delete(collectible.id)
            collectibles.remove(collectible)
            # Update score label
            all_labels["score_label"].config(text=f"Score: {score}")

    if game_active:
        root.after(60, move_collectibles)


# Function to start the game
def start_game(level):
    clearGUI()
    all_labels["Loading Screen"].place(relx=0.4, rely=0.4, relwidth=0.2, relheight=0.2)
    global player_moves
    global number
    global n
    n = 4
    global game_active

    if level.seed[0] == "r":
        number = str(mp.mpf(1) / int(level.seed[1:]))
    else:
        number = str(eval("mp." + level.seed[1:]))

    # Train the neural net and get the player moves
    if level.seed[0] == "r":
        player_moves = Brain.format_prediction(
            Brain.train_model(str(mp.mpf(1) / int(level.seed[1:])), level.rounds_played, player_network, 5))
    else:
        player_moves = Brain.format_prediction(
            Brain.train_model(str(eval("mp." + level.seed[1:])), level.rounds_played, player_network, 5))
    print("Player Moves: ", player_moves)

    game_active = True
    clearGUI()

    # Show the score
    global player, score
    score = 0
    all_labels["score_label"].config(text=f"Score: {score}")

    # Display the lanes
    for i in range(10):
        all_labels["lane" + str(i)].place(x=10, y=i * 50)

    # Create the player
    player.place_avatar()
    all_labels["score_label"].place(relx=0.4, rely=0.9, relwidth=0.2, relheight=0.1)  # Display score
    all_buttons["quit_button"].place(relx=0.895, rely=0.935, relwidth=0.1, relheight=0.06)
    all_buttons["quit_button"].configure(command=lambda: build_screen(level))

    # Bind keys for manual movement
    root.bind("<Up>", player.manual_move)
    root.bind("<Down>", player.manual_move)
    for i in range(10):
        root.bind(str(i), player.manual_move)

    # Start the game loop
    create_collectible()
    move_collectibles()


# This is the screen that displays when the game is loading
def loading_screen():
    clearGUI()


def purchase_node():
    if player.money >= 50 > player.nodes:
        player.money -= 50
        player.nodes += 1
        all_labels["money_label"].config(text="Money: " + str(player.money))
        all_labels["nodes_label"].config(text="Nodes: " + str(player.nodes))
        if player.money < 50:
            all_buttons["purchase_node"].place_forget()
        build_screen(current_level)


def purchase_layer():
    if player.money >= 500 and player.layers < 5:
        player.money -= 500
        player.layers += 1
        all_labels["money_label"].config(text="Money: " + str(player.money))
        all_labels["layers_label"].config(text="Layers: " + str(player.layers))
        if player.money < 500:
            all_buttons["purchase_layer"].place_forget()
        build_screen(current_level)


# Define a function that adds a layer to the neural network when the add layer button is clicked. It should also
# add the add and remove buttons for the new layer.
def add_layer():
    if player_network.total_layers() < player.layers and player_network.total_nodes() < player.nodes:
        player_network.layers.append(Brain.Layer(1))
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
    build_screen(current_level)


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
    build_screen(current_level)


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
    build_screen(current_level)


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
        build_screen(current_level)


# This displays the screen where the player can visually build the neural network.
def build_screen(level):
    # Clear the canvas
    clearGUI()
    global game_active
    game_active = False
    global current_level
    current_level = level

    # Display the labels for nodes, layers, rounds, and money at the top of the screen
    all_labels["nodes_label"].configure(text="Nodes: " + str(player.nodes))
    all_labels["nodes_label"].place(relx=0.01, rely=0.01, relwidth=0.19, relheight=0.05)
    all_labels["layers_label"].configure(text="Layers: " + str(player.layers))
    all_labels["layers_label"].place(relx=0.21, rely=0.01, relwidth=0.19, relheight=0.05)
    all_labels["money_label"].configure(text="Money: " + str(player.money))
    all_labels["money_label"].place(relx=0.41, rely=0.01, relwidth=0.24, relheight=0.05)
    all_labels["rounds_label"].configure(text="Rounds played: " + str(level.rounds_played))
    all_labels["rounds_label"].place(relx=0.66, rely=0.01, relwidth=0.33, relheight=0.05)

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
    all_buttons["play_button"].configure(command=lambda: start_game(level))
    # Display the quit button
    all_buttons["quit_button"].place(relx=0.01, rely=0.935, relwidth=0.1, relheight=0.06)
    all_buttons["quit_button"].configure(command=main_menu)


# This displays the levels of the game.
def level_screen():
    # Clear the canvas
    clearGUI()

    # Display the levels
    all_buttons["quit_button"].place(relx=0.895, rely=0.935, relwidth=0.1, relheight=0.06)
    all_buttons["quit_button"].configure(command=main_menu)

    # Display the levels
    for i in range(6):
        all_buttons["level" + str(i)].place(relx=0.1, rely=0.1 + 0.1 * i, relwidth=0.8, relheight=0.08)


# This is the entry point of the game. It displays the main menu.
def main_menu():
    # Clear the canvas
    clearGUI()

    # Display the main menu
    all_buttons["levels_button"].place(relx=0.05, rely=0.05, relwidth=0.15, relheight=0.05)
    all_buttons["levels_button"].configure(command=level_screen)


all_buttons["level0"].configure(command=lambda: build_screen(Level("r7", 0)))
all_buttons["level1"].configure(command=lambda: build_screen(Level("r43", 0)))
all_buttons["level2"].configure(command=lambda: build_screen(Level("asqrt(2)", 0)))
all_buttons["level3"].configure(command=lambda: build_screen(Level("tln(2)", 0)))
all_buttons["level4"].configure(command=lambda: build_screen(Level("tpi", 0)))
all_buttons["level5"].configure(command=lambda: build_screen(Level("te", 0)))

# Add an option menu for the seed
# seed_var = tk.StringVar(value="Seed")
# seed_label = tk.OptionMenu(root, seed_var,
#                            *["r7", "r31", "r17", "r19", "r23", "r43", "r29", "r29", "r42", "r97",
#                              "asqrt(2)", "aphi",
#                              "te", "tpi"])

main_menu()

# Start the Tkinter event loop
root.mainloop()
