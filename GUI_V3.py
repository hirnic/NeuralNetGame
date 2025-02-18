# This is the graphical user interface the self.game_logic.player will be using.

import BrainV2
import tkinter as tk


class GameGUI:
    def __init__(self, root, canvas, game_logic, shop):
        self.root = root
        self.canvas = canvas
        self.game_logic = game_logic
        self.shop = shop
        self.all_frames = {"layer0": tk.Frame(self.root),
                  "layer1": tk.Frame(self.root),
                  "layer2": tk.Frame(self.root),
                  "layer3": tk.Frame(self.root),
                  "layer4": tk.Frame(self.root),
                  }
        self.all_labels = {"Loading Screen": tk.Label(self.root, text="Loading...", font=("Helvetica", 16)),
                  "lane0": tk.Label(self.root, text="0", font=("Helvetica", 16)),
                  "lane1": tk.Label(self.root, text="1", font=("Helvetica", 16)),
                  "lane2": tk.Label(self.root, text="2", font=("Helvetica", 16)),
                  "lane3": tk.Label(self.root, text="3", font=("Helvetica", 16)),
                  "lane4": tk.Label(self.root, text="4", font=("Helvetica", 16)),
                  "lane5": tk.Label(self.root, text="5", font=("Helvetica", 16)),
                  "lane6": tk.Label(self.root, text="6", font=("Helvetica", 16)),
                  "lane7": tk.Label(self.root, text="7", font=("Helvetica", 16)),
                  "lane8": tk.Label(self.root, text="8", font=("Helvetica", 16)),
                  "lane9": tk.Label(self.root, text="9", font=("Helvetica", 16)),
                  "score_label": tk.Label(self.root, text="Score: 0", font=("Helvetica", 16)),
                  "move_label": tk.Label(self.root, text="Move: 0", font=("Helvetica", 12)),
                  "generation_label": tk.Label(self.root, text="Generation: 0", font=("Helvetica", 12)),

                  "money_label": tk.Label(self.root, text="Money: 0", font=("Helvetica", 16)),
                  "nodes_label": tk.Label(self.root, text="Nodes: 1", font=("Helvetica", 16)),
                  "layers_label": tk.Label(self.root, text="Layers: 1", font=("Helvetica", 16)),
                  "rounds_label": tk.Label(self.root, text="Rounds played: 0", font=("Helvetica", 16)),
                  }
        self.all_buttons = {
        "play_button": tk.Button(self.root, text="Play", font=("Helvetica", 20)),
        "quit_button": tk.Button(self.root, text="Quit", font=("Helvetica", 20)),

        "supervise_button": tk.Button(self.root, text="Supervised Training", font=("Helvetica", 20)),
        "unsupervise_button": tk.Button(self.root, text="Unsupervised Training", font=("Helvetica", 20)),
        "autotrain_button": tk.Button(self.root, text="Build Network", font=("Helvetica", 20)),

        "slow_button": tk.Button(self.root, text="Slow", font=("Helvetica", 20)),
        "medium_button": tk.Button(self.root, text="Medium", font=("Helvetica", 20)),
        "fast_button": tk.Button(self.root, text="Fast", font=("Helvetica", 20)),
        "hyper_button": tk.Button(self.root, text="Hyper", font=("Helvetica", 20)),

        "purchase_node": tk.Button(self.root, text="Purchase Node", font=("Helvetica", 20)),
        "purchase_layer": tk.Button(self.root, text="Purchase Layer", font=("Helvetica", 20)),

        "add_layer": tk.Button(self.root, text="+", font=("Helvetica", 20)),
        "remove_layer": tk.Button(self.root, text="-", font=("Helvetica", 20)),
        "layer0_add": tk.Button(self.root, text="+", font=("Helvetica", 20)),
        "layer0_remove": tk.Button(self.root, text="-", font=("Helvetica", 20)),
        "layer1_add": tk.Button(self.root, text="+", font=("Helvetica", 20)),
        "layer1_remove": tk.Button(self.root, text="-", font=("Helvetica", 20)),
        "layer2_add": tk.Button(self.root, text="+", font=("Helvetica", 20)),
        "layer2_remove": tk.Button(self.root, text="-", font=("Helvetica", 20)),
        "layer3_add": tk.Button(self.root, text="+", font=("Helvetica", 20)),
        "layer3_remove": tk.Button(self.root, text="-", font=("Helvetica", 20)),
        "layer4_add": tk.Button(self.root, text="+", font=("Helvetica", 20)),
        "layer4_remove": tk.Button(self.root, text="-", font=("Helvetica", 20)),
    }

    def clear_GUI(self):
        self.canvas.delete("all")
        for frame in self.all_frames:
            self.all_frames[frame].place_forget()
        for label in self.all_labels:
            self.all_labels[label].place_forget()
        for button in self.all_buttons:
            self.all_buttons[button].place_forget()
        # Clear the collectibles
        for collectible in self.game_logic.collectibles:
            self.canvas.delete(collectible.id)
        self.game_logic.collectibles.clear()
        # Clear the self.game_logic.player
        if self.game_logic.player is not None:
            self.canvas.delete(self.game_logic.player.id)
            self.game_logic.player.id = None

    # Define a function that adds a layer to the neural network when the add layer button is clicked. It should also
    # add the add and remove buttons for the new layer.
    def add_layer(self):
        if self.game_logic.player_network.total_layers() < self.game_logic.player.layers and self.game_logic.player_network.total_nodes() < self.game_logic.player.nodes:
            self.game_logic.player_network.layers.append(BrainV2.Layer(1))
            self.all_frames["layer" + str((len(self.game_logic.player_network.layers) - 1))].place(
                relx=0.10 + 0.11 * (len(self.game_logic.player_network.layers) - 1), rely=0.2, relwidth=0.09, relheight=0.5)
            for j in range(self.game_logic.player_network.layers[(len(self.game_logic.player_network.layers) - 1)].nodes):
                tk.Label(
                    self.all_frames["layer" + str((len(self.game_logic.player_network.layers) - 1))],
                    text="O",
                    font=("Helvetica", 16)).place(
                    relx=0.1,
                    rely=0.5 - 0.05 * self.game_logic.player_network.layers[(len(self.game_logic.player_network.layers) - 1)].nodes + j * 0.1,
                    relwidth=0.8,
                    relheight=0.1)
            self.all_buttons["layer" + str((len(self.game_logic.player_network.layers) - 1)) + "_add"].place(
                relx=0.12 + 0.11 * (len(self.game_logic.player_network.layers) - 1), rely=0.71, relwidth=0.05, relheight=0.05)
        self.build_screen()

    # Define a function that removes a layer from the neural network when the remove layer button is clicked. It should
    # also remove the add and remove buttons for the removed layer.
    def remove_layer(self):
        if len(self.game_logic.player_network.layers) > 1:
            self.game_logic.player_network.layers.pop()
            for widget in self.all_frames["layer" + str(len(self.game_logic.player_network.layers))].winfo_children():
                widget.destroy()
            self.all_frames["layer" + str(len(self.game_logic.player_network.layers))].place_forget()
            self.all_buttons["layer" + str(len(self.game_logic.player_network.layers)) + "_add"].place_forget()
            self.all_buttons["layer" + str(len(self.game_logic.player_network.layers)) + "_remove"].place_forget()
        self.build_screen()

    # Define a function that adds a node to a layer when the add node button is clicked.
    def add_node(self, layer):
        if (self.game_logic.player_network.layers[int(layer)].nodes < 10
                and self.game_logic.player_network.total_nodes() < self.game_logic.player.nodes):
            self.game_logic.player_network.layers[int(layer)].nodes += 1
            for widget in self.all_frames["layer" + layer].winfo_children():
                widget.destroy()
            self.all_frames["layer" + layer].place(relx=0.10 + 0.11 * int(layer), rely=0.2, relwidth=0.09, relheight=0.5)
            for j in range(self.game_logic.player_network.layers[int(layer)].nodes):
                tk.Label(
                    self.all_frames["layer" + layer],
                    text="O",
                    font=("Helvetica", 16)).place(
                    relx=0.1,
                    rely=0.5 - 0.05 * self.game_logic.player_network.layers[int(layer)].nodes + j * 0.1,
                    relwidth=0.8,
                    relheight=0.1)
        self.build_screen()

    # Define a function that removes a node from a layer when the remove node button is clicked.
    def remove_node(self, layer):
        if self.game_logic.player_network.layers[int(layer)].nodes > 1:
            self.game_logic.player_network.layers[int(layer)].nodes -= 1
            for widget in self.all_frames["layer" + layer].winfo_children():
                widget.destroy()
            self.all_frames["layer" + layer].place(relx=0.10 + 0.11 * int(layer), rely=0.2, relwidth=0.09, relheight=0.5)
            for j in range(self.game_logic.player_network.layers[int(layer)].nodes):
                tk.Label(
                    self.all_frames["layer" + layer],
                    text="O",
                    font=("Helvetica", 16)).place(
                    relx=0.1,
                    rely=0.5 - 0.05 * self.game_logic.player_network.layers[int(layer)].nodes + j * 0.1,
                    relwidth=0.8,
                    relheight=0.1)
            self.build_screen()

    # This displays the screen where the self.game_logic.player can visually build the neural network.
    def build_screen(self):
        # Clear the canvas
        self.clear_GUI()


        # Display the labels for nodes, layers, rounds, and money at the top of the screen
        self.all_labels["nodes_label"].configure(text="Nodes: " + str(self.game_logic.player.nodes))
        self.all_labels["nodes_label"].place(relx=0.01, rely=0.01, relwidth=0.19, relheight=0.05)
        self.all_labels["layers_label"].configure(text="Layers: " + str(self.game_logic.player.layers))
        self.all_labels["layers_label"].place(relx=0.21, rely=0.01, relwidth=0.19, relheight=0.05)
        self.all_labels["money_label"].configure(text="Money: " + str(self.game_logic.player.money))
        self.all_labels["money_label"].place(relx=0.41, rely=0.01, relwidth=0.24, relheight=0.05)
        # self.all_labels["rounds_label"].configure(text="Rounds played: " + str(level.rounds_played))
        # self.all_labels["rounds_label"].place(relx=0.66, rely=0.01, relwidth=0.33, relheight=0.05)

        # Display the visual representation of the neural net hidden layers.
        for i in range(len(self.game_logic.player_network.layers)):
            self.all_frames["layer" + str(i)].place(relx=0.10 + 0.11 * i, rely=0.2, relwidth=0.09, relheight=0.5)
            for j in range(self.game_logic.player_network.layers[i].nodes):
                tk.Label(
                    self.all_frames["layer" + str(i)],
                    text="O",
                    font=("Helvetica", 16)).place(
                    relx=0.1,
                    rely=0.5 - 0.05 * self.game_logic.player_network.layers[i].nodes + j * 0.1,
                    relwidth=0.8,
                    relheight=0.1)

        # Place the add layer button and remove layer button to the left of the first layer
        if self.game_logic.player_network.total_layers() < self.game_logic.player.layers and self.game_logic.player_network.total_nodes() < self.game_logic.player.nodes:
            self.all_buttons["add_layer"].place(relx=0.01, rely=0.40, relwidth=0.05, relheight=0.05)
        if len(self.game_logic.player_network.layers) > 1:
            self.all_buttons["remove_layer"].place(relx=0.01, rely=0.45, relwidth=0.05, relheight=0.05)
        self.all_buttons["remove_layer"].configure(command=self.remove_layer)
        self.all_buttons["add_layer"].configure(command=self.add_layer)

        # Place the add and remove buttons for each layer
        for i in range(len(self.game_logic.player_network.layers)):
            if self.game_logic.player_network.layers[i].nodes < 10 and self.game_logic.player_network.total_nodes() < self.game_logic.player.nodes:
                self.all_buttons["layer" + str(i) + "_add"].place(
                    relx=0.12 + 0.11 * i, rely=0.71, relwidth=0.05, relheight=0.05)
            if self.game_logic.player_network.layers[i].nodes > 1:
                self.all_buttons["layer" + str(i) + "_remove"].place(
                    relx=0.12 + 0.11 * i, rely=0.76, relwidth=0.05, relheight=0.05)
        for i in range(5):
            self.all_buttons["layer" + str(i) + "_add"].configure(command=lambda x=i: self.add_node(str(x)))
            self.all_buttons["layer" + str(i) + "_remove"].configure(command=lambda x=i: self.remove_node(str(x)))

        # Display the purchase node button
        if self.game_logic.player.money >= 50 > self.game_logic.player.nodes:
            self.all_buttons["purchase_node"].place(relx=0.66, rely=0.25, relwidth=0.33, relheight=0.1)
        self.all_buttons["purchase_node"].configure(command=self.shop.purchase_node)

        # Display the purchase layer button
        if self.game_logic.player.money >= 500 and self.game_logic.player.layers < 5:
            self.all_buttons["purchase_layer"].place(relx=0.66, rely=0.40, relwidth=0.33, relheight=0.1)
        self.all_buttons["purchase_layer"].configure(command=self.shop.purchase_layer)

        # Display the play button
        self.all_buttons["play_button"].place(relx=0.895, rely=0.935, relwidth=0.1, relheight=0.06)
        self.all_buttons["play_button"].configure(command=self.game_logic.start_game)
        # Display the quit button
        self.all_buttons["quit_button"].place(relx=0.01, rely=0.935, relwidth=0.1, relheight=0.06)
        self.all_buttons["quit_button"].configure(command=self.main_menu)


    # This displays the screen where the self.game_logic.player can control the rectangle to catch the collectibles.
    def supervise_screen(self):
        self.clear_GUI()

        # Display the speed buttons
        self.all_buttons["slow_button"].place(relx=0.2, rely=0.05, relwidth=0.6, relheight=0.19)
        self.all_buttons["slow_button"].configure(command=lambda: self.game_logic.set_speed(1 / 3))
        self.all_buttons["medium_button"].place(relx=0.2, rely=0.25, relwidth=0.6, relheight=0.19)
        self.all_buttons["medium_button"].configure(command=lambda: self.game_logic.set_speed(2 / 3))
        self.all_buttons["fast_button"].place(relx=0.2, rely=0.45, relwidth=0.6, relheight=0.19)
        self.all_buttons["fast_button"].configure(command=lambda: self.game_logic.set_speed(1))
        self.all_buttons["hyper_button"].place(relx=0.2, rely=0.65, relwidth=0.6, relheight=0.19)
        self.all_buttons["hyper_button"].configure(command=lambda: self.game_logic.set_speed(10 / 3))
        self.all_buttons["quit_button"].place(relx=0.01, rely=0.935, relwidth=0.1, relheight=0.06)
        self.all_buttons["quit_button"].configure(command=self.main_menu)
        self.all_buttons["play_button"].place_forget()


    # This is the entry point of the game. It displays the main menu.
    def main_menu(self):
        # Clear the canvas
        self.clear_GUI()

        # Display the main menu
        self.all_buttons["supervise_button"].place(relx=0.05, rely=0.05, relwidth=0.5, relheight=0.05)
        self.all_buttons["supervise_button"].configure(command=self.supervise_screen)
        # Display the main menu
        self.all_buttons["unsupervise_button"].place(relx=0.05, rely=0.15, relwidth=0.5, relheight=0.05)
        self.all_buttons["unsupervise_button"].configure(command=lambda: self.game_logic.set_speed(10/3, False))
        if self.game_logic.player_network.training_set:
            self.all_buttons["autotrain_button"].place(relx=0.05, rely=0.25, relwidth=0.5, relheight=0.05)
            self.all_buttons["autotrain_button"].configure(command=self.build_screen)
