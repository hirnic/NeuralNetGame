# This is the file for all the game logic.

import BrainV2
import random
import numpy as np
import torch


screen_width = 600
screen_height = 500

# Number of lanes (10 horizontal lanes)
num_lanes = 10
lane_height = screen_height // num_lanes  # Height of each lane

game_speed = 1  # Game milliseconds per real millisecond
player_width = 30
player_height = 30
player_velocity = 1 * game_speed  # pixels per millisecond?
player_lane = 5  # Start at the middle lane (lane 5)
player_moves = []  # List containing the next move

collectible_width = 20
collectible_height = 20
collectible_speed = 10  # pixels per movement
generation_rate = int(333 / game_speed)  # milliseconds per generation
movement_rate = int(60 / game_speed)  # milliseconds per movement
delay = int((screen_width * movement_rate / collectible_speed) / game_speed)  # milliseconds
score = 0

model = None  # This is the neural network model

game_active = False  # This determines whether the game is active or not
network_active = False  # This determines whether the player is controlled by a neural network or not
random_mode = False  # This determines whether the player makes random moves or not (generally for unsupervised train)

current_game_data = []


# Here we have the class for the trial data.
class TrialData:
    def __init__(self, on_screen, next_move):
        self.on_screen = on_screen  # This is a list of lists [lane, x_position, velocity, price] for all collectibles on the screen
        self.next_move = next_move  # Int
        self.score = 0  # Int represents the number of points gained due to this action


class Player:
    def __init__(self, canvas):
        self.canvas = canvas
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
        self.lane = lane  # Move directly to a specified lane
        self.y1 = self.lane * lane_height
        self.y2 = self.lane * lane_height + self.height
        # Move the item by dx and dy
        self.canvas.coords(self.id, self.x1, self.y1, self.x2, self.y2)
        self.canvas.update()

    def manual_move(self, event):
        if event.keysym == 'Up' and self.lane > 0:
            self.lane -= 1  # Move up one lane
        elif event.keysym == 'Down' and self.lane < num_lanes - 1:
            self.lane += 1  # Move down one lane
        elif event.keysym in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            self.lane = int(event.keysym)  # Move directly to a specified lane
        self.y1 = self.lane * lane_height
        self.y2 = self.lane * lane_height + self.height
        self.canvas.coords(self.id, self.x1, self.y1, self.x2, self.y2)
        self.canvas.update()

    def place_avatar(self):
        self.id = self.canvas.create_rectangle(50, 5 * lane_height, 50 + player_width,
                                          player_lane * lane_height + player_height, fill="blue")


# Here we have a class for collectibles. It is just a canvas rectangle, but it also has a price associated with it.
class Collectible:
    def __init__(self, price, lane, velocity, canvas):
        self.price = price
        self.lane = lane
        self.velocity = velocity  # float
        self.canvas = canvas
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
        self.canvas.move(self.id, (-1) * collectible_speed * self.velocity, 0)


# This holds all the information about the screen and contains methods for converting to Pytorch-ready data
class Screen:
    def __init__(self, screen_shot):
        self.on_screen = screen_shot  # Usually collectibles

    def to_tensor(self):
        # This converts the collectible objects to a tensor
        tensor = torch.tensor([])
        for block in self.on_screen:
            tensor = torch.cat((tensor, torch.tensor([block.lane, block.x1, block.velocity, block.price],
                                                     dtype=torch.float32)))
        for _ in range(12 - len(tensor) // 4):
            tensor = torch.cat((tensor, torch.tensor([-1, -1, -1, -1], dtype=torch.float32)))
        return tensor[:48]


class GameLogic:
    def __init__(self, root, canvas, GUI, player):
        self.root = root
        self.canvas = canvas
        self.GUI = GUI
        self.player = player
        self.collectibles = []
        self.player_network = BrainV2.Net([BrainV2.Layer(1)], 100)  # Initialize the player network


    def end_game(self):
        global game_active, network_active, random_mode
        random_mode = False
        game_active = False
        network_active = True
        self.canvas.delete("all")
        self.player.id = None
        for data_point in current_game_data:
            self.player_network.training_set = sorted(self.player_network.training_set, key=lambda x: x.score)
            if len(self.player_network.training_set) < self.player_network.memory:
                self.player_network.training_set.append(data_point)
            elif data_point.score > self.player_network.training_set[0].score:
                self.player_network.training_set[0] = data_point
        self.root.unbind("<Up>")
        self.root.unbind("<Down>")
        for i in range(10):
            self.root.unbind(str(i))
        self. GUI.main_menu()

    # This is the function that generates the collectibles.
    def generate_collectible(self):
        price = random.randint(1, 10)
        if price == 10:
            pass
        elif price > 7:
            price = 5
        else:
            price = 1
        collectible = Collectible(price, random.randint(0, num_lanes - 1), random.uniform(1, 3), self.canvas)
        self.collectibles.append(collectible)
        self.canvas.update()


    # This is the function that moves the player and tracks all the data.
    def move_player(self):
        global score
        # We need to convert the screen data to the input data for the neural network.
        # The input data is of the form [[lane, x_position, velocity, price], ...] for each collectible on screen.
        # Keep in mind, the output data is of the form [next move], where next move is an int (0, 9)
        screen_shot = Screen(self.collectibles)
        new_input_tensor = screen_shot.to_tensor()
        if network_active:
            epsilon = random.uniform(0, 1)
            if epsilon < self.player_network.exploration_rate:  # Exploration
                next_move = torch.tensor([random.randint(0, 9)], dtype=torch.float32)
            else:  # Exploitation
                with torch.no_grad():
                    next_move = model(new_input_tensor)
            print("Next Move: ", next_move[0])
        elif random_mode:
            next_move = torch.tensor([random.randint(0, 9)], dtype=torch.float32)
        else:
            next_move = torch.tensor([self.player.lane], dtype=torch.float32)

        self.player.auto_move(int(np.round(next_move[0])) % 10)

        trial_data = TrialData(new_input_tensor, next_move)

        if network_active:
            for collectible in self.collectibles:
                if (int(np.round(next_move[0])) % 10 == collectible.lane
                        and (self.player.x1 < collectible.x1 < self.player.x2
                             or self.player.x1 < collectible.x2 < self.player.x2)):
                    trial_data.score += collectible.price
                    self.player.money += collectible.price
                    self.canvas.delete(collectible.id)
                    self.collectibles.remove(collectible)
                    score += collectible.price
                    self.GUI.all_labels["score_label"].configure(text="Score: " + str(score))
        else:
            for collectible in self.collectibles:
                if ((self.player.x1 < collectible.x1 < self.player.x2
                     or self.player.x1 < collectible.x2 < self.player.x2)
                        and self.player.lane == collectible.lane):
                    trial_data.score += collectible.price
                    self.player.money += collectible.price
                    self.canvas.delete(collectible.id)
                    self.collectibles.remove(collectible)
                    score += collectible.price
                    self.GUI.all_labels["score_label"].configure(text="Score: " + str(score))

        current_game_data.append(trial_data)


    # This is the function that moves the collectibles.
    def move_collectibles(self):
        for collectible in self.collectibles:
            collectible.move()
            if collectible.x1 < 0:
                self.collectibles.remove(collectible)
                self.canvas.delete(collectible.id)
            # detect_collision(collectible)
        self.move_player()
        if not self.collectibles:
            self.end_game()
        self.canvas.update()
        if game_active:
            self.canvas.after(movement_rate, self.move_collectibles)


    def run_game(self):
        # Get the game screen started
        self.GUI.clear_GUI()
        self.player.place_avatar()

        # Display the lanes at the left hand side of the screen
        for i in range(10):
            self.GUI.all_labels["lane" + str(i)].place(x=10, y=i * 50)

        # Display the labels for score, move, and generation
        self.GUI.all_labels["move_label"].place(relx=0.01, rely=0.9, relwidth=0.2, relheight=0.05)
        self.GUI.all_labels["move_label"].configure(text="Move: 0")
        self.GUI.all_labels["generation_label"].place(relx=0.21, rely=0.9, relwidth=0.2, relheight=0.05)
        self.GUI.all_labels["generation_label"].configure(text="Generation: 0")
        self.GUI.all_labels["score_label"].configure(text="Score: " + str(score))
        self.GUI.all_labels["score_label"].place(relx=0.4, rely=0.9, relwidth=0.2, relheight=0.05)
        self.GUI.all_buttons["quit_button"].place(relx=0.895, rely=0.935, relwidth=0.1, relheight=0.06)
        self.GUI.all_buttons["quit_button"].configure(command=lambda: self.GUI.main_menu())

        # Begin main loop.
        # Make blocks
        if random_mode:
            for n in range(1000):
                self.canvas.after(generation_rate * n, self.generate_collectible)
        else:
            for n in range(100):
                self.canvas.after(generation_rate * n, self.generate_collectible)
        # Move blocks, player, and check for collisions
        self.canvas.after(10, self.move_collectibles)


    def start_game(self):
        self.GUI.clear_GUI()
        self.GUI.all_labels["Loading Screen"].place(relx=0.4, rely=0.4, relwidth=0.2, relheight=0.2)
        self.root.update()

        global score, game_active, current_game_data
        score = 0
        game_active = True
        current_game_data = []

        # Train the neural network for the level
        if network_active:
            global model, game_speed, generation_rate, movement_rate, player_velocity, delay
            game_speed = 1
            delay = int((screen_width * movement_rate / collectible_speed) / game_speed)
            generation_rate = int(333 * (1 / game_speed))
            movement_rate = int(60 * (1 / game_speed))
            player_velocity = 1 * game_speed

            model = BrainV2.train_model(self.player_network)
            model.eval()
        else:
            self.root.bind("<Up>", self.player.manual_move)
            self.root.bind("<Down>", self.player.manual_move)
            for i in range(10):
                self.root.bind(str(i), self.player.manual_move)

        # Start running the game
        self.run_game()


    def set_speed(self, speed, supervised=True):
        global game_speed, generation_rate, movement_rate, player_velocity, delay
        game_speed = speed
        delay = int((screen_width * movement_rate / collectible_speed) / game_speed)
        generation_rate = int(333 * (1 / game_speed))
        movement_rate = int(60 * (1 / game_speed))
        player_velocity = 1 * game_speed
        global network_active
        network_active = False
        if not supervised:
            global random_mode
            random_mode = True
        self.start_game()
