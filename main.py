# So, I learned that I can use an extra python script to avoid circular imports... * sigh * I got so far in the other
# files that I don't want to start over. So, here we are. Starting over.

import tkinter as tk
from GUI_V3 import GameGUI
import Game_Logic
from Game_Logic import GameLogic
from Shop_Logic import GameShop

# Set up the window
root = tk.Tk()
root.title("Catch the Collectibles")
root.geometry("600x600")  # Updated geometry

screen_width = 600
screen_height = 500

# Canvas for game
canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg="white")
canvas.pack()

# Initialize game logic and shop first
player = Game_Logic.Player(canvas)
Logic = GameLogic(root, canvas, None, player)  # No GUI yet
Shop = GameShop(player, None)  # No GUI yet

# Initialize GUI and pass game_logic and shop later
GUI = GameGUI(root, canvas, Logic, Shop)  # Pass Logic and Shop here

# Now set the references in the game logic
Logic.game_gui = GUI
Shop.GUI = GUI  # In case you need GUI reference in Shop

# Update the shop and game logic with the GUI
Logic.GUI = GUI  # Update game logic with the GUI
Shop.GUI = GUI  # Update shop with the GUI

GUI.main_menu()

root.mainloop()