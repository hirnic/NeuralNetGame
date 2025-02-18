# This file is for shopping.

class GameShop:
    def __init__(self, player, GUI):
        self.player = player
        self.GUI = GUI

        
    def purchase_node(self):
        if self.player.money >= 50 > self.player.nodes:
            self.player.money -= 50
            self.player.nodes += 1
            self.GUI.all_labels["money_label"].config(text="Money: " + str(self.player.money))
            self.GUI.all_labels["nodes_label"].config(text="Nodes: " + str(self.player.nodes))
            if self.player.money < 50:
                self.GUI.all_buttons["purchase_node"].place_forget()
            self.GUI.build_screen()


    def purchase_layer(self):
        if self.player.money >= 500 and self.player.layers < 5:
            self.player.money -= 500
            self.player.layers += 1
            self.GUI.all_labels["money_label"].config(text="Money: " + str(self.player.money))
            self.GUI.all_labels["layers_label"].config(text="Layers: " + str(self.player.layers))
            if self.player.money < 500:
                self.GUI.all_buttons["purchase_layer"].place_forget()
            self.GUI.build_screen()

