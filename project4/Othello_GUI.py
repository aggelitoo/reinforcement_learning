# %%
import tkinter as tk
import numpy as np
from functools import partial
import keras
from keras import layers, regularizers, Model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from MCT_Othello_classes import * 
from othelloBoard import OthelloBoard as OB
from othello_rl_helper_fcts import *

# %%
class OthelloGUI():
    def __init__(self, starting_tree, manager, board_size, value_model, policy_model, mapped_actions, nr_simulations=1000, c=0.5):
        self.value_model = value_model  # The neural network for value estimation
        self.policy_model = policy_model  # The neural network for policy prediction
        self.mapped_actions = mapped_actions
        self.nr_simulations = nr_simulations
        self.c = c  # Exploration constant for UCB

        self.manager = manager
        self.board_size = board_size
        # self.agent = mcts_agent  # AI player
        self.game = OB(board_size)  # Game board
        # self.buttons = [[None for _ in range(board_size)] for _ in range(board_size)]
        self.current_node = starting_tree # starting at the root, this keeps track where in the tree we are
        # self.current_expanded_tree = starting_tree
        self.manager.title("Othello with MCTS Agent")
        self.create_board()
        self.update_board()

    def create_board(self):
        """Creates the Othello board."""
        self.manager.configure(bg="#006400")  # Dark green background for the window
        
        self.canvas = tk.Canvas(self.manager, width=self.board_size * 60, height=self.board_size * 60, bg="#006400")
        self.canvas.grid(row=0, column=0, columnspan=self.board_size)
        self.canvas.bind("<Button-1>", self.human_move)  # Bind click event to board
        
        self.update_board()

    def update_board(self):
        """Updates the board UI based on the game state with the playing bricks/pieces."""
        self.canvas.delete("all")  # Clear the board

        cell_size = 60  # Size of each square
        offset = 5  # Padding for circle inside the square

        for row in range(self.board_size):
            for col in range(self.board_size):
                x1, y1 = col * cell_size, row * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size

                # Draw green board squares
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="#228B22", outline="black")

                # Draw pieces
                if self.game.board[col][row] == 1:  # Black piece
                    self.canvas.create_oval(x1 + offset, y1 + offset, x2 - offset, y2 - offset, fill="black", outline="black")
                elif self.game.board[col][row] == -1:  # White piece
                    self.canvas.create_oval(x1 + offset, y1 + offset, x2 - offset, y2 - offset, fill="white", outline="black")
                    
    def human_move(self, event):
        """Handles the human player's move."""
        cell_size = 60  # Same size used for drawing
        col = event.x // cell_size
        row = event.y // cell_size

        if (row, col) in self.game.move_generator():  # Check move validity
            self.game.make_move((row, col))
            self.game.to_play *= -1  # Switch turn
            self.update_board()
            
            if not self.game.move_generator():  # If no valid moves, pass turn
                self.game.to_play *= -1
                if not self.game.move_generator():
                    self.show_winner()
                    return
                
            # Move down the tree with chosen action
            if self.current_node._untried_actions:
                self.manager.after(3000, self.manager.destroy)
                print("Expanding from Human move")
                for _ in range(self.nr_simulations):
                    self.current_node = self.expand_from_unseen_state()

 
            for child in self.current_node.child_nodes:
                if child.parent_action == (row, col): 
                    print("Move down tree")
                    self.current_node = child

            self.manager.after(500, self.ai_move)  # AI move with delay


    def ai_move(self):
        """Handles AI move by calling MCTS agent."""
        # current_state = self.game.board
        best_move = self.select_best_move()  # AI chooses a move
        
        if best_move:
            self.game.make_move(best_move)
            self.game.to_play *= -1  # Switch turn
            self.update_board()

            if not self.game.move_generator():  # If no valid moves, pass turn
                self.game.to_play *= -1
                if not self.game.move_generator():
                    self.show_winner()

            # Move down the tree with chosen action
            if self.current_node._untried_actions:
                self.manager.after(3000, self.manager.destroy)
                print("Expanding ai move")
                for _ in range(self.nr_simulations):
                    self.current_node = self.expand_from_unseen_state()

          
            for child in self.current_node.child_nodes:
                if child.parent_action == best_move: 
                    print("Move down tree")
                    self.current_node = child
        
    def select_best_move(self):
        """
        Runs MCTS and selects the best move from the current node.
        """
        if self.current_node._untried_actions:
            self.manager.after(3000, self.manager.destroy)
            print("Expanding Best move")
            for _ in range(self.nr_simulations):
                self.current_node = self.expand_from_unseen_state()

        best_child = self.current_node.best_child(self.c)
        return best_child.parent_action
    
    def expand_from_unseen_state(self):
        stack = [(self.current_node, self.current_node)]  # Stack holds pairs of (root, root)
        while stack:
            expanded_node, current_node = stack.pop()  # Get the last node to process

            if current_node._untried_actions:  # If the node has untried actions, expand it
                next_node = current_node.expand()
                # Instead of using recursion, use a random value for the q_val
                q_val = np.array(self.value_model(np.expand_dims(np.array(next_node.state), axis=(0, -1)))[0][0])
                next_node.update_q(q_val)
                next_node.backpropagate(next_node.q_value)

                # Push the root back into the stack to continue processing from the root
                stack.append((expanded_node, expanded_node))

            else:  # If no untried actions, move to the best child or terminal node
                if current_node.pass_counter != 2:
                    
                    flatten_state = np.ndarray.flatten(np.array(current_node.state))
                    policy_dist = np.array(self.policy_model(np.expand_dims(flatten_state, axis=0))[0])
                    for child in current_node.child_nodes:
                        child.p_action = policy_dist[self.mapped_actions[child.parent_action]]

                    best_child = current_node.best_child(self.c)
                    stack.append((expanded_node, best_child))  # Continue with the best child

                else:  # Terminal node/state
                    reward = current_node.find_winner(current_node.state)
                    current_node.update_q(reward)
                    current_node.backpropagate(reward)
                    current_node.terminal_visits += 1
                    return expanded_node
    

    def show_winner(self):
        """Displays the winner when the game ends."""
        winner = self.game.find_winner(self.game.board)
        if winner == 1:
            msg = "Black (⚫) Wins!"
        elif winner == -1:
            msg = "White (⚪) Wins!"
        else:
            msg = "It's a draw!"
        result_label = tk.Label(self.manager, text=msg, font=("Arial", 16))
        result_label.grid(row=self.board_size, columnspan=self.board_size)






# %%
###########################################################################
# ------------------ Beginning of VALUE NN architecture ------------------
###########################################################################

def residual_block(x, channels=64, kernel_size=(3,3), weight_decay=0.001):
    shortcut = x  # No projection needed if dimensions already match.
    
    x = layers.Conv2D(channels, kernel_size=kernel_size,
                      padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = layers.Conv2D(channels, kernel_size=kernel_size,
                      padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    
    # Direct addition is fine here.
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    
    return x

# Build the model
inputs = layers.Input(shape=(6, 6, 1)) # 6x6 Othello

# Initial convolution block (producing 16 channels)
x = layers.Conv2D(64, kernel_size=(3,3),
                  padding='same', use_bias=False,
                  kernel_regularizer=regularizers.l2(0.001))(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

# Add a number of residual blocks with constant 64 channels
x = residual_block(x, channels=64, kernel_size=(3,3), weight_decay=0.001)

# Flatten the features and output a single scalar value (for a value network)
x = layers.Flatten()(x)
outputs = layers.Dense(1, name="value_output")(x)

# Create the model
value_model = Model(inputs=inputs, outputs=outputs)
# value_model.summary()

# Optimizer for the value model
value_model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
    loss = keras.losses.MeanSquaredError()
    # metrics = [keras.metrics.managerMeanSquaredError]
)



##########################################################################
#  ---------------- Beginning of Policy NN architecture ------------------ 
##########################################################################

# Policy model architecture
policy_model = tf.keras.Sequential([

    layers.Dense(128, input_shape=(36,)),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.025),

    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.025),

    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.025),

    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.025),

    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.025),

    layers.Dense(36),
    layers.Activation('softmax')
])

# policy_model.summary()

policy_model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9), 
                 loss=keras.losses.KLDivergence())


# Loading the model weights
# Load the model weights
# Florence trained model
# value_model.load_weights('value_second_30cycles50episodes_c02.weights.h5')
# policy_model.load_weights('policy_second_30cycles50episodes_c02.weights.h5')
# file = read_files("second_30cycles50episodes_c02_data")
# trees, _, _, _ = file


# August trained model
value_model.load_weights('./weights/value_30cycles25episodes_c05.weights.h5')
policy_model.load_weights('./weights/policy_30cycles25episodes_c05.weights.h5')
file = read_files("./data_from_training/complete_training_30cycles25episodes_c05_data")
trees, _, _, _ = file

# %%
# ================== MAIN ==================
if __name__ == "__main__":
    # from your_mcts_agent_module import MCTSAgent  # Import your agent
    # from your_othello_board_module import OthelloBoard  # Import your board class

    trained_tree = trees[-1] # pick the tree that has explored most of the states
    manager = tk.Tk()
    mapped_actions = map_actions_to_integers(6)
    # agent = MCTSAgent(value_model, policy_model, 6, mapped_actions, num_simulations=1, c=0.2)  # Initialize your trained agent
    board_size = 6  # Othello standard size
    app = OthelloGUI(trained_tree, manager, board_size, value_model, policy_model, mapped_actions, nr_simulations=1, c=0.5)
    manager.mainloop()
# %%
