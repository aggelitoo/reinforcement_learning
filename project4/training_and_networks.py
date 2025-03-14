# %%
import pickle
import numpy as np
import tensorflow as tf
import keras
from keras import layers, regularizers, Model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from mc_tree_search import *
from preprocessing_fcts import *

# %%
###########################################################################
# ------------------ Beginning of VALUE NN architecture -------------------
###########################################################################

# We now want to curate training/validation data sets for NN
height, width, channels = 4, 4, 1

# Input layer
inputs = layers.Input(shape=(height, width, channels))

def simple_residual_block(x, channels=16, kernel_size=(3,3), weight_decay=0.001):
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
inputs = layers.Input(shape=(4, 4, 1))

# Initial convolution block (producing 16 channels)
x = layers.Conv2D(16, kernel_size=(3,3),
                  padding='same', use_bias=False,
                  kernel_regularizer=regularizers.l2(0.001))(inputs)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

# Add two residual blocks with constant 16 channels
x = simple_residual_block(x, channels=16, kernel_size=(3,3), weight_decay=0.001)
# x = simple_residual_block(x, channels=16, kernel_size=(3,3), weight_decay=0.001)

# Flatten the features and output a single scalar value (for a value network)
x = layers.Flatten()(x)
outputs = layers.Dense(1, name="value_output")(x)

# Create the model
value_model = Model(inputs=inputs, outputs=outputs)
value_model.summary()

##########################################################################
# -------------------- End of VALUE NN architecture ----------------------
##########################################################################

# Initial value network data - from completely random games
games_iter0 = read_files()
state_return_tuples = unpack_positions_returns(games_iter0)
state_return_tuples = unique_board_positions(state_return_tuples)
X, y = value_predictors_targets(state_return_tuples)

# splitting data into training and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=80085)

# Optimizer for the value model
value_model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
    loss = keras.losses.MeanSquaredError()
    # metrics = [keras.metrics.RootMeanSquaredError]
)

# Training the value model
history = value_model.fit(
    X_train,
    y_train,
    batch_size = 32,
    epochs = 50,
    validation_data=(X_test, y_test)
)

value_model.save_weights('zeroth_network.weights.h5')

# evaluating the value model
test_loss = value_model.evaluate(X_test, y_test, verbose=2)

# %%
# plotting the training and validation curve for value network
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label = 'Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# %%
##########################################################################
# ----------------------- Functions to build tree -----------------------
##########################################################################

def build_tree(n):
    game = OthelloBoard(n)
    root = MCTSNode(n, game.board, game.to_play)
    episodes = []
    learning_curve = []
    temp_terminal = {}
    # for _ in range(nr_terminal):
    count = 0
    while True:
        terminal_state, reward = expand_tree(root, root)      
        episodes.append((episode(terminal_state)[:-1], reward))
        learning_curve.append(reward)
        temp_terminal[terminal_state] = terminal_state.terminal_visits
        if terminal_state.terminal_visits == 100:
            break

    return episodes, temp_terminal, learning_curve


def expand_tree(root, node):
    if node._untried_actions:  # Not a leaf
        return expand_node(root, node)
    else:
        if node.pass_counter != 2:
            best_child = node.best_child()
            return expand_tree(root, best_child)
        else: # Terminal node/state
            reward = node.find_winner(node.state)
            node.backpropagate(reward)
            node.terminal_visits += 1
            return node, reward


def expand_node(root, node):
    next_node = node.expand()
    q_val = value_model.predict(np.expand_dims(np.array(next_node.state), axis=(0,-1)))[0][0]
    next_node.update_q(q_val)
    next_node.backpropagate(next_node.q_value)
    return expand_tree(root, root)


def episode(node):
    episode_list = [np.array(node.state)]
    if node.parent:
        episode_list.extend(episode(node.parent))
    return episode_list

# %%
replay_buffer_1, term_State, lc = build_tree(4)

# %%
##########################################################################
# ----------------------- Online training section -----------------------
##########################################################################

"""
Given a replay buffer, we want to be able to continously feed new game
information into the value network in the form of mini batches. This
section aims to prepare for thats.

Replay buffer will be in the form of a list of tuples, where the first 
elements in each tuple is a game consisting of a sequence of board positions
and the second element is the observed reward from that game.
"""

value_model.load_weights('zeroth_network.weights.h5')

def online_value_training(value_model, replay_buffer):
    """
    PRUTT
    """
    state_return_tuples = unpack_positions_returns(replay_buffer)
    X_buffer, y_buffer = value_predictors_targets(state_return_tuples)
    X_train_onl, X_test_onl, y_train_onl, y_test_onl = train_test_split(X_buffer, y_buffer,
                                                                        test_size=0.2)

    history = value_model.fit(
        X_train_onl, y_train_onl,               
        batch_size=32,                  
        epochs=10,                       
        validation_data=(X_test_onl, y_test_onl)
    )

    return value_model, history

value_model, history_1 = online_value_training(value_model, replay_buffer_1)

print(value_model.predict(temp))

# %%
