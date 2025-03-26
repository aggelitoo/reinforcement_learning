# %%
import numpy as np
import tensorflow as tf
import keras
from keras import layers, regularizers, Model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from MCT_Othello_classes import *
from othello_rl_helper_fcts import *

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
x = residual_block(x, channels=64, kernel_size=(3,3), weight_decay=0.001)

# Flatten the features and output a single scalar value (for a value network)
x = layers.Flatten()(x)
outputs = layers.Dense(1, name="value_output")(x)

# Create the model
value_model = Model(inputs=inputs, outputs=outputs)
value_model.summary()

# Optimizer for the value model
value_model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9),
    loss = keras.losses.MeanSquaredError()
    # metrics = [keras.metrics.RootMeanSquaredError]
)

##########################################################################
#  ---------------- Beginning of Policy NN architecture ------------------ 
##########################################################################

# Policy model architecture
policy_model = keras.Sequential([

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

policy_model.summary()

policy_model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9), 
                    loss=keras.losses.KLDivergence())

# %%
# Loading the model weights
# Load the model weights
# value_model.load_weights('zeroth_value_nn.weights.h5')
# policy_model.load_weights('zeroth_policy_nn.weights.h5')


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

Prerequisites for the following function is to already have pre-trained
value and policy networks. 
"""

# %%
def online_simulation(n, c, value_model, policy_model, nr_cycles, nr_episodes_per_tree = 160, history_val = [], history_pol = []):
    ''' hehiha '''

    trees = []
    episodes = []
    
    value_model_history = [history_val]
    policy_model_history = [history_pol]

    for _ in range(nr_cycles):
        
        ##### Build tree with updated model #####
        tree, replay_buffer = build_tree(n, value_model, policy_model, c, nr_episodes_per_tree)

        trees.append(tree)
        episodes.append(replay_buffer)
        
        ##### Update value network #####
        value_model, value_history_temp = online_value_training(value_model, replay_buffer,
                                                                epochs=20, batch_size=64)
        policy_model, policy_history_temp = online_policy_training(policy_model, tree,
                                                                epochs=50, batch_size=128)
        value_model_history.append(value_history_temp)
        policy_model_history.append(policy_history_temp)

        print('Cycle done!')
        
    
    return trees, episodes, value_model_history, policy_model_history


# %%
trees, episodes, val_hist, pol_hist = online_simulation(n=6, c=0.5,
                                                        value_model=value_model,
                                                        policy_model=policy_model,
                                                        nr_cycles=10,
                                                        nr_episodes_per_tree=25)

# %%
# saving the training from the online training
second_10cycles25episodes_c05_data = [trees, episodes, val_hist, pol_hist]
with open('second_10cycles25episodes_c05_data', 'wb') as handle:
    pickle.dump(second_10cycles25episodes_c05_data,
                handle, protocol=pickle.HIGHEST_PROTOCOL)
    
value_model.save_weights('value_second_10cycles25episodes_c05.weights.h5')
policy_model.save_weights('policy_second_10cycles25episodes_c05.weights.h5')

##########################################################################
# ------------------- Initial training of VALUE NN ----------------------
##########################################################################

# Initial value network data - from completely random games
# Unique board positions from first 100k random games
path = './othello_random_simulations/othello_sim_boards_100000_6x6'
games = read_files(path)

# %%
state_return_tuples = unpack_positions_returns(games)
state_return_tuples = unique_board_positions(state_return_tuples)

X, y = value_predictors_targets(state_return_tuples)

# splitting data into training and validation
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X, y, test_size=0.2,
                                                                    random_state=80085)

train_dataset_value_model = create_dataset(X_train_val, y_train_val, batch_size=256,
                                           shuffle_buffer_size=10000)
val_dataset_value_model = create_dataset(X_test_val, y_test_val, batch_size=256,
                                         shuffle_buffer_size=10000)

# Training the value model
history_value = value_model.fit(
    train_dataset_value_model,
    epochs = 50,
    validation_data=val_dataset_value_model
    # callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)]
)

# with open('initial_value_training_history', 'wb') as handle:
#     pickle.dump(history_value, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # Saving the weights from the first good network
# value_model.save_weights('zeroth_value_nn.weights.h5')

# %%
##########################################################################
#  ------------------ Initial training of Policy NN  --------------------- 
##########################################################################

sav = read_files("./othello_random_simulations/othello_sim_sa_visits")
X, y = policy_predictors_targets(sav, 6)

X_train_pol, X_test_pol, y_train_pol, y_test_pol = train_test_split(X, y, test_size=0.2,
                                                    random_state=80085)

train_dataset_policy_model = create_dataset(X_train_pol, y_train_pol, batch_size=128)
val_dataset_policy_model = create_dataset(X_test_pol, y_test_pol, batch_size=128)
 
history_policy = policy_model.fit(train_dataset_policy_model, 
                        validation_data=val_dataset_policy_model,
                        epochs=500)

# with open('initial_policy_training_history_2', 'wb') as handle:
#     pickle.dump(history_policy, handle, protocol=pickle.HIGHEST_PROTOCOL)

# saving weights from first good network
# policy_model.save_weights('zeroth_policy_nn_2.weights.h5')


##########################################################################
#  -------------- Concatenating history from different runs ------------- 
##########################################################################

# concatenating and plotting history from first and second iteration of VALUE network
initial_value_training_history_1 = read_files('initial_value_training_history')
initial_value_training_history_2 = read_files('initial_value_training_history_2')

complete_value_history_train_loss = initial_value_training_history_1.history['loss'] + initial_value_training_history_2.history['loss']
complete_value_history_val_loss = initial_value_training_history_1.history['val_loss'] + initial_value_training_history_2.history['val_loss']

plt.plot(complete_value_history_train_loss, label='Training loss')
plt.plot(complete_value_history_val_loss, label = 'Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('value network')
plt.legend(loc='upper right')
plt.show()

# %%
# concatenating and plotting history from first and second iteration of POLICY network
initial_policy_training_history_1 = read_files('initial_policy_training_history')
initial_policy_training_history_2 = read_files('initial_policy_training_history_2')

complete_policy_history_train_loss = initial_policy_training_history_1.history['loss'] + initial_policy_training_history_2.history['loss']
complete_policy_history_val_loss = initial_policy_training_history_1.history['val_loss'] + initial_policy_training_history_2.history['val_loss']

plt.plot(complete_policy_history_train_loss, label='Training loss')
plt.plot(complete_policy_history_val_loss, label = 'Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('policy network')
plt.legend(loc='upper right')
plt.show()

