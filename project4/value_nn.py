# %%
import pickle
import numpy as np
import tensorflow as tf
import keras
from keras import layers, regularizers, Model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Unpickling and saving the games to a list
def read_files():
    with open("othello_sim_boards_100000", "rb") as fp: 
        boards = pickle.load(fp)
    return boards

games = read_files()

# each element in data is a tuple of a board position (state)
# and label (-1, 0 or 1), denoting how that particular game ended
data = []
for game in games:
    for position in game[0]:
        data.append((position, game[1]))

# We only want to keep unique (board, reward) tuples
unique_dict = {}
for state, target in data:
    # Convert the matrix to a hashable representation using .tobytes()
    key = (state.tobytes(), target)
    
    # Only add unique tuples
    if key not in unique_dict:
        unique_dict[key] = (state, target)

# Saving the unique (board, reward) tuples as data list
data = list(unique_dict.values())

# %%
# We now want to curate training/validation data sets for NN
height, width, channels = 4, 4, 1

# order data into predictors and targets
X = np.array([x for x, _ in data])
y = np.array([y for _, y in data])

# expanding X to include channel
X = np.expand_dims(X, axis=-1)

# splitting data into training and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=80085)

# %%
######################################################################
# ------------------ Beginning of NN architecture -------------------
######################################################################

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

######################################################################
# -------------------- End of NN architecture ----------------------
######################################################################

# Optimizer for the model and training
value_model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
    loss = keras.losses.MeanSquaredError()
    # metrics = [keras.metrics.RootMeanSquaredError]
)

# Training the model
history = value_model.fit(
    X_train,
    y_train,
    batch_size = 32,
    epochs = 50,
    validation_data=(X_test, y_test)
)

# evaluating the model
test_loss = value_model.evaluate(X_test, y_test, verbose=2)

# %%
# plotting the training and validation curve
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label = 'Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
# %%
