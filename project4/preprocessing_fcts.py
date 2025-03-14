import numpy as np

def unpack_positions_returns(games):
    """
    Takes in games in the form of a list of tuples where the first element of the
    tuple is a list of board positions representing one full game, and the second
    element in the tuple is the reward (-1, 0, or 1) from that game.

    Returns a list of tuples where each tuple consists of a single board position
    and the reward associated with the game where that board position came from.
    This is a (state s_t, return G_t) tuple in RL terms.
    """
    data = []
    for game in games:
        for position in game[0]:
            data.append((position, game[1]))
    return data

def unique_board_positions(state_return_tuples):
    """
    Takes in list of tuples (state, return) where state is a board position and returns
    a list of only the unique tuples.

    This function will most likely be used only once, on the purely random simulated data.
    """
    unique_dict = {}
    for state, target in state_return_tuples:
        # Convert the matrix to a hashable representation using .tobytes()
        key = (state.tobytes(), target)
        
        # Only add unique tuples
        if key not in unique_dict:
            unique_dict[key] = (state, target)

    # Saving the unique (board, reward) tuples as data list
    return list(unique_dict.values())

def value_predictors_targets(state_return_tuples):
    """
    Takes in list of tuples (state, return) where state is board position and return is
    observed reward from game.

    And returns them in the form of X, y, ready to be used in a model.
    """
    # orders data into predictros and targets
    X = np.array([x for x, _ in state_return_tuples])
    y = np.array([y for _, y in state_return_tuples])

    # expanding X to include #channels=1
    X = np.expand_dims(X, axis=-1)

    return X, y

# Unpickling and saving the games to a list
def read_files():
    with open("othello_sim_boards_100000", "rb") as fp: 
        boards = pickle.load(fp)
    return boards