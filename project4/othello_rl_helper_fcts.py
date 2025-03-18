import numpy as np
import pickle
import random
import tensorflow as tf
from MCT_Othello_classes import *
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

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
def read_files(path):
    with open(path, "rb") as fp: 
        boards = pickle.load(fp)
    return boards

def create_dataset(X, y, batch_size=128, shuffle_buffer_size=10000):
    """
    X and y should already be processed according to above preprocessing
    functions before being passed to this function.

    This function is simply to make training more efficient from memory.
    """
    # Create dataset from tensor slices
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Shuffle the dataset (reshuffles each epoch)
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance optimization
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def map_states_to_action_visits(sav):
    state_to_action_visits = {}
    for state, action, count in sav:
        key = (state.tobytes())
        if key not in state_to_action_visits:
            state_to_action_visits[key] = [(action, count)]
        else:
            state_to_action_visits[key].append((action, count))

    return state_to_action_visits

def map_actions_to_integers(n):
    actions = [(i,j) for i in range(n) for j in range(n)]
    map_actions = dict(zip(actions, [i for i in range(n*n)]))
    return map_actions


def policy_predictors_targets(sav, n):
    """
    Takes in a list of tuples (state, action, count) where state is board position,
    action is the action taken in this state and the count is the number of visits 
    to the next state when taking the action.

    It should be the N(s,a) count from the UCB algorithm. 

    The predictors are the states and the target will be the N(s,a) count
    """
    map_sav = map_states_to_action_visits(sav)
    mapped_actions = map_actions_to_integers(n)
    X = []
    y = []

    for state, value in map_sav.items():
        x = np.frombuffer(state, dtype=int)
        X.append(x)
        y_i = np.zeros(n*n)
        total_visits = 0
        for action, visits in value:
            total_visits += visits
            y_i[mapped_actions[action]] = visits

        y_i = y_i/total_visits
        y.append(y_i)

    X = np.array(X)
    y = np.array(y)

    return X, y

def episode(node):
    episode_list = [np.array(node.state)]
    if node.parent:
        episode_list.extend(episode(node.parent))
    return episode_list

def treetraversal(node, res):
    """
    Recursive helper function to state_action_visits. 
    """
    if not node:
        return

    if node.parent != None:
        tup = (np.array(node.parent.state), node.parent_action, node._nof_visits)
        res.append(tup)

    for child in node.child_nodes:
        treetraversal(child, res)


def state_action_visits(root):
    """ 
    Given MCTS, recursively goes through the tree and returns a list with tuples 
    containing state s, action a and N(s,a).
    """
    res = []
    treetraversal(root, res)
    return res


def results_distribution(episodes):
    """
    Given episodes, return a tuple with win, draw and loss.
    """
    win = 0
    loss = 0
    draw = 0
    
    for i in range(len(episodes)):
        if episodes[i][1] == -1:
            loss += 1

        elif episodes[i][1] == 0:
            draw += 1

        elif episodes[i][1] == 1:
            win += 1
    
    return (win, draw, loss)


def terminal_state_visits(episodes):
    """ 
    Given episodes, returns a dictionary with terminal state as key
    and number of visits in that terminal state as value.
    """
    term_state_visits = {}
    
    for i in range(len(episodes)):
        term_state = (episodes[i][0][0].tobytes(),)   # Make terminal state hashable
        term_state_visits[term_state] = term_state_visits.get(term_state, 0) + 1
    
    return term_state_visits


def bar_plot_term_states(episodes):
    """ 
    Given episodes, plot the distribution over how often a terminal state is visited.
    """
    win, draw, loss = results_distribution(episodes)
    tot_games = len(episodes)
    win_proc = np.round(win/tot_games*100, decimals=4)
    draw_proc = np.round(draw/tot_games*100, decimals=4)
    loss_proc = np.round(loss/tot_games*100, decimals=4)
    
    term_state_visits = terminal_state_visits(episodes)
    x_vals = [i for i in range(len(term_state_visits))]
    plt.bar(x_vals, term_state_visits.values())
    
    # Add a text box with game results
    textstr = f"Total games = {tot_games}\nWin: {win_proc}%\nDraw: {draw_proc}%\nLoss: {loss_proc}%"

    # Position the text box
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.68, 0.9, textstr, transform=plt.gca().transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.xlabel("Terminal states")
    plt.ylabel("Number of visits")
    plt.grid()

            
def online_value_training(value_model, replay_buffer, epochs=20, batch_size=64):
    """
    Given a replay buffer in the form of list of tuples, (where each tuple consists of
    a list of board positions from one game, and the reward from said game) and a value model,
    trains this model using the replay buffer through given number of epochs.

    Returns the trained value_model
    """
    state_return_tuples = unpack_positions_returns(replay_buffer)
    X_buffer, y_buffer = value_predictors_targets(state_return_tuples)
    X_train_onl, X_test_onl, y_train_onl, y_test_onl = train_test_split(X_buffer, y_buffer,
                                                                        test_size=0.2,
                                                                        random_state=80085)
    
    train_dataset = create_dataset(X_train_onl, y_train_onl)
    val_dataset = create_dataset(X_test_onl, y_test_onl)

    history_onl = value_model.fit(
        train_dataset,               
        batch_size=batch_size,                  
        epochs=epochs,                       
        validation_data=val_dataset
    )

    return value_model, history_onl

def online_policy_training(policy_model, tree, epochs=20, n=6, batch_size=64):
    """
    Given a replay buffer in the form of list of state-actions and their visits and a policy model,
    trains this model using the replay buffer through given number of epochs.

    Returns the trained policy_model
    """
    savs = state_action_visits(tree)
    X_buffer, y_buffer = policy_predictors_targets(savs, n)
    X_train_onl, X_test_onl, y_train_onl, y_test_onl = train_test_split(X_buffer, y_buffer,
                                                                        test_size=0.2,
                                                                        random_state=80085)
    
    train_dataset = create_dataset(X_train_onl, y_train_onl)
    val_dataset = create_dataset(X_test_onl, y_test_onl)

    history_onl = policy_model.fit(
        train_dataset,               
        batch_size=batch_size,                  
        epochs=epochs,                       
        validation_data=val_dataset
    )

    return policy_model, history_onl

def simulate_random_games(n):
    game_boards = []
    game = OthelloBoard(n)

    while True:
        moves = game.move_generator()
        if moves == []:
            game.pass_counter += 1
            game.change_turn()

        else:
            game.pass_counter = 0 # reset counter
            action = random.choice(moves)
            board = game.make_move(action)[0]
            game_boards.append(np.array(board))
            game.change_turn()
        
        if game.pass_counter == 2:
            reward = game.find_winner()
            break

    return (game_boards, reward)

def save_games(n, nr_simulations):
    all_game_boards = []

    for _ in range(nr_simulations):
        boards, actions = simulate_random_games(n)
        all_game_boards.append(boards)

    with open(f"othello_sim_boards_{nr_simulations}", "wb") as fp:   #Pickling
        pickle.dump(all_game_boards, fp)

def build_tree(n, value_model, policy_model, c, nr_simulations=1000):
    
    game = OthelloBoard(n)
    root = MCTSNode(n, game.board, game.to_play)
    episodes = []
    mapped_actions = map_actions_to_integers(n)
    
    while True:
        terminal_state, reward = expand_tree_iteratively(root, value_model, policy_model, mapped_actions, c)
        episodes.append((episode(terminal_state)[:-1], reward))
        print('Episode done!')

        if len(episodes) == nr_simulations:
            break

    return root, episodes

def expand_tree_iteratively(root, value_model, policy_model, mapped_actions, c):
    stack = [(root, root)]  # Stack holds pairs of (root, root)

    while stack:
        current_root, current_node = stack.pop()  # Get the last node to process

        if current_node._untried_actions:  # If the node has untried actions, expand it
            next_node = current_node.expand()
            # Instead of using recursion, use a random value for the q_val
            q_val = np.array(value_model(np.expand_dims(np.array(next_node.state), axis=(0, -1)))[0][0])
            next_node.update_q(q_val)
            next_node.backpropagate(next_node.q_value)

            # Push the root back into the stack to continue processing from the root
            stack.append((current_root, current_root))

        else:  # If no untried actions, move to the best child or terminal node
            if current_node.pass_counter != 2:
                
                flatten_state = np.ndarray.flatten(np.array(current_node.state))
                policy_dist = np.array(policy_model(np.expand_dims(flatten_state, axis=0))[0])
                for child in current_node.child_nodes:
                    child.p_action = policy_dist[mapped_actions[child.parent_action]]

                best_child = current_node.best_child(c)
                stack.append((current_root, best_child))  # Continue with the best child

            else:  # Terminal node/state
                reward = current_node.find_winner(current_node.state)
                current_node.update_q(reward)
                current_node.backpropagate(reward)
                current_node.terminal_visits += 1
                return current_node, reward
            
def unpack_plot_history(history_list):
    for i, history in enumerate(history_list):
        if history != []:
            label = f"C{i}"
            plt.plot(history.history['loss'], label=f'Training loss', color = label)
            plt.plot(history.history['val_loss'], linestyle="dashed", label = 'Validation loss', color = label)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

def cycle_results_histogram(episodes_list):
    """
    Takes in an ordered (from first to last training cycle) episodes list where the length of the
    list denotes the number of training cycles that were needed to create it (each element in the
    list should be a list of episodes).
    
    In return, plots a histogram with sections colored according to the number of white wins, black
    wins, and draws (grey).

    As white should win at 6x6 othello, we expect to see the bars further to the right to be
    increasingly dominated by the white sections.
    """
    results = []
    for episodes in episodes_list:
        results.append(results_distribution(episodes))

    # Generate x-axis indices for the number of tuples in the list
    x = range(len(results))
    labels = list(range(1, len(results) + 1))

    # Unpack each tuple into separate segments
    segment1 = [t[0] for t in results]  # Black section
    segment2 = [t[1] for t in results]  # Grey section
    segment3 = [t[2] for t in results]  # White section

    plt.figure(figsize=(10, 5))

    # Plot the first segment (black)
    plt.bar(x, segment1, color='black', edgecolor='black')

    # Plot the second segment (grey), stacked on top of the first
    plt.bar(x, segment2, bottom=segment1, color='grey', edgecolor='black')

    # Compute the bottom for the third segment (segment1 + segment2)
    bottom3 = [a + b for a, b in zip(segment1, segment2)]

    # Plot the third segment (white)
    plt.bar(x, segment3, bottom=bottom3, color='white', edgecolor='black')

    # # Add a text annotation above each bar showing the white percentage.
    # for i, t in enumerate(results):
    #     total = sum(t)
    #     # Calculate the white percentage (t[2] is the white segment)
    #     white_percentage = (t[2] / total) * 100 if total else 0
    #     # Use .3g to format the number with a maximum of 3 significant digits
    #     plt.text(i, total + 1, f'{white_percentage:.3g}%', ha='center', va='bottom', fontsize=5)

    plt.xlabel('Training cycle')
    plt.ylabel('Episodes')
    plt.ylim(0, max([sum(t) for t in results]) * 1.1)  # Slightly higher than max for visual clarity
    plt.xticks(x, labels)
    plt.show()

def terminal_visits_histogram_colored(episodes):
    """
    Takes in a list of episodes and returns a histogram with as many bars as
    unique terminal states that were visited in those episodes, with each bar
    colored according to who wins (or draw) in that position. White bars for
    white wins, black bars for black wins, and grey for draws.
    """

    nr_games = len(episodes)
    win, draw, loss = results_distribution(episodes)
    w_rate, d_rate, l_rate = np.round(np.array([win, draw, loss])*(100/nr_games), decimals=0)

    terminal_states = {}
    for episode in episodes:
        key = (episode[0][0].tobytes(), episode[1])
        if key not in terminal_states:
            terminal_states[key] = 1
        else:
            terminal_states[key] += 1

    color_mapping = {
        -1: 'white',  # white for -1
        0: 'grey',   # grey for 0
        1: 'black'   # black for 1
    }

    # Extract rewards, visits, and colors
    identifiers = []
    heights = []
    colors = []
    for (identifier, y_value), height in terminal_states.items():
        if isinstance(identifier, bytes):
            identifier = identifier.decode('latin-1')
        identifiers.append(identifier)
        heights.append(height)
        colors.append(color_mapping[y_value])

    x = range(len(terminal_states))

    plt.figure(figsize=(7.2, 5))
    plt.bar(identifiers, heights, color=colors, edgecolor='black')

    # Add a text box with game results
    textstr = f"Total games = {nr_games}\nBlack wins: {w_rate}%\nDraws: {d_rate}%\nWhite wins: {l_rate}%"

    # Position the text box
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(0.025, 0.975, textstr, transform=plt.gca().transAxes, fontsize=9,
            verticalalignment='top', bbox=props)


    plt.xticks(x, [str(i + 1) for i in x])

    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.xlabel('Unique terminal states')
    plt.ylabel('Visits')

    plt.show()

def remove_empty_lists(list):
    # remove all empty lists from the list
    list = [x for x in list if x]
    return list

def unpack_plot_train_val_curves(history_list, cycle_labels):
    """
    All empty lists have to removed from the input list here in order for it to work.
    See above helper function.

    Cycle_labels should be a list of of labels for the cycles, i.e. [3,6,9,...] if
    every third cycle is passed in the history list
    """
    
    fig = plt.figure(figsize=(12, 6))

    for i, history in enumerate(history_list):
        plt.subplot(2,5,i+1)
        color = f"C{i}"

        plt.plot(history.history['loss'], label=f'Training loss', color = color)
        plt.plot(history.history['val_loss'], linestyle="dashed", label = 'Validation loss', color = color)
        plt.title(f'Cycle {cycle_labels[i]}')
        plt.tick_params(axis='x', which='both', bottom=False, 
            top=False, labelbottom=False) 
        plt.tick_params(axis='y', which='both', right=False, 
            left=False, labelleft=False) 

    textstr = '\u2500' * 5 + '  Training loss \n--------  Validation loss'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.text(-0.125, -0.10, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    fig.text(0.5, 0.04, 'Epoch', va='center', ha='center', fontsize=14)
    fig.text(0.09, 0.5, 'Loss', va='center', ha='center', rotation='vertical', fontsize=14)
    plt.show()
