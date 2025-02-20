# %%
import numpy as np
import random

# %%
# Gridworld parameters
N = 5                   # grid size (N x N)
T = 30                  # maximum time steps per episode
alpha = 0.1             # learning rate
gamma = 0.9             # discount factor
episodes = 1000         # number of episodes
epsilon = 0.1           # exploration rate

# Create grid coordinates
grid = [(r, c) for r in range(N) for c in range(N)]

# Define actions
actions = ["up", "right", "down", "left"]
# Using (row, col) convention, we define the effect of each action.
# For example, "up" decreases the row index.
actions_dict = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1)
}

# Define rewards
rewards_dict = {
    "collect_apple": 1,
    "caught_by_monster": -1,
    "empty": 0
}

# %%
def initial_positions():
    """
    Randomly choose distinct positions for the agent, monster, and apple.
    
    Returns:
        state (tuple): (agent_pos, monster_pos, apple_pos)
    """
    return random.sample(grid, 3)

def respawn_apple(agent_pos, monster_pos):
    """
    Respawn the apple in a new unoccupied cell.
    
    Parameters:
        agent_pos (tuple): current position of the agent.
        monster_pos (tuple): current position of the monster.
    
    Returns:
        (tuple): new apple position.
    """
    available_positions = [pos for pos in grid 
                            if pos != agent_pos and pos != monster_pos]
    return random.choice(available_positions)

def move(pos, action):
    """
    Returns the new position after taking an action.
    If the move would take the entity out of bounds, the entity remains in place.
    
    Parameters:
        pos (tuple): current (row, col) position.
        action (str): one of the defined actions.
    
    Returns:
        (tuple): new position.
    """
    delta = actions_dict[action]
    new_r = pos[0] + delta[0]
    new_c = pos[1] + delta[1]
    # Check grid boundaries
    if 0 <= new_r < N and 0 <= new_c < N:
        return (new_r, new_c)
    else:
        return pos

def step(state, agent_action, monster_action):
    """
    Execute one step in the gridworld with simultaneous moves.
    
    Parameters:
        state (tuple): (agent_pos, monster_pos, apple_pos)
        agent_action (str): action chosen by the agent.
        monster_action (str): action chosen by the monster.
    
    Returns:
        next_state (tuple): the updated state.
        reward (int): immediate reward obtained.
        done (bool): True if the episode terminates.
    """
    agent_pos, monster_pos, apple_pos = state
    
    # Compute new positions simultaneously.
    new_agent_pos = move(agent_pos, agent_action)
    new_monster_pos = move(monster_pos, monster_action)
    
    # Check collision: if the agent and monster land on the same cell, agent is caught.
    if new_agent_pos == new_monster_pos:
        return (new_agent_pos, new_monster_pos, apple_pos), rewards_dict["caught_by_monster"], True
    
    # Check for apple collection.
    apple_collected = False
    reward = rewards_dict["empty"]
    if new_agent_pos == apple_pos:
        reward = rewards_dict["collect_apple"]
        apple_collected = True
        
    # Respawn apple if collected.
    if apple_collected:
        new_apple_pos = respawn_apple(new_agent_pos, new_monster_pos)
    else:
        new_apple_pos = apple_pos
    
    next_state = (new_agent_pos, new_monster_pos, new_apple_pos)
    return next_state, reward, False

def epsilon_greedy(Q, state, epsilon=0.1):
    """
    Select an action using the epsilon-greedy policy.
    
    Parameters:
        Q (dict): Q-value table mapping states to dictionaries of action-values.
        state (tuple): current state.
        epsilon (float): probability of choosing a random action.
    
    Returns:
        (str): chosen action.
    """
    if random.random() < epsilon or state not in Q:
        return random.choice(actions)
    else:
        # Choose the best action; break ties randomly.
        max_value = max(Q[state].values())
        best_actions = [a for a, v in Q[state].items() if v == max_value]
        return random.choice(best_actions)

# %%
# Initialize Q-value dictionary.
# We add states on the fly to avoid pre-initializing a huge state space.
Q = {}

# Q-learning training loop.
for episode in range(episodes):
    state = initial_positions()  # (agent_pos, monster_pos, apple_pos)
    total_reward = 0
    t = 0
    done = False
    
    while t < T and not done:
        # Ensure the current state is in Q.
        if state not in Q:
            Q[state] = {a: 0.0 for a in actions}
        
        # Choose actions: agent via epsilon-greedy, monster randomly.
        agent_action = epsilon_greedy(Q, state, epsilon)
        monster_action = random.choice(actions)
        
        # Take a simultaneous step in the environment.
        next_state, reward, done = step(state, agent_action, monster_action)
        
        # Ensure the next state is in Q (if not terminal).
        if not done and next_state not in Q:
            Q[next_state] = {a: 0.0 for a in actions}
        
        # Q-learning update.
        best_next = max(Q[next_state].values()) if not done else 0
        Q[state][agent_action] += alpha * (reward + gamma * best_next - Q[state][agent_action])
        
        state = next_state
        total_reward += reward
        t += 1
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}: Total Reward = {total_reward}")