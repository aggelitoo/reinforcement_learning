# %%
import numpy as np
import random
from matplotlib import pyplot as plt
import pandas as pd

# Gridworld parameters
N = 5                   # grid size (N x N)
T = 30                  # maximum time steps per episode
alpha = 0.1             # learning rate
gamma = 0.9             # discount factor
episodes = 10**5        # number of episodes per method

# Create grid coordinates
grid = [(r, c) for r in range(N) for c in range(N)]

# Define actions and their effects (using (row, col) convention)
actions = ["up", "right", "down", "left"]
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

# Parameters for epsilon decay
epsilon_0 = 1.0   # Initial exploration rate
epsilon_min = 0.01 # Minimum exploration
decay_rate = 0.0001 # Controls speed of decay

def get_epsilon(episode):
    """ Exponentially decaying epsilon """
    return epsilon_min + (epsilon_0 - epsilon_min) * np.exp(-decay_rate * episode)

def initial_positions():
    """Return distinct starting positions for agent, monster, and apple."""
    return tuple(random.sample(grid, 3))

def respawn_apple(agent_pos, monster_pos):
    """Return a new apple position that is not occupied by agent or monster."""
    available_positions = [pos for pos in grid if pos != agent_pos and pos != monster_pos]
    return random.choice(available_positions)

def move(pos, action):
    """
    Given a position and an action, return the new position.
    If the move is out-of-bounds, return the original position.
    """
    delta = actions_dict[action]
    new_r = pos[0] + delta[0]
    new_c = pos[1] + delta[1]
    if 0 <= new_r < N and 0 <= new_c < N:
        return (new_r, new_c)
    else:
        return pos

def step(state, agent_action, monster_action):
    """
    Execute one simultaneous step for agent and monster.
    Returns next_state, reward, and a done flag.
    """
    agent_pos, monster_pos, apple_pos = state
    new_agent_pos = move(agent_pos, agent_action)
    new_monster_pos = move(monster_pos, monster_action)
    
    # Check if agent and monster collide.
    if new_agent_pos == new_monster_pos:
        return (new_agent_pos, new_monster_pos, apple_pos), rewards_dict["caught_by_monster"], True
    
    # Check for apple collection.
    reward = rewards_dict["empty"]
    if new_agent_pos == apple_pos:
        reward = rewards_dict["collect_apple"]
        new_apple_pos = respawn_apple(new_agent_pos, new_monster_pos)
    else:
        new_apple_pos = apple_pos

    next_state = (new_agent_pos, new_monster_pos, new_apple_pos)
    return next_state, reward, False

def epsilon_greedy(Q, state, epsilon):
    """
    Return an action chosen by the epsilon-greedy policy based on Q.
    """
    if state not in Q or random.random() < epsilon:
        return random.choice(actions)
    max_val = max(Q[state].values())
    best_actions = [a for a, v in Q[state].items() if v == max_val]
    return random.choice(best_actions)

def epsilon_greedy_double(Q1, Q2, state, epsilon):
    """
    For double Q-learning: choose an action using the sum of Q1 and Q2 values.
    """
    if state not in Q1 or state not in Q2 or random.random() < epsilon:
        return random.choice(actions)
    combined = {a: Q1[state][a] + Q2[state][a] for a in actions}
    max_val = max(combined.values())
    best_actions = [a for a, v in combined.items() if v == max_val]
    return random.choice(best_actions)

# %%
# ---------------------------
# Off-Policy Q-Learning
# ---------------------------
Q_learning = {}
Q_learning_rewards = []

print("Starting Off-Policy Q-Learning...")
for episode in range(episodes):
    state = initial_positions()  # (agent_pos, monster_pos, apple_pos)
    if state not in Q_learning:
        Q_learning[state] = {a: 0.0 for a in actions}
    
    epsilon = get_epsilon(episode)
    total_reward = 0
    t = 0
    done = False
    while t < T and not done:
        # Select action using Q_learning's own epsilon-greedy.
        agent_action = epsilon_greedy(Q_learning, state, epsilon)
        monster_action = random.choice(actions)
        
        next_state, reward, done = step(state, agent_action, monster_action)
        total_reward += reward
        
        if not done:
            if next_state not in Q_learning:
                Q_learning[next_state] = {a: 0.0 for a in actions}
            best_next = max(Q_learning[next_state].values())
        else:
            best_next = 0
        
        Q_learning[state][agent_action] += alpha * (reward + gamma * best_next - Q_learning[state][agent_action])
        state = next_state
        t += 1
    
    # save accumulated episode reward
    Q_learning_rewards.append(total_reward)

    if (episode + 1) % 100 == 0:
        print(f"Off-Policy Q-Learning Episode {episode+1}: Total Reward = {total_reward}")

# %%
# ---------------------------
# 1-Step SARSA
# ---------------------------
Q_sarsa = {}
Q_sarsa_rewards = []

print("\nStarting 1-Step SARSA...")
for episode in range(episodes):
    state = initial_positions()
    if state not in Q_sarsa:
        Q_sarsa[state] = {a: 0.0 for a in actions}
    # Choose initial action using Q_sarsa.
    agent_action = epsilon_greedy(Q_sarsa, state, epsilon)
    
    epsilon = get_epsilon(episode)
    total_reward = 0
    t = 0
    done = False
    while t < T and not done:
        if state not in Q_sarsa:
            Q_sarsa[state] = {a: 0.0 for a in actions}
        monster_action = random.choice(actions)
        next_state, reward, done = step(state, agent_action, monster_action)
        total_reward += reward
        
        if not done:
            if next_state not in Q_sarsa:
                Q_sarsa[next_state] = {a: 0.0 for a in actions}
            next_action = epsilon_greedy(Q_sarsa, next_state, epsilon)
            Q_sarsa[state][agent_action] += alpha * (reward + gamma * Q_sarsa[next_state][next_action] - Q_sarsa[state][agent_action])
        else:
            Q_sarsa[state][agent_action] += alpha * (reward - Q_sarsa[state][agent_action])
        
        state = next_state
        if not done:
            agent_action = next_action
        t += 1
    
    Q_sarsa_rewards.append(total_reward)
    if (episode + 1) % 100 == 0:
        print(f"SARSA Episode {episode+1}: Total Reward = {total_reward}")

# ---------------------------
# Double Q-Learning
# ---------------------------
# In double Q-learning, the behavior policy is derived from the sum of Q_double1 and Q_double2.
Q_double1 = {}
Q_double2 = {}
Q_double_rewards = []

print("\nStarting Double Q-Learning...")
for episode in range(episodes):
    state = initial_positions()
    for Q in (Q_double1, Q_double2):
        if state not in Q:
            Q[state] = {a: 0.0 for a in actions}
    
    epsilon = get_epsilon(episode)
    total_reward = 0
    t = 0
    done = False
    # Choose initial action using the double Q behavior policy.
    agent_action = epsilon_greedy_double(Q_double1, Q_double2, state, epsilon)
    
    while t < T and not done:
        for Q in (Q_double1, Q_double2):
            if state not in Q:
                Q[state] = {a: 0.0 for a in actions}
        monster_action = random.choice(actions)
        next_state, reward, done = step(state, agent_action, monster_action)
        total_reward += reward
        
        for Q in (Q_double1, Q_double2):
            if next_state not in Q:
                Q[next_state] = {a: 0.0 for a in actions}
        
        if not done:
            next_action = epsilon_greedy_double(Q_double1, Q_double2, next_state, epsilon)
            # Randomly update one of the two Q-tables.
            if random.random() < 0.5:
                best_action = max(Q_double1[next_state], key=Q_double1[next_state].get)
                target = reward + gamma * Q_double2[next_state][best_action]
                Q_double1[state][agent_action] += alpha * (target - Q_double1[state][agent_action])
            else:
                best_action = max(Q_double2[next_state], key=Q_double2[next_state].get)
                target = reward + gamma * Q_double1[next_state][best_action]
                Q_double2[state][agent_action] += alpha * (target - Q_double2[state][agent_action])
        else:
            # Terminal state update.
            if random.random() < 0.5:
                Q_double1[state][agent_action] += alpha * (reward - Q_double1[state][agent_action])
            else:
                Q_double2[state][agent_action] += alpha * (reward - Q_double2[state][agent_action])
        
        state = next_state
        if not done:
            agent_action = next_action
        t += 1
    
    Q_double_rewards.append(total_reward)
    if (episode + 1) % 100 == 0:
        print(f"Double Q-Learning Episode {episode+1}: Total Reward = {total_reward}")

# At the end, Q_learning, Q_sarsa, and (Q_double1, Q_double2) hold the learned action values for each method.

# %%
# ---------------------------
# n-step SARSA Function
# ---------------------------
def n_step_sarsa(n, episodes):
    """
    Runs n-step SARSA on the gridworld for a given number of episodes.
    No pre-initialization of all states is needed; states are added as encountered.
    
    Parameters:
      n        : number of steps for bootstrapping
      episodes : number of episodes to run
      
    Returns:
      Q_n      : learned Q-value table (dictionary)
      rewards_n: list of total rewards per episode
    """
    Q_n = {}
    rewards_n = []
    for episode in range(episodes):
        epsilon_val = get_epsilon(episode)
        # Initialize starting state and Q-values for that state if needed.
        state = initial_positions()
        if state not in Q_n:
            Q_n[state] = {a: 0.0 for a in actions}
        # Choose initial action using the epsilon-greedy policy.
        action = epsilon_greedy(Q_n, state, epsilon_val)
        
        # Lists to store the trajectory:
        states = [state]           # states[0] is the initial state
        actions_list = [action]    # actions_list[0] is the initial action
        rewards_list = [0]         # rewards_list[0] is a dummy reward
        
        T_episode = 30
        t = 0
        total_reward = 0
        
        while True:
            if t < T_episode:
                monster_action = random.choice(actions)
                next_state, reward, done = step(state, action, monster_action)
                total_reward += reward
                rewards_list.append(reward)
                states.append(next_state)
                if done:
                    T_episode = t + 1
                else:
                    if next_state not in Q_n:
                        Q_n[next_state] = {a: 0.0 for a in actions}
                    next_action = epsilon_greedy(Q_n, next_state, epsilon_val)
                    actions_list.append(next_action)
            tau = t - n + 1
            if tau >= 0:
                # Determine the upper index for the reward sum.
                limit = min(tau + n, T_episode)
                G = 0.0
                for i in range(tau + 1, limit + 1):
                    G += (gamma ** (i - tau - 1)) * rewards_list[i]
                if tau + n < T_episode:
                    G += (gamma ** n) * Q_n[states[tau + n]][actions_list[tau + n]]
                Q_n[states[tau]][actions_list[tau]] += alpha * (G - Q_n[states[tau]][actions_list[tau]])
            t += 1
            if tau == T_episode - 1:
                break
            # Update state and action only if available.
            if t < len(states):
                state = states[t]
            if t < len(actions_list):
                action = actions_list[t]
        rewards_n.append(total_reward)
        if (episode + 1) % 1000 == 0:
            print(f"n-step SARSA (n={n}) Episode {episode+1}: Total Reward = {total_reward}")
    return Q_n, rewards_n

# %% [code]
# ---------------------------
# Run n-step SARSA for different n values
# ---------------------------
# print("\nStarting n-step SARSA (n=2)...")
_, rewards_n2 = n_step_sarsa(2, episodes)

# print("\nStarting n-step SARSA (n=4)...")
_, rewards_n4 = n_step_sarsa(4, episodes)

# For comparison, n=1-step SARSA is equivalent to standard 1-step SARSA:
print("\nStarting n-step SARSA (n=1) [equivalent to 1-step SARSA]...")
_, rewards_n1 = n_step_sarsa(1, episodes)


# %%
# ---------------------------
# Plotting the Learning Curves
# ---------------------------
avg_len = 1000

# creating lists of average over 1000 last episodes
Q_sarsa_avg = pd.Series(Q_sarsa_rewards).rolling(window=avg_len, min_periods=1).mean().to_numpy()
Q_learning_avg = pd.Series(Q_learning_rewards).rolling(window=avg_len, min_periods=1).mean().to_numpy()
Q_dbl_avg = pd.Series(Q_double_rewards).rolling(window=avg_len, min_periods=1).mean().to_numpy()

# Compute rolling averages over the last 1000 episodes
n1_avg = pd.Series(rewards_n1).rolling(window=avg_len, min_periods=1).mean().to_numpy()
n2_avg = pd.Series(rewards_n2).rolling(window=avg_len, min_periods=1).mean().to_numpy()
n4_avg = pd.Series(rewards_n4).rolling(window=avg_len, min_periods=1).mean().to_numpy()

episodes_range = list(range(1, episodes + 1))
plt.figure(figsize=(10, 6))
plt.plot(episodes_range, Q_learning_avg, label="Off-Policy Q-Learning")
# plt.plot(episodes_range, Q_sarsa_avg, label="1-Step SARSA")
plt.plot(episodes_range, Q_dbl_avg, label="Double Q-Learning")
plt.plot(episodes_range, n1_avg, label="n-step SARSA (n=1)")
plt.plot(episodes_range, n2_avg, label="n-step SARSA (n=2)")
plt.plot(episodes_range, n4_avg, label="n-step SARSA (n=4)")
plt.xlabel("Episode")
plt.ylabel("Rolling average reward (1000 previous episodes)")
plt.legend()
plt.grid(True)
plt.show()
# %%
