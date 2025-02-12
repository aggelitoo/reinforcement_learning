# %%
import numpy as np
from collections import defaultdict

# %%
# Define the gridworld environment 
class GridWorld:
    def __init__(self, size=(4, 4), goal=(3, 3)):
        self.size = size
        self.goal = goal
        self.start = (0, 0)
        self.state = self.start
        self.actions = ['left', 'up', 'right', 'down']
        self.action_map = {'left': (0, -1), 'up': (-1, 0), 'right': (0, 1), 'down': (1, 0)}
    
    def step(self, action):
        """ Take an action and return the next state and reward. """
        move = self.action_map[action]
        next_state = (self.state[0] + move[0], self.state[1] + move[1])
        
        # Check if out of bounds (hitting a wall)
        if not (0 <= next_state[0] < self.size[0] and 0 <= next_state[1] < self.size[1]):
            return self.state, -100  # Wall penalty but stay in the same state
        
        self.state = next_state
        
        # Check if goal reached
        if self.state == self.goal:
            return self.state, 10
        
        return self.state, -1  # Default step penalty
    
    def reset(self):
        """ Reset the environment to a random state to ensure exploring starts. """
        self.state = (np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1]))
        return self.state

# Monte Carlo Control with Exploring Starts
def mc_control(env, gamma=0.9, epsilon=0.1, episodes=1000, every_visit=True):
    Q = defaultdict(lambda: {a: 0 for a in env.actions})
    returns = defaultdict(list)
    
    for _ in range(episodes):
        episode = []
        state = env.reset()
        
        while True:
            if np.random.rand() < epsilon:
                action = np.random.choice(env.actions)
            else:
                action = max(Q[state], key=Q[state].get, default=np.random.choice(env.actions))
            
            next_state, reward = env.step(action)
            episode.append((state, action, reward))
            
            if next_state == env.goal:
                break  # End episode only on goal
            
            state = next_state
        
        G = 0
        visited = set()
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            if every_visit or (state, action) not in visited:
                returns[(state, action)].append(G)
                Q[state][action] = np.mean(returns[(state, action)])
                visited.add((state, action))
    
    policy = {s: max(Q[s], key=Q[s].get, default=np.random.choice(env.actions)) for s in Q}
    return policy, Q

# %%
env = GridWorld()
optimal_policy, optimal_value = mc_control(env, epsilon=0.05)
print("Optimal Policy:", optimal_policy)

# %%
