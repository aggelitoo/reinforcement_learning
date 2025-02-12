# %%
import numpy as np
from matplotlib import pyplot as plt

# %%
# MDP components
states = ["pristine", "worn", "broken"]
actions = ["continue", "repair"]
gamma = 0.9  # Discount factor
theta = 1e-6  # tolerance

# Transition probabilities p(s' | s, a)
P = {
    "pristine": {"continue": {"pristine": 0.8, "worn": 0.2, "broken": 0},
                 "repair": {"pristine": 1, "worn": 0, "broken": 0}},

    "worn": {"continue": {"pristine": 0, "worn": 0.8, "broken": 0.2},
             "repair": {"pristine": 1, "worn": 0, "broken": 0}},

    "broken": {"continue": {"pristine": 1, "worn": 0, "broken": 0},
               "repair": {"pristine": 1, "worn": 0, "broken": 0}} 
}

# Rewards r(s, a)
r = {
    "pristine": {"continue": 4, "repair": -5},
    "worn": {"continue": 2, "repair": -5},
    "broken": {"continue": -10, "repair": -10}
}

# Policy and value function initializations (arbitrary)
policy = {s: "continue" for s in states}
V = {s: 0.0 for s in states}


# %%
def policy_evaluation(policy, V, P, r, theta, gamma):
    """Performs policy evaluation until value function converges."""
    iterations = 0
    history = []
    while True:
        delta = 0
        for s in states:
            v = V[s]  # Store old value
            a = policy[s]  # Policy dictates action
            V[s] = sum(P[s][a][s_next] * (r[s][a] + gamma * V[s_next]) for s_next in states)
            delta = max(delta, abs(v - V[s]))
        history.append(sum(V.values()))  # Track value function sum over time
        iterations += 1
        if delta < theta:
            break
    return iterations, history

def policy_improvement(policy, V, P, r, gamma):
    """Performs policy improvement by selecting greedy actions."""
    policy_stable = True
    for s in states:
        old_action = policy[s]
        action_values = {a: sum(P[s][a][s_next] * (r[s][a] + gamma * V[s_next]) for s_next in states) for a in actions}
        best_action = max(action_values, key=action_values.get)
        policy[s] = best_action
        if old_action != best_action:
            policy_stable = False
    return policy_stable

def policy_iteration(P, r, gamma):
    """Runs the policy iteration algorithm."""
    policy = {s: "continue" for s in states}  # Initial policy
    V = {s: 0.0 for s in states}  # Initialize state-value function
    iteration_count = 0
    history = []
    while True:
        iteration_count += 1
        eval_iterations, eval_history = policy_evaluation(policy, V, P, r, theta, gamma)
        history.extend(eval_history)
        if policy_improvement(policy, V, P, r, gamma):
            break  # Stop if policy is stable
    return policy, V, iteration_count, history

def value_iteration(P, r, gamma):
    """Runs the value iteration algorithm."""
    V = {s: 0.0 for s in states}  # Reset value function
    iteration_count = 0
    history = []
    while True:
        delta = 0
        for s in states:
            v = V[s]
            V[s] = max(sum(P[s][a][s_next] * (r[s][a] + gamma * V[s_next]) for s_next in states) for a in actions)
            delta = max(delta, abs(v - V[s]))
        history.append(sum(V.values()))  # Track value function sum over time
        iteration_count += 1
        if delta < theta:
            break
    policy = {s: max(actions, key=lambda a: sum(P[s][a][s_next] * (r[s][a] + gamma * V[s_next]) for s_next in states)) for s in states}
    return policy, V, iteration_count, history


# %%
# Run policy iteration
optimal_policy, optimal_values, iteration_count, history_pi = policy_iteration(P, r, gamma)
print("Optimal Policy (Policy iteration):", optimal_policy)
print("Optimal State Values (Policy iteration):", optimal_values)

# Run value iteration
optimal_policy_vi, optimal_values_vi, iteration_count_vi, history_vi = value_iteration(P, r, gamma)
print("\nOptimal Policy (Value Iteration):", optimal_policy_vi)
print("Optimal State Values (Value Iteration):", optimal_values_vi)

# %%
gamma_values = [0.7, 0.8, 0.9]
fig, axes = plt.subplots(2, 1, figsize=(8, 10))

# Policy Iteration Plot
for gamma in gamma_values:
    _, _, _, history_pi = policy_iteration(P, r, gamma)
    axes[0].plot(history_pi, label=f'Policy Iteration (γ={gamma})')
axes[0].set_ylabel("Sum of State Values")
axes[0].legend(loc = 'lower right')
axes[0].grid()

# Value Iteration Plot
for gamma in gamma_values:
    _, _, _, history_vi = value_iteration(P, r, gamma)
    axes[1].plot(history_vi, label=f'Value Iteration (γ={gamma})')
axes[1].set_xlabel("Iterations")
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()

# %%
