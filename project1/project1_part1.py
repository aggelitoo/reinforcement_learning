# %%
import numpy as np


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
    """
    Performs policy evaluation until value function converges.
    """
    while True:
        delta = 0
        for s in states:
            v = V[s]
            a = policy[s]
            V[s] = sum(P[s][a][s_next] * (r[s][a] + gamma * V[s_next]) for s_next in states)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

def policy_improvement(policy, V, P, r, gamma):
    """
    Performs policy improvement by selecting greedy actions.
    """
    policy_stable = True
    for s in states:
        old_action = policy[s]
        action_values = {a: sum(P[s][a][s_next] * (r[s][a] + gamma * V[s_next]) for s_next in states) for a in actions}
        best_action = max(action_values, key=action_values.get)
        policy[s] = best_action
        if old_action != best_action:
            policy_stable = False
    return policy_stable

def policy_iteration():
    """
    Runs the policy iteration algorithm.
    See course book page 80 for algorithm that we have used in pseudo code.
    """
    while True:
        policy_evaluation(policy, V, P, r, theta, gamma)
        if policy_improvement(policy, V, P, r, gamma):
            break  # Stop if policy is stable
    return policy, V

def value_iteration():
    """Runs the value iteration algorithm."""
    V = {s: 0.0 for s in states}  # Reset value function
    while True:
        delta = 0
        for s in states:
            v = V[s]
            V[s] = max(sum(P[s][a][s_next] * (r[s][a] + gamma * V[s_next]) for s_next in states) for a in actions)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    policy = {s: max(actions, key=lambda a: sum(P[s][a][s_next] * (r[s][a] + gamma * V[s_next]) for s_next in states)) for s in states}
    return policy, V

# %%
# Run policy iteration
optimal_policy, optimal_values = policy_iteration()
print("Optimal Policy (Policy iteration):", optimal_policy)
print("Optimal State Values (Policy iteration):", optimal_values)

# Run value iteration
optimal_policy_vi, optimal_values_vi = value_iteration()
print("\nOptimal Policy (Value Iteration):", optimal_policy_vi)
print("Optimal State Values (Value Iteration):", optimal_values_vi)

# %%
