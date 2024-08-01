import numpy as np
import math
import random
import matplotlib.pyplot as plt

class Pendulum:
    def __init__(self, theta=math.pi, theta_dot=0.0):
        self.theta = theta  # Initial angle (downward position)
        self.theta_dot = theta_dot  # Initial angular velocity
        self.g = 9.8  # Gravity
        self.length = 1.0  # Length of the pendulum
        self.dt = 0.1  # Time step
        self.I = 1.0  # Moment of inertia

    def step(self, u):
        # Pendulum dynamics
        theta_acc = (u - self.g * math.sin(self.theta)) / self.I
        self.theta_dot += theta_acc * self.dt
        self.theta += self.theta_dot * self.dt
        return np.array([self.theta, self.theta_dot])

    def reset(self):
        self.theta = math.pi
        self.theta_dot = 0.0
        return np.array([self.theta, self.theta_dot])

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_cost = 0.0

    def is_leaf(self):
        return len(self.children) == 0

    def add_child(self, child_node):
        self.children.append(child_node)

    def best_child(self, exploration_weight=1.4):
        # UCT formula to balance exploration and exploitation
        weights = []
        for child in self.children:
            if child.visits == 0:
                weights.append(float('inf'))  # Infinite value to ensure exploration
            else:
                weights.append(
                    (child.total_cost / child.visits) +
                    exploration_weight * np.sqrt(np.log(self.visits) / child.visits)
                )
        return self.children[np.argmin(weights)]

def rollout(pendulum, max_steps=10):
    state = pendulum.state.copy()
    total_cost = 0.0
    for _ in range(max_steps):
        u = random.choice([-2.0, 0.0, 2.0])  # Random policy
        next_state = pendulum.step(u)
        cost = abs(next_state[0])  # Cost: distance from upright position
        total_cost += cost
        if abs(next_state[0]) < 0.1 and abs(next_state[1]) < 0.1:  # Terminal condition
            break
    return total_cost

def backpropagate(node, cost):
    while node:
        node.visits += 1
        node.total_cost += cost
        node = node.parent

def select_action(root_node, pendulum, max_simulations=500):
    for _ in range(max_simulations):
        node = root_node
        pendulum.state = node.state.copy()

        # Selection
        while not node.is_leaf():
            node = node.best_child()

        # Expansion
        if node.visits > 0:
            for action in [-2.0, 0.0, 2.0]:
                next_state = pendulum.step(action)
                child_node = MCTSNode(next_state, parent=node, action=action)
                node.add_child(child_node)

            if node.children:
                node = random.choice(node.children)

        # Simulation
        cost = rollout(pendulum)
        # Backpropagation
        backpropagate(node, cost)

    # Select the action with the lowest average cost
    best_child = root_node.best_child(exploration_weight=0.0)
    return best_child.action, best_child

def main():
    pendulum = Pendulum()
    root_node = MCTSNode(state=pendulum.reset())

    # Run MCTS to determine the best action
    best_action, best_child = select_action(root_node, pendulum)
    print(f"Best action: {best_action}")

    # Estimate the distribution of control inputs
    action_distribution = {}
    for child in root_node.children:
        action_distribution[child.action] = child.visits / root_node.visits

    # Plot the distribution of control inputs
    actions = list(action_distribution.keys())
    probabilities = list(action_distribution.values())
    plt.bar(actions, probabilities, width=0.4)
    plt.xlabel('Control Input')
    plt.ylabel('Probability')
    plt.title('Estimated Distribution of Control Inputs')
    plt.show()

if __name__ == "__main__":
    main()
