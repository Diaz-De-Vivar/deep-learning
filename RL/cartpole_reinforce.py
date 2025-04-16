# python cartpole_reinforce.py
#!/usr/bin/env python3
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple policy network
class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

def discount_rewards(rewards, gamma=0.99):
    """
    Compute the discounted reward for each time step.
    Args:
        rewards (list of float): rewards collected during one episode.
        gamma (float): discount factor.
    Returns:
        list of float: discounted rewards.
    """
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    return discounted_rewards

def main():
    # Create the CartPole-v1 environment
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]   # For CartPole, this is 4.
    action_size = env.action_space.n                # For CartPole, this is 2.

    # Instantiate our policy network and the optimizer
    policy = Policy(state_size, action_size)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    gamma = 0.99  # Discount factor

    num_episodes = 500
    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        total_reward = 0
        done = False

        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state)
            # Get action probabilities from the policy network
            action_probs = policy(state_tensor)
            distribution = torch.distributions.Categorical(action_probs)
            action = distribution.sample()  # Sample an action according to the policy
            log_prob = distribution.log_prob(action)

            # Take the action in the environment
            next_state, reward, done, _ = env.step(action.item())

            # Record log probability and reward
            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward

            state = next_state

        # Compute discounted rewards
        discounted_rewards = discount_rewards(rewards, gamma)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        # Normalize discounted rewards to improve convergence
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        # Compute the loss and perform a gradient update
        loss = 0
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            loss += -log_prob * Gt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}\tTotal Reward: {total_reward}")

    env.close()

if __name__ == '__main__':
    main()