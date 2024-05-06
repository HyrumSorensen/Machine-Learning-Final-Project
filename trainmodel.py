import gymnasium as gym
import cv2
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

BATCH_SIZE = 32
GAMMA = 0.7

def train_model(memory, model, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert lists of numpy arrays to numpy arrays before conversion to tensors for better performance
    states = torch.tensor(np.array(states), dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    dones = torch.tensor(dones, dtype=torch.float32)

    current_q = model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    next_q = model(next_states).max(1)[0]
    target_q = rewards + GAMMA * next_q * (1 - dones)
    loss = F.mse_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            # First layer with larger receptive field
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Second layer with medium receptive field
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Third layer to capture more complex features
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            # Fully connected layer to output the action values
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        """Utilizes a dummy input to pass through the conv layers to calculate the flat output size."""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """Defines the forward pass of the neural network."""
        conv_out = self.conv(x).view(x.size()[0], -1)  # Flatten the output of conv layers
        return self.fc(conv_out)


def preprocess_observation(obs):
    if not isinstance(obs, np.ndarray) or obs.ndim != 2:
        raise ValueError("Expected observation to be a 2D numpy array")
    resized = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    return normalized

def main():
    env = gym.make("ALE/SpaceInvaders-v5", obs_type="grayscale", frameskip=5, repeat_action_probability=0.25)
    input_shape = (1, 84, 84)
    n_actions = env.action_space.n
    model = DQN(input_shape, n_actions)
    optimizer = optim.Adam(model.parameters(), lr=0.00025)

    memory = deque(maxlen=50000)
    num_steps = 1000000
    save_interval = 20000
    update_interval = 500
    epsilon = 1.0
    epsilon_decay = 0.9999
    min_epsilon = 0.01

    observation, info = env.reset()
    observation = preprocess_observation(observation)
    observation = np.expand_dims(observation, axis=0)
    last_lives = info['lives']  # Initialize last_lives with the starting lives
    game_over_penalty = -35  # Define a significant penalty for losing all lives

    for step in range(num_steps):
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            observation_array = np.array([observation], dtype=np.float32)  # Convert list of numpy arrays to a single numpy array
            observation_tensor = torch.tensor(observation_array)  # Then convert to tensor
            action = model(observation_tensor).max(1)[1].item()

        next_observation, reward, done, _, info = env.step(action)

        # Check for life loss or life reset indicating the loss of the last life
        current_lives = info['lives']
        if current_lives < last_lives or (last_lives == 1 and current_lives == 3):
            reward += game_over_penalty  # Apply a significant penalty

        next_observation = preprocess_observation(next_observation)
        next_observation = np.expand_dims(next_observation, axis=0)
        memory.append((observation, action, reward, next_observation, done))
        observation = next_observation
        train_model(memory, model, optimizer)
        last_lives = current_lives  # Update last_lives to the current lives count

        if done:
            observation, info = env.reset()
            observation = preprocess_observation(observation)
            observation = np.expand_dims(observation, axis=0)
            last_lives = info['lives']  # Reset last_lives on a new episode

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if step % save_interval == 0:
            torch.save(model.state_dict(), f"space_invaders_model_step_{step}.pth")
            print(f"Model saved at step {step}")
        if step % update_interval == 0:
            print(f"At step: {step}")

    env.close()

if __name__ == "__main__":
    main()




