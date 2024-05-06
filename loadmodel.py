import gymnasium as gym
import numpy as np
import torch
import argparse
import logging
from collections import deque
from trainmodel import DQN, preprocess_observation

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_path, input_shape, n_actions):
    model = DQN(input_shape, n_actions)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

def test_model(env, model, num_episodes=5):
    scores = []
    steps = []
    for _ in range(num_episodes):
        observation, info = env.reset()
        observation = preprocess_observation(observation)
        observation = np.expand_dims(observation, axis=0)  # Add batch dimension
        total_reward = 0
        step_count = 0

        done = False
        while not done:
            obs_tensor = torch.tensor([observation], dtype=torch.float32)
            with torch.no_grad():
                q_values = model(obs_tensor)
            action = q_values.max(1)[1].item()

            observation, reward, done, _, info = env.step(action)
            observation = preprocess_observation(observation)
            observation = np.expand_dims(observation, axis=0)

            total_reward += reward
            step_count += 1

        scores.append(total_reward)
        steps.append(step_count)
        logging.info(f'Episode finished with score: {total_reward}, steps: {step_count}')

    return scores, steps

def main(model_path, num_episodes):
    setup_logging()
    env = gym.make("ALE/SpaceInvaders-v5", obs_type="grayscale", render_mode="human", frameskip=5, repeat_action_probability=0.25)
    input_shape = (1, 84, 84)
    n_actions = env.action_space.n

    model = load_model(model_path, input_shape, n_actions)
    test_model(env, model, num_episodes)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a DQN model on Space Invaders.")
    parser.add_argument('--model-file', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to test the model.')
    args = parser.parse_args()
    
    main(args.model_file, args.episodes)
