from imitation.util.util import make_vec_env
from stable_baselines3 import PPO, DQN
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

# Evaluate learner using the original reward

rng = np.random.default_rng(0)
venv = make_vec_env("FrozenLake-v1", n_envs=4, rng=rng)

learner = DQN.load("dqn_frozen-lake-learned-reward-PC", env=venv)

n_eval_episodes = 1000
reward_mean, reward_std = evaluate_policy(learner.policy, venv, n_eval_episodes)
reward_stderr = reward_std / np.sqrt(n_eval_episodes)
print(f"Reward: {reward_mean:.0f} +/- {reward_stderr:.0f}")
