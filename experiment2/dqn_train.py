import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env("FrozenLake-v1", n_envs=4)

model = DQN("MlpPolicy", vec_env, verbose=1,
            gamma=0.85,
            batch_size=32
            )
model.learn(total_timesteps=350_000, log_interval=4)
model.save("dqn_frozen-lake")
