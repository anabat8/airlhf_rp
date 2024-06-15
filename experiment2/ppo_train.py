import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def vectorize_env(env, n_envs):
    return gym.vector.SyncVectorEnv([lambda: env] * n_envs)


vec_env = make_vec_env("FrozenLake-v1", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0003)
model.learn(total_timesteps=280_000, progress_bar=True)
model.save("frozen-lake")
