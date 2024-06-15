from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env("FrozenLake-v1", n_envs=4)
model = PPO.load("frozen-lake", env=vec_env)

expert_reward, _ = evaluate_policy(model, vec_env)
print(f"Expert reward: {expert_reward:.2f}")
