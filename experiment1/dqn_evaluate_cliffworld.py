from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN
from functools import partial

from seals import base_envs
from seals.diagnostics.cliff_world import CliffWorldEnv
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

env_creator = partial(CliffWorldEnv, height=4, horizon=40, width=7, use_xy_obs=True)
env_single = env_creator()

state_env_creator = lambda: base_envs.ExposePOMDPStateWrapper(env_creator())

# This is just a vectorized environment
state_venv = DummyVecEnv([state_env_creator] * 4)

model = DQN.load("dqn_cliff-world", env=state_venv)

expert_reward, _ = evaluate_policy(model, state_venv)
print(f"Expert reward: {expert_reward:.2f}")
