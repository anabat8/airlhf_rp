import cv2
import numpy as np
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

obs = state_venv.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = state_venv.step(action)
    state_venv.render("human")
    cv2.waitKey(300)
    if np.any(dones):
        if np.any(rewards == 10):
            print(np.where(dones == 1))
            cv2.waitKey(2000)
        else:
            print("lost")