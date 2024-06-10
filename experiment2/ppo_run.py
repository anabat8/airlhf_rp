from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import cv2

vec_env = make_vec_env("FrozenLake-v1", n_envs=4)
model = PPO.load("frozen-lake", env=vec_env)
obs = vec_env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
    cv2.waitKey(300)
    if np.any(dones):
        if np.any(rewards == 1):
            print(np.where(dones == 1))
            cv2.waitKey(2000)
        else:
            print("lost")
