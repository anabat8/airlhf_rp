import pathlib

import numpy as np
import torch as th

from stable_baselines3.ppo import MlpPolicy

from experiment3.RLHFAgent import RLHFAgent
from experiment3.Utils import Utils
from experiment3.Environment import Environment

# Initialize environment
SEED = 0
env = Environment("seals:seals/CartPole-v0", SEED)
venv = env.init_vec_env()

# Load trained AIRLAgent reward function
reward_net_irl = th.load("airl_agent/airl_reward_net.pt")

# Initialize RLHFAgent

rlhfAgent = RLHFAgent(env_object=env)
rlhfAgent.set_reward_from_airl(reward_net_irl)
rlhfAgent.init_gen_algo(policy_name="PPO", ac_policy=MlpPolicy)
rlhfAgent.init_trajectory_gen(env_object=env)

# Train RLHFAgent with reward initialized by AIRL

rlhfAgent.train(save_path=pathlib.Path("irlhf_agent"))

print("Done training agent.")

# Evaluate

reward_net = th.load("irlhf_agent/reward_net.pt")
Utils.train_with_learned_reward(learned_reward=reward_net,
                                save_path="irlhf_agent/irlhf_agent_trained_with_learned_reward",
                                venv=venv)
Utils.evaluate_trained_agent_with_learned_reward(load_path="irlhf_agent/irlhf_agent_trained_with_learned_reward",
                                                 venv=venv)
