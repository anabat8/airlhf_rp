from functools import partial

from imitation.algorithms import preference_comparisons
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from matplotlib import pyplot as plt
from seals import base_envs
from seals.diagnostics.cliff_world import CliffWorldEnv
from stable_baselines3 import PPO, DQN
import numpy as np
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.vec_env import DummyVecEnv


class FigureRecorderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        # Plot values (here a random variable)
        figure = plt.figure()
        figure.add_subplot().plot(np.random.random(3))
        # Close the figure after logging it
        self.logger.record("trajectory/figure", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
        plt.close()
        return True


# Construct network

rng = np.random.default_rng(0)

# Initialize environment

env_creator = partial(CliffWorldEnv, height=4, horizon=40, width=7, use_xy_obs=True)
env_single = env_creator()

state_env_creator = lambda: base_envs.ExposePOMDPStateWrapper(env_creator())

# This is just a vectorized environment
state_venv = DummyVecEnv([state_env_creator] * 4)

model = DQN.load("dqn_cliff-world", env=state_venv)

reward_net = BasicRewardNet(
    state_venv.observation_space, state_venv.action_space, normalize_input_layer=RunningNorm
)

fragmenter = preference_comparisons.RandomFragmenter(
    warning_threshold=0,
    rng=rng,
)

# Gather synthetic human preferences

gatherer = preference_comparisons.SyntheticGatherer(rng=rng)

preference_model = preference_comparisons.PreferenceModel(reward_net)
reward_trainer = preference_comparisons.BasicRewardTrainer(
    preference_model=preference_model,
    loss=preference_comparisons.CrossEntropyRewardLoss(),
    epochs=3,
    rng=rng,
)

trajectory_generator = preference_comparisons.AgentTrainer(
    algorithm=model,
    reward_fn=reward_net,
    venv=state_venv,
    exploration_frac=0.1,
    rng=rng,
)

pref_comparisons = preference_comparisons.PreferenceComparisons(
    trajectory_generator,
    reward_net,
    num_iterations=60,
    fragmenter=fragmenter,
    preference_gatherer=gatherer,
    reward_trainer=reward_trainer,
    fragment_length=10,
    transition_oversampling=5,
    initial_comparison_frac=0.1,
    allow_variable_horizon=False,
    initial_epoch_multiplier=6,
    query_schedule="hyperbolic",
)

# Train network, learn reward function through human preferences

pref_comparisons.train(
    total_timesteps=5_000,
    total_comparisons=700,
)

print("--------------------DONE LEARNING THE REWARD-----------------------")

# Wrap environment with learned reward

learned_reward_venv = RewardVecEnvWrapper(state_venv, reward_net.predict_processed)

print("PRINTING LEARNED REWARD")

# Train an agent with the learned reward

learner = DQN("MlpPolicy", learned_reward_venv, verbose=1)

learner.learn(total_timesteps=200_000, progress_bar=True)

learner.save("dqn_cliff-world-learned-reward-PC")
