from functools import partial

from imitation.algorithms import preference_comparisons
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.util.networks import RunningNorm
from seals import base_envs
from seals.diagnostics.cliff_world import CliffWorldEnv
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np
import matplotlib.pyplot as plt
import torch as th

from imitation.algorithms.mce_irl import (
    MCEIRL,
    mce_occupancy_measures,
    mce_partition_fh,
    TabularPolicy,
)
from imitation.data import rollout
from imitation.rewards import reward_nets
from stable_baselines3.dqn.policies import DQNPolicy

# Initialize environment

env_creator = partial(CliffWorldEnv, height=4, horizon=40, width=7, use_xy_obs=True)
env_single = env_creator()

state_env_creator = lambda: base_envs.ExposePOMDPStateWrapper(env_creator())

# This is just a vectorized environment because `generate_trajectories` expects one

state_venv = DummyVecEnv([state_env_creator] * 4)

_, _, pi = mce_partition_fh(env_single)

_, om = mce_occupancy_measures(env_single, pi=pi)

rng = np.random.default_rng()
expert = TabularPolicy(
    state_space=env_single.state_space,
    action_space=env_single.action_space,
    pi=pi,
    rng=rng,
)

expert_trajs = rollout.generate_trajectories(
    policy=expert,
    venv=state_venv,
    sample_until=rollout.make_min_timesteps(5000),
    rng=rng,
)

print("Expert stats: ", rollout.rollout_stats(expert_trajs))


def train_mce_irl(demos, hidden_sizes, lr=0.01, **kwargs):
    reward_net = reward_nets.BasicRewardNet(
        env_single.observation_space,
        env_single.action_space,
        hid_sizes=hidden_sizes,
        use_action=False,
        use_done=False,
        use_next_state=False,
    )

    mce_irl = MCEIRL(
        demos,
        env_single,
        reward_net,
        log_interval=250,
        optimizer_kwargs=dict(lr=lr),
        rng=rng,
    )
    occ_measure = mce_irl.train(**kwargs)

    imitation_trajs = rollout.generate_trajectories(
        policy=mce_irl.policy,
        venv=state_venv,
        sample_until=rollout.make_min_timesteps(5000),
        rng=rng,
    )
    print("Imitation stats: ", rollout.rollout_stats(imitation_trajs))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    env_single.draw_value_vec(occ_measure)
    plt.title("Occupancy for learned reward")
    plt.xlabel("Gridworld x-coordinate")
    plt.ylabel("Gridworld y-coordinate")
    plt.subplot(1, 2, 2)
    _, true_occ_measure = mce_occupancy_measures(env_single)
    env_single.draw_value_vec(true_occ_measure)
    plt.title("Occupancy for true reward")
    plt.xlabel("Gridworld x-coordinate")
    plt.ylabel("Gridworld y-coordinate")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    env_single.draw_value_vec(
        reward_net(th.as_tensor(env_single.observation_matrix), None, None, None)
        .detach()
        .numpy()
    )
    plt.title("Learned reward")
    plt.xlabel("Gridworld x-coordinate")
    plt.ylabel("Gridworld y-coordinate")
    plt.subplot(1, 2, 2)
    env_single.draw_value_vec(env_single.reward_matrix)
    plt.title("True reward")
    plt.xlabel("Gridworld x-coordinate")
    plt.ylabel("Gridworld y-coordinate")
    plt.show()

    return mce_irl


mce_irl_from_trajs = train_mce_irl(expert_trajs[0:10], hidden_sizes=[256])

model = DQN("MlpPolicy", state_venv, verbose=1,
            gamma=0.85,
            batch_size=32,
            #policy_kwargs=mce_irl_from_trajs.policy,
            )

# model.policy.init_weights(mce_irl_from_trajs.policy._modules)
# model.policy = mce_irl_from_trajs.policy
# model.policy.load_state_dict(state_dict=mce_irl_from_trajs.policy.__dict__)
# use mce_irl_from_trajs.reward_net?

# INTEGRATE LEARNED REWARD + POLICY IN RLHF PROCESS #

reward = mce_irl_from_trajs.reward_net

print(reward)
#reward_net = BasicRewardNet(
#    state_venv.observation_space, state_venv.action_space, normalize_input_layer=RunningNorm
#)

fragmenter = preference_comparisons.RandomFragmenter(
    warning_threshold=0,
    rng=rng,
)

# Gather synthetic human preferences

gatherer = preference_comparisons.SyntheticGatherer(rng=rng)

preference_model = preference_comparisons.PreferenceModel(reward)
reward_trainer = preference_comparisons.BasicRewardTrainer(
    preference_model=preference_model,
    loss=preference_comparisons.CrossEntropyRewardLoss(),
    epochs=3,
    rng=rng,
)

trajectory_generator = preference_comparisons.AgentTrainer(
    algorithm=model,
    reward_fn=reward,
    venv=state_venv,
    exploration_frac=0,
    rng=rng,
)

pref_comparisons = preference_comparisons.PreferenceComparisons(
    trajectory_generator,
    reward,
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

learned_reward_venv = RewardVecEnvWrapper(state_venv, reward.predict_processed)

print("PRINTING LEARNED REWARD")

# Train an agent with the learned reward

learner = PPO("MlpPolicy", learned_reward_venv, verbose=1)

learner.policy = mce_irl_from_trajs.policy

learner.learn(total_timesteps=200_000, progress_bar=True)

learner.save("dqn_cliff-world-learned-reward-PC-AND-MCE")



