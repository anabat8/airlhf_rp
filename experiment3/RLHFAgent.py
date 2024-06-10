import pathlib

import numpy as np
import torch as th
from imitation.algorithms import preference_comparisons
from imitation.policies import serialize
from imitation.policies.base import NormalizeFeaturesExtractor
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# device = th.device("cuda" if th.cuda.is_available() else "cpu")
device = th.device("cpu")


class RLHFAgent:

    def __init__(self, env_object, epochs=10):
        self.reward = BasicRewardNet(
            env_object.venv.observation_space, env_object.venv.action_space, normalize_input_layer=RunningNorm
        ).to(device)
        self.gen_algo = None
        self.fragmenter = preference_comparisons.RandomFragmenter(
            warning_threshold=0,
            rng=env_object.rng
        )
        # Gatherer for synthetic human preferences
        self.gatherer = preference_comparisons.SyntheticGatherer(rng=env_object.rng)
        # Preference model
        self.preference_model = preference_comparisons.PreferenceModel(self.reward)
        self.reward_trainer = preference_comparisons.BasicRewardTrainer(
            preference_model=self.preference_model,
            loss=preference_comparisons.CrossEntropyRewardLoss(),
            epochs=epochs,
            rng=env_object.rng, )
        self.trajectory_generator = None
        self.pref_comparisons = None

    def init_gen_algo(self, policy_name, ac_policy, env_object, lr=0.001, n_steps=32, batch_size=64, n_epochs=20,
                      gae_lambda=0.8, gamma=0.98, clip_range=0.2, ent_coef=0.0, vf_coef=0.1):
        if policy_name == "PPO":
            self.gen_algo = PPO(
                policy=ac_policy,
                env=env_object.venv,
                learning_rate=lr,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gae_lambda=gae_lambda,
                gamma=gamma,
                clip_range=clip_range,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                seed=env_object.seed,
                policy_kwargs=dict(
                    features_extractor_class=NormalizeFeaturesExtractor,
                    features_extractor_kwargs=dict(normalize_class=RunningNorm),
                ),
                device=device
            )

    def init_trajectory_gen(self, env_object, exploration_frac=0.05):
        self.trajectory_generator = preference_comparisons.AgentTrainer(
            algorithm=self.gen_algo,
            reward_fn=self.reward,
            venv=env_object.venv,
            exploration_frac=exploration_frac,
            rng=env_object.rng, )

    def set_reward_from_airl(self, airl_reward_net):
        self.reward = airl_reward_net
        self.preference_model = preference_comparisons.PreferenceModel(self.reward)

    def train(self, save_path, env_object, num_it=60, fragment_length=100, transition_oversampling=1,
              initial_comp_frac=0.1, initial_epoch_multiplier=4, query_schedule="hyperbolic", total_timesteps=400_000,
              total_human_comparisons=250):
        self.pref_comparisons = preference_comparisons.PreferenceComparisons(
            self.trajectory_generator,
            self.reward,
            num_iterations=num_it,
            fragmenter=self.fragmenter,
            preference_gatherer=self.gatherer,
            reward_trainer=self.reward_trainer,
            fragment_length=fragment_length,
            transition_oversampling=transition_oversampling,
            initial_comparison_frac=initial_comp_frac,
            allow_variable_horizon=False,
            initial_epoch_multiplier=initial_epoch_multiplier,
            query_schedule=query_schedule,
        )

        env_object.venv.seed(env_object.seed)

        learner_rewards_before_training, _ = evaluate_policy(
            self.gen_algo, env_object.venv, 100, return_episode_rewards=True
        )

        # Train network, learn reward function through human preferences
        self.pref_comparisons.train(total_timesteps=total_timesteps, total_comparisons=total_human_comparisons)

        env_object.venv.seed(env_object.seed)

        learner_rewards_after_training, _ = evaluate_policy(
            self.gen_algo, env_object.venv, 100, return_episode_rewards=True
        )

        print(
            "Rewards before training:",
            np.mean(learner_rewards_before_training),
            "+/-",
            np.std(learner_rewards_before_training),
        )
        print(
            "Rewards after training:",
            np.mean(learner_rewards_after_training),
            "+/-",
            np.std(learner_rewards_after_training),
        )

        # Save model
        self.save(save_path=save_path, learner_rewards=learner_rewards_after_training)

    def save(self, save_path: pathlib.Path, learner_rewards):
        save_path.mkdir(parents=True, exist_ok=True)
        th.save(learner_rewards, save_path / "learner_rewards.pt")
        th.save(self.pref_comparisons.model, save_path / "reward_net.pt")
        serialize.save_stable_model(
            save_path / "gen_policy",
            self.gen_algo,
        )

# rlhfAgent = RLHFAgent()
# rlhfAgent.init_gen_algo(policy_name="PPO", ac_policy=MlpPolicy)
# rlhfAgent.init_trajectory_gen()
# rlhfAgent.train(save_path=pathlib.Path('rlhf_agent'))

# reward_net = th.load("rlhf_agent/reward_net.pt")
#
# utils = Utils()
# utils.train_with_learned_reward(learned_reward=reward_net, save_path="rlhf_agent/rlhf_agent_trained_with_learned_reward")
# utils.evaluate_trained_agent_with_learned_reward(load_path="rlhf_agent/rlhf_agent_trained_with_learned_reward")