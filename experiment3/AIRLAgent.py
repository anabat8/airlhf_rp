import pathlib

import numpy as np
from imitation.policies import serialize
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch as torch

from experiment3.Expert import Expert

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class AIRLAgent:

    def __init__(self, env_object, expert_type, nr_demonstrations=60, random_prob=0, switch_prob=0):
        self.gen_algo = None
        self.reward = BasicShapedRewardNet(
            observation_space=env_object.venv.observation_space,
            action_space=env_object.venv.action_space,
            normalize_input_layer=RunningNorm,
        ).to(device)
        self.expert = Expert(env_object, "ppo", expert_type)
        self.expert_policy = self.expert.init_expert_policy(random_prob=random_prob, switch_prob=switch_prob)
        self.expert_demonstrations = self.expert.init_rollouts(min_episodes=nr_demonstrations)
        self.airl_trainer = None

    def init_gen_algo(self, ac_policy, env_object, batch_size=64, ent_coef=0.0, lr=0.0005, gamma=0.95, clip_range=0.1,
                      vf_coef=0.1, n_epochs=5):
        if self.expert.policy_name == "ppo":
            self.gen_algo = PPO(
                env=env_object.venv,
                policy=ac_policy,
                batch_size=batch_size,
                ent_coef=ent_coef,
                learning_rate=lr,
                gamma=gamma,
                clip_range=clip_range,
                vf_coef=vf_coef,
                n_epochs=n_epochs,
                seed=env_object.seed,
                device=device
            )

    def train(self, env_object, train_steps=400_000, batch_size=2048, buffer_cap=512, n_disc_updates_per_round=16):
        self.airl_trainer = AIRL(
            demonstrations=self.expert_demonstrations,
            demo_batch_size=batch_size,
            gen_replay_buffer_capacity=buffer_cap,
            n_disc_updates_per_round=n_disc_updates_per_round,
            venv=env_object.venv,
            gen_algo=self.gen_algo,
            reward_net=self.reward
        )

        env_object.venv.seed(env_object.seed)

        learner_rewards_before_training, _ = evaluate_policy(
            self.gen_algo, env_object.venv, 100, return_episode_rewards=True
        )

        self.airl_trainer.train(train_steps)

        env_object.venv.seed(env_object.seed)

        # Runs policy for n_eval_episodes episodes, returns a list of rewards and episode lengths per episode
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
        self.save(save_path=pathlib.Path('airl_agent'), learner_rewards=learner_rewards_after_training)

    def save(self, save_path: pathlib.Path, learner_rewards):
        """Save discriminator and generator."""
        # do not serialize the whole trainer (including e.g. expert demonstrations)
        save_path.mkdir(parents=True, exist_ok=True)
        # Save the learned rewards
        torch.save(learner_rewards, save_path / "learner_rewards.pt")
        # reward_train is the same as reward_net
        # Save the reward model
        torch.save(self.airl_trainer.reward_train, save_path / "reward_net.pt")
        # Save the unshaped version of reward network used for testing
        torch.save(self.airl_trainer.reward_test, save_path / "reward_test.pt")
        # Save the policy
        serialize.save_stable_model(
            save_path / "gen_policy",
            self.airl_trainer.gen_algo,
        )
