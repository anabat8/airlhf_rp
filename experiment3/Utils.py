import numpy as np
import torch

from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

from experiment3.Environment import Environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Utils:
    @staticmethod
    def train_with_learned_reward(learned_reward, save_path, tensorboard_dir, tb_log_name,
                                  config, ac_policy, env_object: Environment, policy_kwargs,
                                  total_timesteps=400_000):

        # Wrap environment with learned reward
        learned_reward_venv = RewardVecEnvWrapper(env_object.venv, learned_reward.predict_processed)

        # Train an agent with the learned reward
        learner = None

        if config['policy_name'] == "ppo":
            learner = PPO(
                policy=ac_policy,
                env=learned_reward_venv,
                learning_rate=config['learning_rate'],
                n_steps=config['n_steps'],
                batch_size=config['batch_size'],
                n_epochs=config['n_epochs'],
                gae_lambda=config['gae_lambda'],
                gamma=config['gamma'],
                clip_range=config['clip_range'],
                ent_coef=config['ent_coef'],
                vf_coef=config['vf_coef'],
                seed=env_object.seed,
                policy_kwargs=policy_kwargs,
                tensorboard_log=tensorboard_dir,
                verbose=1,
                device=device
            )

        # Use testing environment for evaluation
        eval_freq = total_timesteps // 50
        eval_callback = EvalCallback(env_object.venv, eval_freq=max(eval_freq // config['num_envs'], 1))

        learner.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name=tb_log_name,
                      reset_num_timesteps=False, callback=eval_callback
                      )

        learner.save(save_path)

    @staticmethod
    def evaluate_trained_agent_with_true_reward(load_path, venv, policy_name="ppo", n_eval_episodes=100):
        learner = None
        if policy_name == "ppo":
            learner = PPO.load(load_path, env=venv)
        # To evaluate the policy with the true reward, DO NOT wrap the vector env with RewardVecWrapper
        # Instead pass the vectorized test env as it is to the evaluate_policy function
        reward_mean, reward_std = evaluate_policy(learner.policy, venv, n_eval_episodes)
        reward_stderr = reward_std / np.sqrt(n_eval_episodes)
        print(f"Reward: {reward_mean:.0f} +/- {reward_stderr:.0f}")
        return reward_mean, reward_std
