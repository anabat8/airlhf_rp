import numpy as np
import imageio

from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
# import wandb as wandb
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import VecVideoRecorder
# from wandb.integration.sb3 import WandbCallback

from experiment3.Environment import Environment


class Utils:
    @staticmethod
    def train_with_learned_reward(learned_reward, save_path, tensorboard_dir, tb_log_name, wandb_project_name,
                                  wandb_save_path, ac_policy, env_object, policy_kwargs, policy_name="PPO",
                                  total_timesteps=400_000, lr=0.001, n_steps=32, batch_size=64, n_epochs=20,
                                  gae_lambda=0.8, gamma=0.98, clip_range=0.2, ent_coef=0.0, vf_coef=0.1):

        # Wrap environment with learned reward
        learned_reward_venv = RewardVecEnvWrapper(env_object.venv, learned_reward.predict_processed)
        # learned_reward_venv.render(mode="rgb_array")
        # learned_reward_venv = VecVideoRecorder(
        #     learned_reward_venv,
        #     f"videos/{tb_log_name}",
        #     record_video_trigger=lambda x: x % 2000 == 0,
        #     video_length=200,
        # )

        # Train an agent with the learned reward
        learner = None
        if policy_name == "PPO":
            learner = PPO(
                policy=ac_policy,
                env=learned_reward_venv,
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
                policy_kwargs=policy_kwargs,
                tensorboard_log=tensorboard_dir,
                verbose=1
            )

        # Initialize Wandb
        # config = {
        #     "policy_type": ac_policy,
        #     "total_timesteps": total_timesteps,
        #     "env_name": "CartPole-v0",
        # }
        #
        # run = wandb.init(
        #     project=wandb_project_name,
        #     config=config,
        #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        #     monitor_gym=True,  # auto-upload the videos of agents playing the game
        #     save_code=True,  # optional
        # )

        learner.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name=tb_log_name,
                      reset_num_timesteps=False,
                      # callback=WandbCallback(gradient_save_freq=100,
                      #                        model_save_path=wandb_save_path,
                      #                        verbose=2)
                      )

        learner.save(save_path)
        # run.finish()

    @staticmethod
    def evaluate_trained_agent_with_learned_reward(load_path, venv, policy_name="PPO", n_eval_episodes=100):
        learner = None
        if policy_name == "PPO":
            learner = PPO.load(load_path, env=venv)
        reward_mean, reward_std = evaluate_policy(learner.policy, venv, n_eval_episodes)
        reward_stderr = reward_std / np.sqrt(n_eval_episodes)
        print(f"Reward: {reward_mean:.0f} +/- {reward_stderr:.0f}")
        return reward_mean, reward_std

    @staticmethod
    def play_agent(load_path, venv, policy_name="PPO"):
        model = None
        if policy_name == "PPO":
            model = PPO.load(load_path, env=venv)
        obs = venv.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = venv.step(action)
            venv.render("human")
            # cv2.waitKey(300)
            # if np.any(dones):
            #     if np.any(rewards == 1):
            #         print(np.where(dones == 1))
            #         cv2.waitKey(2000)
            #     else:
            #         print("lost")

    @staticmethod
    def make_gif_agent(load_path, venv, policy_name="PPO"):
        model = None
        if policy_name == "PPO":
            model = PPO.load(load_path, env=venv)
        images = []
        obs = model.env.reset()
        img = model.env.render(mode="rgb_array")
        for i in range(350):
            images.append(img)
            action, _ = model.predict(obs)
            obs, _, _, _ = model.env.step(action)
            img = model.env.render(mode="rgb_array")

        imageio.mimsave("cartpole_irlhf_agent.gif", [np.array(img) for i, img in enumerate(images) if i % 2 == 0],
                        fps=29)

# utils = Utils()
# utils.play_agent(load_path="irlhf_agent/gen_policy/model.zip")
