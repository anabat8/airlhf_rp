from imitation.data import rollout
from imitation.policies.exploration_wrapper import ExplorationWrapper
from imitation.policies.serialize import load_policy
from stable_baselines3 import PPO

from experiment3.Environment import Environment


class Expert:

    def __init__(self, env: Environment, policy_name, expert_type):
        self.expert_policy = None
        self.env = env
        self.policy_name = policy_name
        self.demonstrations = None
        self.expert_type = expert_type

    def init_expert_policy(self):
        if self.policy_name == "ppo" and self.expert_type == "optimal":
            self.expert_policy = load_policy(
                "ppo-huggingface",
                organization="HumanCompatibleAI",
                env_name=self.env.env_id,
                venv=self.env.venv,
            )
        elif self.policy_name == "ppo" and self.expert_type == "suboptimal":
            # Make sure create_suboptimal_policy_ppo has been called before
            # such that the suboptimal policy was created and can be loaded
            self.expert_policy = self.load_suboptimal_policy_ppo()
            # self.expert_policy = ExplorationWrapper(self.expert_policy)
        return self.expert_policy

    def load_suboptimal_policy_ppo(self):
        policy = PPO.load("airl_agent/suboptimal_expert/ppo_cartpole")
        return policy

    def create_suboptimal_policy_ppo(self):
        model = PPO("MlpPolicy", self.env.venv, verbose=1)
        model.learn(total_timesteps=25000)  # tuned optimal expert uses 1e5 timesteps (RL ZOO)
        model.save("airl_agent/suboptimal_expert/ppo_cartpole")

    def init_rollouts(self, min_timesteps=None, min_episodes=60):
        self.demonstrations = rollout.rollout(
            self.expert_policy,
            self.env.venv,
            rollout.make_sample_until(min_timesteps=min_timesteps, min_episodes=min_episodes),
            rng=self.env.rng,
        )

        print("Expert stats: ", rollout.rollout_stats(self.demonstrations))

        return self.demonstrations
