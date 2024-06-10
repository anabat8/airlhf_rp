from imitation.data import rollout
from imitation.policies.serialize import load_policy

from experiment3.Environment import Environment


class Expert:

    def __init__(self, env: Environment, policy_name):
        self.expert_policy = None
        self.env = env
        self.policy_name = policy_name
        self.demonstrations = None

    def init_expert_policy(self):
        if self.policy_name == "PPO":
            self.expert_policy = load_policy(
                "ppo-huggingface",
                organization="HumanCompatibleAI",
                env_name=self.env.env_id,
                venv=self.env.venv,
            )
        return self.expert_policy

    def init_rollouts(self, min_timesteps=None, min_episodes=60):
        self.demonstrations = rollout.rollout(
            self.expert_policy,
            self.env.venv,
            rollout.make_sample_until(min_timesteps=min_timesteps, min_episodes=min_episodes),
            rng=self.env.rng,
        )

        print("Expert stats: ", rollout.rollout_stats(self.demonstrations))

        return self.demonstrations
