from imitation.data import rollout
from imitation.policies.exploration_wrapper import ExplorationWrapper
from imitation.policies.serialize import load_policy

from experiment3.Environment import Environment


class Expert:

    def __init__(self, env: Environment, policy_name, expert_type):
        self.expert_policy = None
        self.env = env
        self.policy_name = policy_name
        self.demonstrations = None
        self.expert_type = expert_type

    def init_expert_policy(self, random_prob, switch_prob):
        if self.policy_name == "ppo":
            self.expert_policy = load_policy(
                "ppo-huggingface",
                organization="HumanCompatibleAI",
                env_name=self.env.env_id,
                venv=self.env.venv,
            )
            if self.expert_type == "suboptimal":
                self.expert_policy = ExplorationWrapper(
                    policy=self.expert_policy,
                    venv=self.env.venv,
                    random_prob=random_prob,
                    switch_prob=switch_prob,
                    rng=self.env.rng,
                    deterministic_policy=True
                    # We use a deterministic version of the expert policy when having suboptimal expert,
                    # to have a clear baseline of optimal behavior before introducing controlled randomness
                )

        print("Random probability is " + str(random_prob) + " and switch probability is " + str(switch_prob))

        return self.expert_policy

    def init_rollouts(self, min_timesteps=None, min_episodes=60):
        self.demonstrations = rollout.rollout(
            self.expert_policy,
            self.env.venv,
            rollout.make_sample_until(min_timesteps=min_timesteps, min_episodes=min_episodes),
            rng=self.env.rng,
        )

        print("Expert stats: ", rollout.rollout_stats(self.demonstrations))
        print("Expert is " + self.expert_type)

        return self.demonstrations
