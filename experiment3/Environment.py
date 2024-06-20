import numpy as np
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env


class Environment:

    def __init__(self, env_id, seed, num_envs):
        self.venv = None
        self.test_venv = None
        self.env_id = env_id
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.num_envs = num_envs

    # Vector environment utilized in training
    def init_vec_env(self):
        self.venv = make_vec_env(env_name=self.env_id,
                                 rng=self.rng,
                                 n_envs=self.num_envs,
                                 post_wrappers=[
                                     lambda env, _: RolloutInfoWrapper(env)
                                 ],
                                 )
        self.venv.render(mode="rgb_array")
        return self.venv

    # Vector environment utilized in testing, when evaluating the policy with the true reward
    def init_test_vec_env(self):
        self.test_venv = make_vec_env(env_name=self.env_id,
                                      rng=self.rng,
                                      n_envs=self.num_envs,
                                      post_wrappers=[
                                          lambda env, _: RolloutInfoWrapper(env)
                                      ],
                                      )
        self.test_venv.render(mode="rgb_array")
        return self.test_venv
