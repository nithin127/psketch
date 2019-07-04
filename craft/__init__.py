import os
from gym.envs.registration import register

register(
    id='CraftEnv-v0',
    entry_point='craft.envs:CraftEnv',
)


register(
    id='CraftEnvFull-v0',
    entry_point='craft.envs:CraftEnvFull',
)
