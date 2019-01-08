from gym.envs.registration import register

register(
    id='mdp-v0',
    entry_point='mdp_environment.envs:MdpEnvLinStatic')

register(
    id='mdp-v1',
    entry_point='mdp_environment.envs:MdpEnvLinVariable')

register(
    id='mdp-v2',
    entry_point='mdp_environment.envs:MdpEnvPlanarStatic')

register(
    id='mdp-v3',
    entry_point='mdp_environment.envs:MdpEnvPlanarVariable')

register(
    id='mdp-v4',
    entry_point='mdp_environment.envs:MdpEnvLinCorridor')
