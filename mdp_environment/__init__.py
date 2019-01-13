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

register(
    id='mdp:2-v4',
    entry_point='mdp_environment.envs:MdpEnvLinCorridor',
    kwargs={"size": 25, "dim": 2})

register(
        id='mdp:5-v4',
    entry_point='mdp_environment.envs:MdpEnvLinCorridor',
    kwargs={"size": 25, "dim": 5})

register(
        id='mdp:10-v4',
    entry_point='mdp_environment.envs:MdpEnvLinCorridor',
    kwargs={"size": 25, "dim": 10})

register(
        id='mdp:25-v4',
    entry_point='mdp_environment.envs:MdpEnvLinCorridor',
    kwargs={"size": 25, "dim": 25})

register(
        id='mdp:50-v4',
    entry_point='mdp_environment.envs:MdpEnvLinCorridor',
    kwargs={"size": 25, "dim": 50})
