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
    entry_point='mdp_environment.envs:MdpEnvLinCorridor',
    kwargs={"size": 45, "dim": 25})

register(
    id='mdp-v5',
    entry_point='mdp_environment.envs:MdpEnvLinCorridorTricky',
    kwargs={"size": 45, "dim": 25})

register(
    id='mdp-v4:2',
    entry_point='mdp_environment.envs:MdpEnvLinCorridor',
    kwargs={"size": 45, "dim": 2})

register(
        id='mdp-v4:5',
    entry_point='mdp_environment.envs:MdpEnvLinCorridor',
    kwargs={"size": 45, "dim": 5})

register(
        id='mdp-v4:10',
    entry_point='mdp_environment.envs:MdpEnvLinCorridor',
    kwargs={"size": 45, "dim": 10})

register(
        id='mdp-v4:25',
    entry_point='mdp_environment.envs:MdpEnvLinCorridor',
    kwargs={"size": 45, "dim": 25})

register(
        id='mdp-v4:50',
    entry_point='mdp_environment.envs:MdpEnvLinCorridor',
    kwargs={"size": 45, "dim": 50})
