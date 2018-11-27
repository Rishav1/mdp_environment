from gym.envs.registration import register

register(
	id='mdp-v0', 
	entry_point='mdp_environment.envs:MdpEnvLinStatic')

register(
	id='mdp-v1',
	entry_point='mdp_environment.envs:MdpEnvLinVariable')
