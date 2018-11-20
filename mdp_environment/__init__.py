from gym.envs.registration import register

register(
	id='mdp-v0', 
	entry_point='mdp_environment.envs:MdpEnvLin', )
