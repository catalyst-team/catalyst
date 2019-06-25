from catalyst import utils


def structed2dict_trajectory(trajectory):
    observations, actions, rewards, dones = trajectory
    observations = utils.structed2dict(observations)
    actions = utils.structed2dict(actions)
    trajectory = observations, actions, rewards, dones
    return trajectory


def dict2structed_trajectory(trajectory):
    observations, actions, rewards, dones = trajectory
    observations = utils.dict2structed(observations)
    actions = utils.dict2structed(actions)
    trajectory = observations, actions, rewards, dones
    return trajectory
