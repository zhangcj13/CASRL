import numpy as np

from gym_collision_avoidance.envs.policies.ExternalPolicy import ExternalPolicy
from gym_collision_avoidance.envs import Config

class Actions():
    def __init__(self):
        self.actions = np.mgrid[1.0:1.1:0.5, -np.pi/6:np.pi/6+0.01:np.pi/12].reshape(2, -1).T
        self.actions = np.vstack([self.actions,np.mgrid[0.5:0.6:0.5, -np.pi/6:np.pi/6+0.01:np.pi/6].reshape(2, -1).T])
        self.actions = np.vstack([self.actions,np.mgrid[0.0:0.1:0.5, -np.pi/6:np.pi/6+0.01:np.pi/6].reshape(2, -1).T])
        self.num_actions = len(self.actions)

class LearningPolicyPPOs(ExternalPolicy):
    """ An RL policy that is still being trained or otherwise fed actions from an external script, but still needs to convert the external actions to this env's format
    """
    def __init__(self):
        ExternalPolicy.__init__(self, str="learning")
        self.is_still_learning = True
        self.ppo_or_learning_policy = True

        self.possible_actions = Actions()

    def external_action_to_action(self, agent, external_action):
        """ Convert the external_action into an action for this environment using properties about the agent.

        For instance, RL network might have continuous outputs between [0-1], which could be 
        scaled by this method to correspond to a speed between [0, pref_speed],
        without the RL network needing to know the agent's preferred speed.

        Args:
            agent (:class:`~gym_collision_avoidance.envs.agent.Agent`): the agent who has this policy
            external_action (int, array, ...): what the learning system returned for an action

        Returns:
            [speed, heading_change] command

        """

        # external_action: [speed scaling btwn 0-1, max heading angle delta btwn 0-1]
        if Config.DISCRETE_CONTROL_FLAG:
            raw_action = self.possible_actions.actions[int(external_action)]
            actions = np.array([agent.pref_speed * raw_action[0], raw_action[1]])
        else:
            heading_change = agent.max_heading_change*external_action[1]
            speed = agent.pref_speed * (external_action[0]+1)/2.0
            actions = np.array([speed, heading_change])
        # print('final action:',actions)
        return actions