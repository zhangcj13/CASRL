import numpy as np
import os
import operator
from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
from gym_collision_avoidance.envs import util
from gym_collision_avoidance.envs.policies.GA3C_CADRL import network
import torch
from gym_collision_avoidance.envs import Config
import torch.nn.functional as F

from ga3c.network.LSTMNet import LSTMNet
from ga3c.network.SpikLSTMNet import SpikLSTMNet

class GA3CPolicy(InternalPolicy):
    """ Pre-trained policy from `Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning <https://arxiv.org/pdf/1805.01956.pdf>`_

    By default, loads a pre-trained LSTM network (GA3C-CADRL-10-LSTM from the paper). There are 11 discrete actions with max heading angle change of $\pm \pi/6$.

    """
    def __init__(self,str="GA3C"):
        InternalPolicy.__init__(self, str=str)

        self.possible_actions = network.Actions()
        num_actions = self.possible_actions.num_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = None

    def initialize_network(self, **kwargs):
        raise NotImplementedError

    def find_next_action(self, obs, agents, i):
        mult_actions = False

        avg_vec = np.array(Config.NN_INPUT_AVG_VECTOR, dtype=np.float32)
        std_vec = np.array(Config.NN_INPUT_STD_VECTOR, dtype=np.float32)

        if type(obs) == dict:
            # Turn the dict observation into a flattened vector
            pref_speed = obs['pref_speed']
            vec_obs = np.array([])
            for state in Config.STATES_IN_OBS:
                if state not in Config.STATES_NOT_USED_IN_POLICY:
                    vec_obs = np.hstack([vec_obs, obs[state].flatten()])
            vec_obs = np.expand_dims(vec_obs, axis=0)
            vec_obs = (vec_obs - avg_vec) / std_vec
        else:
            vec_obs = (obs[:, 1:] - avg_vec) / std_vec
            mult_actions=True

        logits_p, _  = self.network(vec_obs)

        softmax_p = F.softmax(logits_p, dim=-1)
        predictions = softmax_p.detach().cpu().numpy()

        if not mult_actions:
            action_index = np.argmax(predictions)
            raw_action = self.possible_actions.actions[action_index]
            action = np.array([agents[i].pref_speed*raw_action[0], raw_action[1]])
        else:
            action=[]
            for n,prediction in enumerate(predictions):
                action_index = np.argmax(prediction)
                raw_action = self.possible_actions.actions[action_index]
                tmp_action = np.array([agents[n].pref_speed * raw_action[0], raw_action[1]])
                action.append(tmp_action)
            action=np.asarray(action)
        return action

class LSTM_GA3CPolicy(GA3CPolicy):
    def __init__(self):
        GA3CPolicy.__init__(self, "LSTM_GA3C")

    def initialize_network(self, **kwargs):
        self.network = LSTMNet(obs_size=4, oas_size=7, num_actions=11)
        self.network.to('cpu')
        checkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '/GA3C/checkpoints/LSTMNet.pth'
        self.network.load_state_dict(torch.load(checkpt_dir))
        self.network.to(self.device)

class SpikLSTM_GA3CPolicy(GA3CPolicy):
    def __init__(self):
        GA3CPolicy.__init__(self, "SpikLSTM_GA3C")

    def initialize_network(self, **kwargs):
        self.network = SpikLSTMNet(obs_size=4, oas_size=7, num_actions=11)
        self.network.to('cpu')
        checkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '/GA3C/checkpoints/SpikLSTMNet.pth'
        self.network.load_state_dict(torch.load(checkpt_dir))
        self.network.to(self.device)

if __name__ == '__main__':
    policy = LSTM_GA3CPolicy()
    policy.initialize_network()