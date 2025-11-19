import numpy as np
import os
import operator
import torch
import torch.nn as nn
from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
from gym_collision_avoidance.envs import util
from gym_collision_avoidance.envs import Config
from ddpg.network.LSTMNet import LSTM_ActorNet
from ddpg.network.SpikLSTMNet import SpikLSTM_ActorNet
from ddpg.network.SpikGTrNet import SpikGTr_ActorNet
from ddpg.network.SDDPGNet import SDDPG_ActorNet


class DDPGPolicy(InternalPolicy):
    def __init__(self, str="SDDPG"):
        InternalPolicy.__init__(self, str=str)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network=None
    def initialize_network(self, **kwargs):
        raise NotImplementedError

    def find_next_action(self, obs, agents, i):
        mult_actions = False
        if type(obs) == dict:
            # Turn the dict observation into a flattened vector
            vec_obs = np.array([])
            for state in Config.STATES_IN_OBS:
                vec_obs = np.hstack([vec_obs, obs[state].flatten()])
            vec_obs = np.expand_dims(vec_obs, axis=0)
        else:
            vec_obs=obs
            mult_actions=True
        with torch.no_grad():
            actions = self.network(vec_obs).to('cpu')
            actions = actions.numpy()
        actions = np.clip(actions, [0., -1.], [1., 1.])

        if not mult_actions:
            [vx, vw] = actions[0]
            if Config.USING_MAX_HEADCHANGE:
                delta_heading = vw * agents[i].max_heading_change
            else:
                delta_heading = vw * Config.DT
            speed = agents[i].pref_speed * vx
            action = np.array([speed, delta_heading])
            actions=action
        else:
            for n in range(len(actions)):
                actions[n][0]*= agents[n].pref_speed
                if Config.USING_MAX_HEADCHANGE:
                    actions[n][1] *= agents[n].max_heading_change
                else:
                    actions[n][1] *= Config.DT
        return actions

class LSTM_DDPGPolicy(DDPGPolicy):
    def __init__(self):
        DDPGPolicy.__init__(self, "LSTM_DDPG")

    def initialize_network(self, **kwargs):
        self.network = LSTM_ActorNet(obs_size=4,oas_size=7,action_space=2)
        self.network.to('cpu')
        checkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '/DDPG/checkpoints/LSTM_ActorNet.pth'
        self.network.load_state_dict(torch.load(checkpt_dir))
        self.network.to(self.device)

class SpikLSTM_DDPGPolicy(DDPGPolicy):
    def __init__(self):
        DDPGPolicy.__init__(self, "SpikLSTM_DDPG")

    def initialize_network(self, **kwargs):
        self.network = SpikLSTM_ActorNet(obs_size=4,oas_size=7,action_space=2)
        self.network.to('cpu')
        checkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '/DDPG/checkpoints/SpikLSTM_ActorNet.pth'
        self.network.load_state_dict(torch.load(checkpt_dir))
        self.network.to(self.device)

class SpikGTr_DDPGPolicy(DDPGPolicy):
    def __init__(self):
        DDPGPolicy.__init__(self, "SpikGTr_DDPG")

    def initialize_network(self, **kwargs):
        self.network = SpikGTr_ActorNet(obs_size=4,oas_size=7,action_space=2)
        self.network.to('cpu')
        checkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '/DDPG/checkpoints/SpikGTr_ActorNet.pth'
        self.network.load_state_dict(torch.load(checkpt_dir))
        self.network.to(self.device)

class SpikDDPGPolicy(DDPGPolicy):
    def __init__(self):
        DDPGPolicy.__init__(self, "SpikDDPG")

    def initialize_network(self, **kwargs):
        self.network = SDDPG_ActorNet(obs_size=4,oas_size=36,action_space=2)
        self.network.to('cpu')
        checkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '/DDPG/checkpoints/SDDPG_ActorNet.pth'
        self.network.load_state_dict(torch.load(checkpt_dir))
        self.network.to(self.device)

    def find_next_action(self, obs, agents, i):
        mult_actions = False
        if type(obs) == dict:
            # Turn the dict observation into a flattened vector
            vec_obs = np.array([])
            for state in Config.STATES_IN_OBS:
                vec_obs = np.hstack([vec_obs, obs[state].flatten()])
            vec_obs = np.expand_dims(vec_obs, axis=0)
        else:
            vec_obs=obs
            mult_actions=True
        with torch.no_grad():
            actions = self.network(vec_obs).to('cpu')
            actions = actions.numpy()

        # actions = np.clip(actions, [0., -1.], [1., 1.])
        lsmax = 1.0
        lsmin = 0.05
        decode_action = []
        for action in actions:
            lspd = action[0] * (lsmax - lsmin) + lsmin
            rspd = action[1] * (lsmax - lsmin) + lsmin
            linear = (lspd + rspd) / 2
            angular = (rspd - lspd) / 0.23
            decode_action.append([linear, angular])

        actions=decode_action

        if not mult_actions:
            [vx, vw] = actions[0]
            if Config.USING_MAX_HEADCHANGE:
                delta_heading = vw * agents[i].max_heading_change
            else:
                delta_heading = vw * Config.DT
            speed = agents[i].pref_speed * vx
            action = np.array([speed, delta_heading])
            actions=action
        else:
            for n in range(len(actions)):
                actions[n][0]*= agents[n].pref_speed
                if Config.USING_MAX_HEADCHANGE:
                    actions[n][1] *= agents[n].max_heading_change
                else:
                    actions[n][1] *= Config.DT
        return actions


if __name__ == '__main__':
    # policy = SpikGTr_DDPGPolicy()
    policy = SpikLSTM_DDPGPolicy()
    policy.initialize_network()

    print('done')