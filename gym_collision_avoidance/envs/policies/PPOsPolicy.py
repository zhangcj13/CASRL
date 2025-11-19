import numpy as np
import os
import operator
from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
from gym_collision_avoidance.envs import util
from gym_collision_avoidance.envs.policies.GA3C_CADRL import network
import torch
from gym_collision_avoidance.envs import Config
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from dppo.network.LSTMPolicy import LSTMActor as DPPO_LSTMActor
from dppo.network.SpikGTrPolicy import SpikGTrActor as DPPO_SpikGTrActor
from dppo.network.SpikLSTMPolicy import SpikLSTMActor as DPPO_SpikLSTMActor
from dppo.network.SpikGTrANPolicy import SpikGTrANActor as DPPO_SpikGTrANActor
from dppo.network.SpikMFPolicy import SpikGMFActor as DPPO_SpikGMFActor
from dppo.network.GTrMeanPolicy import GTrMeanActor as DPPO_GTrMeanActor

class Actions():
    def __init__(self):
        self.actions = np.mgrid[1.0:1.1:0.5, -np.pi/6:np.pi/6+0.01:np.pi/12].reshape(2, -1).T
        self.actions = np.vstack([self.actions,np.mgrid[0.5:0.6:0.5, -np.pi/6:np.pi/6+0.01:np.pi/6].reshape(2, -1).T])
        self.actions = np.vstack([self.actions,np.mgrid[0.0:0.1:0.5, -np.pi/6:np.pi/6+0.01:np.pi/6].reshape(2, -1).T])
        self.num_actions = len(self.actions)

class ActorCritic(nn.Module):
    def __init__(self, obs_size=4, oas_size=7, action_dim=2, ACT_ARCH=None,CRT_ARCH=None,device=None):
        super(ActorCritic, self).__init__()
        # actor
        self.actor=None
        self.critic=None
        if ACT_ARCH is not None:
            self.actor = globals()[ACT_ARCH](obs_size=obs_size, oas_size=oas_size, action_space=action_dim,device=device)
        # critic
        if CRT_ARCH is not None:
            self.critic = globals()[CRT_ARCH](obs_size=obs_size, oas_size=oas_size, action_space=action_dim,device=device)
        self.device=device

class PPOsPolicy(InternalPolicy):
    """ Pre-trained policy from `Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning <https://arxiv.org/pdf/1805.01956.pdf>`_

    By default, loads a pre-trained LSTM network (GA3C-CADRL-10-LSTM from the paper). There are 11 discrete actions with max heading angle change of $\pm \pi/6$.

    """
    def __init__(self,str="PPOs"):
        InternalPolicy.__init__(self, str=str)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = None
        self.has_continuous_action_space = True

        self.possible_actions = Actions()

    def initialize_network(self, **kwargs):
        raise NotImplementedError

    def find_next_action(self, obs, agents, idx):
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
            mult_actions = True


        state = self.rescale_input(vec_obs, seq_cut=True)

        if self.has_continuous_action_space:
            with torch.no_grad():
                means = self.network.actor(state)
                actions = means.detach().cpu().numpy()

            if not mult_actions:
                [vx, vw] = actions[0]

                heading_change = agents[idx].max_heading_change * vw
                speed = agents[idx].pref_speed * (vx + 1) / 2.0
                actions = np.array([speed, heading_change])
            else:
                for n in range(len(actions)):
                    actions[n][0] = agents[n].pref_speed*(actions[n][0]+1.0)/2.0
                    actions[n][1] = agents[n].max_heading_change*actions[n][1]
        else:
            with torch.no_grad():
                probs = self.network.actor(state)
                action_ids = torch.argmax(probs,-1).cpu().numpy()

            if not mult_actions:
                action_id = action_ids[0]
                raw_action = self.possible_actions.actions[int(action_id)]
                actions = np.array([agents[idx].pref_speed * raw_action[0], raw_action[1]])
            else:
                actions=[]
                for n, id in enumerate(idx):
                    action_id = action_ids[n]
                    raw_action = self.possible_actions.actions[int(action_id)]
                    action = np.array([agents[id].pref_speed * raw_action[0], raw_action[1]])
                    actions.append(action)
                # for n in range(len(action_ids)):
                #     action_id = action_ids[n]
                #     raw_action = self.possible_actions.actions[int(action_id)]
                #     action = np.array([agents[n].pref_speed * raw_action[0], raw_action[1]])
                #     actions.append(action)
                actions=np.asarray(actions)
        return actions

    def rescale_input(self, x_normalized, return_vec=False, batch_first=False, seq_cut=False):
        host_agent_vec = x_normalized[:,
                         Config.FIRST_STATE_INDEX:Config.HOST_AGENT_STATE_SIZE + Config.FIRST_STATE_INDEX:]
        # host_agent_vec = torch.Tensor(host_agent_vec).to(self.device)
        host_agent_vec = Variable(torch.from_numpy(host_agent_vec)).float().to(self.device)
        # if Config.USING_LASER:
        #     laser_scan = x_normalized[:, Config.HOST_AGENT_STATE_SIZE + Config.FIRST_STATE_INDEX:]
        #     laser_scan = torch.Tensor(laser_scan).to(self.device)
        #     return host_agent_vec, laser_scan

        num_other_agents = np.clip(x_normalized[:, 0] + 1, 0, 1e6)
        other_agent_vec = x_normalized[:, Config.HOST_AGENT_STATE_SIZE + Config.FIRST_STATE_INDEX:]
        # other_agent_vec = torch.Tensor(other_agent_vec).to(self.device)
        other_agent_vec = Variable(torch.from_numpy(other_agent_vec)).float().to(self.device)

        if return_vec:
            return host_agent_vec, other_agent_vec, num_other_agents
        other_agent_seq = torch.reshape(other_agent_vec, [-1, Config.MAX_NUM_OTHER_AGENTS_OBSERVED,
                                                          Config.OTHER_AGENT_FULL_OBSERVATION_LENGTH])

        if seq_cut:
            max_effective_length = np.max(num_other_agents).astype(np.int)
            max_effective_length = max_effective_length if max_effective_length >= 1 else 1
            other_agent_seq = other_agent_seq[:, :max_effective_length, :]
        if not batch_first:
            other_agent_seq = torch.transpose(other_agent_seq, 0, 1)
        return host_agent_vec, other_agent_seq, num_other_agents

class LSTM_DPPOPolicy(PPOsPolicy):
    def __init__(self):
        PPOsPolicy.__init__(self, "LSTM_DPPO")

    def initialize_network(self, **kwargs):
        actor = DPPO_LSTMActor(obs_size=4, oas_size=7, action_space=2,device=self.device)
        actor.to('cpu')
        checkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '/PPOs/checkpoints/DPPO_LSTMActor.pth'
        actor.load_state_dict(torch.load(checkpt_dir))
        actor.to(self.device)
        self.network=ActorCritic()
        self.network.actor=actor

# DC: discrete control
class DC_LSTM_DPPOPolicy(PPOsPolicy):
    def __init__(self):
        PPOsPolicy.__init__(self, "LSTM_DPPO_DC")
        self.has_continuous_action_space = False
    def initialize_network(self, **kwargs):
        actor = DPPO_LSTMActor(obs_size=4, oas_size=7, action_space=11,device=self.device,continuous_action_space=False)
        actor.to('cpu')
        checkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '/PPOs/checkpoints/DPPO_LSTMActor_DC.pth'
        actor.load_state_dict(torch.load(checkpt_dir))
        actor.to(self.device)
        self.network=ActorCritic()
        self.network.actor=actor

class DC_SpikLSTM_DPPOPolicy(PPOsPolicy):
    def __init__(self):
        PPOsPolicy.__init__(self, "SpikLSTM_DPPO_DC")
        self.has_continuous_action_space = False
    def initialize_network(self, **kwargs):
        actor = DPPO_SpikLSTMActor(obs_size=4, oas_size=7, action_space=11,device=self.device,continuous_action_space=False)
        actor.to('cpu')
        checkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '/PPOs/checkpoints/DPPO_SpikLSTMActor_DC.pth'
        actor.load_state_dict(torch.load(checkpt_dir))
        actor.to(self.device)
        self.network=ActorCritic()
        self.network.actor=actor
        # print('load {} success'.format(checkpt_dir))

class DC_SpikGTr_DPPOPolicy(PPOsPolicy):
    def __init__(self):
        PPOsPolicy.__init__(self, "SpikGTr_DPPO_DC")
        self.has_continuous_action_space = False
    def initialize_network(self, **kwargs):
        actor = DPPO_SpikGTrActor(obs_size=4, oas_size=7, action_space=11,device=self.device,continuous_action_space=False)
        actor.to('cpu')
        checkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '/PPOs/checkpoints/DPPO_SpikGTrActor_DC.pth'
        actor.load_state_dict(torch.load(checkpt_dir))
        actor.to(self.device)
        self.network=ActorCritic()
        self.network.actor=actor
        # print('load {} success'.format(checkpt_dir))

class DC_SpikGTrAN_DPPOPolicy(PPOsPolicy):
    def __init__(self):
        PPOsPolicy.__init__(self, "SpikGTrAN_DPPO_DC")
        self.has_continuous_action_space = False
    def initialize_network(self, **kwargs):
        actor = DPPO_SpikGTrANActor(obs_size=4, oas_size=7, action_space=11,device=self.device,continuous_action_space=False)
        actor.to('cpu')
        checkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '/PPOs/checkpoints/DPPO_SpikGTrANActor_DC.pth'
        actor.load_state_dict(torch.load(checkpt_dir))
        actor.to(self.device)
        self.network=ActorCritic()
        self.network.actor=actor
        # print('load {} success'.format(checkpt_dir))

class DC_SpikMF_DPPOPolicy(PPOsPolicy):
    def __init__(self):
        PPOsPolicy.__init__(self, "SpikMF_DPPO_DC")
        self.has_continuous_action_space = False
    def initialize_network(self, **kwargs):
        actor = DPPO_SpikGMFActor(obs_size=4, oas_size=7, action_space=11,device=self.device,continuous_action_space=False)
        actor.to('cpu')
        checkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '/PPOs/checkpoints/DPPO_SpikGMFActor_DC.pth'
        actor.load_state_dict(torch.load(checkpt_dir))
        actor.to(self.device)
        self.network=ActorCritic()
        self.network.actor=actor
        # print('load {} success'.format(checkpt_dir))

# DC: discrete control
class DC_GTrMean_DPPOPolicy(PPOsPolicy):
    def __init__(self):
        PPOsPolicy.__init__(self, "GTrMean_DPPO_DC")
        self.has_continuous_action_space = False
    def initialize_network(self, **kwargs):
        actor = DPPO_GTrMeanActor(obs_size=4, oas_size=7, action_space=11,device=self.device,continuous_action_space=False)
        actor.to('cpu')
        checkpt_dir = os.path.dirname(os.path.realpath(__file__)) + '/PPOs/checkpoints/GTrMeanActor.pth'
        actor.load_state_dict(torch.load(checkpt_dir))
        actor.to(self.device)
        self.network=ActorCritic()
        self.network.actor=actor


if __name__ == '__main__':
    policy = DC_LSTM_DPPOPolicy()
    policy.initialize_network()

