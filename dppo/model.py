import os

#os.environ['GYM_CONFIG_CLASS'] = 'TrainPhase_PPOS'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.multiprocessing as mp
import math
from torch.distributions import Categorical

from dppo.network.LSTMPolicy import LSTMActor, LSTMCritic
from dppo.network.SpikLSTMPolicy import SpikLSTMActor
from dppo.network.GTrPolicy import GTrActor,GTrCritic
from dppo.network.SpikGTrPolicy import SpikGTrActor
# from dppo.network.SpikGTrMFPolicy import SpikGTrMFActor
from dppo.network.GTrS2SPolicy import GTrS2SActor,GTrS2SCritic
from dppo.network.GTrXLPolicy import GTrXLActor,GTrXLCritic
from dppo.network.SpikTrANPolicy import SpikTrANActor

from dppo.network.SpikGTrANPolicy import SpikGTrANActor
from dppo.network.SpikMFPolicy import SpikGMFActor
from dppo.network.SpikGTrMeanPolicy import SpikGTrMeanActor
from dppo.network.GTrMeanPolicy import GTrMeanActor,GTrMeanCritic


from gym_collision_avoidance.envs import Config

class Actions():
    def __init__(self):
        self.actions = np.mgrid[1.0:1.1:0.5, -np.pi/6:np.pi/6+0.01:np.pi/12].reshape(2, -1).T
        self.actions = np.vstack([self.actions,np.mgrid[0.5:0.6:0.5, -np.pi/6:np.pi/6+0.01:np.pi/6].reshape(2, -1).T])
        self.actions = np.vstack([self.actions,np.mgrid[0.0:0.1:0.5, -np.pi/6:np.pi/6+0.01:np.pi/6].reshape(2, -1).T])
        self.num_actions = len(self.actions)


def log_normal_density(x, mean, log_std, std):
    """returns guassian density given x on log scale"""
    variance = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * variance) - 0.5 * np.log(2 * np.pi) - log_std  # num_env * frames * act_size
    log_density = log_density.sum(dim=-1, keepdim=True)  # num_env * frames * 1
    return log_density


class Model(nn.Module):
    def __init__(self, obs_size=4, oas_size=7, action_dim=2, device=None,continuous_action_space=True):
        super(Model, self).__init__()

        # self.action_space = action_space
        self.logstd = nn.Parameter(torch.zeros(action_dim))

        self.actor = globals()[Config.ACT_ARCH](obs_size=obs_size, oas_size=oas_size, action_space=action_dim,
                                                device=device,continuous_action_space=continuous_action_space)
        self.continuous_action_space = continuous_action_space
        self.critic = globals()[Config.CRT_ARCH](obs_size=obs_size, oas_size=oas_size, action_space=action_dim,
                                                 device=device)

        self.device = self.actor.device
        if Config.NORMALIZE_INPUT:
            self.avg_vec = np.array(Config.NN_INPUT_AVG_VECTOR, dtype=np.float32)
            self.std_vec = np.array(Config.NN_INPUT_STD_VECTOR, dtype=np.float32)
        self.train()

    def forward(self, inputs,input_grad=True):
        state = self.rescale_input(inputs, seq_cut=True,input_grad=input_grad)
        # actor
        if self.continuous_action_space:
            mean = self.actor(state)
            logstd = self.logstd.expand_as(mean)
            std = torch.exp(logstd)
            action = torch.normal(mean, std)
            logprob = log_normal_density(action, mean, std=std, log_std=logstd)
            # critic
            v = self.critic(state)
            return v, action, logprob, mean
        else:
            # prob = self.actor(state)
            # dist = Categorical(prob)
            # action = dist.sample()
            # # logprob = dist.log_prob(action)
            # # action_
            # # critic
            # v = self.critic(state)
            # return v, action, prob
            probs = self.actor(state)
            dist = Categorical(probs)
            action = dist.sample()
            a_logprob = dist.log_prob(action)
            v = self.critic(state)
            return v,action,a_logprob,probs
    
    def generate_action(self, state, action_bound= [[-1, -1], [1, 1]], test=False,multi_actions_flag=False):

        if self.continuous_action_space:
            v, a, logprob, mean = self.forward(state)
            v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
            scaled_action = np.clip(a, a_min=action_bound[0], a_max=action_bound[1])
            if test:
                mean = mean.data.cpu().numpy()
                scaled_action = np.clip(mean, a_min=action_bound[0], a_max=action_bound[1])
            return v, a, logprob, scaled_action
        else:
            # v, action, prob = self.forward(state)
            # action_id = action.item()
            # action_prob = prob[..., action_id].detach().cpu().numpy()
            # v, action = v.data.cpu().numpy(), action.data.cpu().numpy()
            # if test:
            #     action_id =  torch.argmax(prob).item()
            # return v, action, action_prob, action_id

            v, a, logprob, probs = self.forward(state)
            if not multi_actions_flag:
                action_id = a.item()
                v, action ,logprob = v.data.cpu().numpy(), a.data.cpu().numpy(),logprob.data.cpu().numpy()
                if test:
                    action_id = torch.argmax(probs).item()
                return v, action, logprob, action_id
            else:
                action_id_list=[]
                for ia in a:
                    action_id_list.append(ia.item())
                # action_id = a.item()
                v, action, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
                if test:
                    action_id_list = torch.argmax(probs,-1).cpu().numpy()
                return v, action, logprob, action_id_list

    def evaluate_actions(self, x, action):
        if self.continuous_action_space:
            v, _, _, mean = self.forward(x, input_grad=False)
            logstd = self.logstd.expand_as(mean)
            std = torch.exp(logstd)
            logprob = log_normal_density(action, mean, log_std=logstd, std=std)
            dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
            dist_entropy = dist_entropy.sum(-1).mean()
            return v, logprob, dist_entropy
        else:
            # v, _, prob = self.forward(x, input_grad=False)
            # dist_entropy = Categorical(prob).entropy()
            # dist_entropy=dist_entropy.sum(0, keepdim=True)
            #
            # action = action.to(torch.int64).unsqueeze(-1)
            # action_prob = prob.gather(-1, action).squeeze(-1)
            # # dist_entropy = Categorical(action_prob).entropy()

            v, _, _, probs = self.forward(x, input_grad=False)
            dist = Categorical(probs)

            dist_entropy = dist.entropy().view(-1, 1)
            dist_entropy = dist_entropy.sum(-1).mean()
            a_logprob = dist.log_prob(action).view(-1, 1)

            return v, a_logprob, dist_entropy

        # logstd = self.logstd.expand_as(mean)
        # std = torch.exp(logstd)
        # evaluate
        # logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        # dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        # dist_entropy = dist_entropy.sum(-1).mean()


    def rescale_input(self, x, return_vec=False, batch_first=False, seq_cut=False,input_grad=True):
        if Config.NORMALIZE_INPUT:
            x_normalized = (x[:, 1:] - self.avg_vec) / self.std_vec
        else:
            x_normalized = x[:, 1:]

        host_agent_vec = x_normalized[:,
                         Config.FIRST_STATE_INDEX:Config.HOST_AGENT_STATE_SIZE + Config.FIRST_STATE_INDEX:]
        if input_grad:
            host_agent_vec = torch.Tensor(host_agent_vec).to(self.device)
        else:
            host_agent_vec=Variable(torch.from_numpy(host_agent_vec)).float().to(self.device)

        num_other_agents = np.clip(x_normalized[:, 0] + 1, 0, 1e6)
        other_agent_vec = x_normalized[:, Config.HOST_AGENT_STATE_SIZE + Config.FIRST_STATE_INDEX:]

        if input_grad:
            other_agent_vec = torch.Tensor(other_agent_vec).to(self.device)
        else:
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
    
    def save(self, path, ep):

        actor_pth = path + '/{}_{}.pth'.format(Config.ACT_ARCH, ep)
        critic_pth = path + '/{}_{}.pth'.format(Config.CRT_ARCH, ep)

        torch.save(self.actor.state_dict(), actor_pth)
        torch.save(self.critic.state_dict(), critic_pth)

        print('########################## network saved when update {} times#########################'.format(ep))
        print('actor model save to {} '.format(actor_pth))
        print('critic model save to {} '.format(critic_pth))
        print('########################## network saved when update {} times#########################'.format(ep))
    
    def load(self, path, ep):
        actor_pth = path + '/{}_{}.pth'.format(Config.ACT_ARCH, ep)
        critic_pth = path + '/{}_{}.pth'.format(Config.CRT_ARCH, ep)

        if not os.path.exists(actor_pth) or not os.path.exists(critic_pth):
            print('{} and {} not exist'.format(actor_pth,critic_pth))
            print('####################### load pre-trained model failed #######################')
            return False

        a_state_dict = torch.load(actor_pth)
        self.actor.load_state_dict(a_state_dict)
        c_state_dict = torch.load(critic_pth)
        self.critic.load_state_dict(c_state_dict)

        print('###################################################')
        print('actor model load from {} '.format(actor_pth))
        print('critic model load from {} '.format(critic_pth))
        print('###################################################')
        return True


    def spike_analysis(self, inputs,):
        state = self.rescale_input(inputs, seq_cut=True, input_grad=True)
        # actor
        spikes = self.actor.get_spike(state)

        return spikes


class Shared_grad_buffers():
    def __init__(self, model):
        self.grads = {}
        for name, p in model.named_parameters():
            self.grads[name + '_grad'] = torch.ones(p.size()).share_memory_()

    def add_gradient(self, model):
        for name, p in model.named_parameters():
            self.grads[name + '_grad'] += p.grad.data

    def reset(self):
        for name, grad in self.grads.items():
            self.grads[name].fill_(0)


class Shared_obs_stats():
    def __init__(self, num_inputs):
        self.n = torch.zeros(num_inputs).share_memory_()
        self.mean = torch.zeros(num_inputs).share_memory_()
        self.mean_diff = torch.zeros(num_inputs).share_memory_()
        self.var = torch.zeros(num_inputs).share_memory_()

    def observes(self, obs):
        # observation mean var updates
        x = obs.data.squeeze()
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = torch.clamp(self.mean_diff / self.n, min=1e-2)

    def normalize(self, inputs):
        obs_mean = Variable(self.mean.unsqueeze(0).expand_as(inputs))
        obs_std = Variable(torch.sqrt(self.var).unsqueeze(0).expand_as(inputs))
        return torch.clamp((inputs - obs_mean) / obs_std, -5., 5.)


if __name__ == '__main__':
    model = Model(continuous_action_space=False).cuda()

    t = np.random.rand(2, 1 + 26 + 7 * 0)
    t[0][1] = 2
    t[1][1] = 3
    o = model(t)

    # o=GPTConfig()

    print(o)


