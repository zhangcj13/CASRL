import random
import numpy as np
import torch
import torch.nn as nn
import os
from collections import deque
from gym_collision_avoidance.envs import Config

from ddpg.network.LSTMNet import LSTM_ActorNet,LSTM_CriticNet
from ddpg.network.SpikLSTMNet import SpikLSTM_ActorNet
from ddpg.network.GTrNet import GTr_ActorNet,GTr_CriticNet
from ddpg.network.SpikGTrNet import SpikGTr_ActorNet
from ddpg.network.SpikGTrNet import SpikGTr_ActorNet as GO_SpikGTr_ActorNet
from ddpg.network.LSTMNet import LSTM_ActorNet as XLSTM_ActorNet

from ddpg.network.SDDPGNet import SDDPG_ActorNet,DDPG_CriticNet

class Agent:
    def __init__(self,
                 state_num,
                 action_num,
                 rescale_state_num,
                 memory_size=1000,
                 batch_size=128,
                 target_tau=0.01,
                 target_update_steps=5,
                 reward_gamma=0.99,
                 actor_lr=0.0001,
                 critic_lr=0.0001,
                 epsilon_start=0.9,
                 epsilon_end=0.01,
                 epsilon_decay=0.999,
                 epsilon_rand_decay_start=60000,
                 epsilon_rand_decay_step=1,
                 use_cuda=True,
                 oas_num=7,
                 other_agents_num=2):
        self.state_num = state_num
        self.action_num = action_num
        self.rescale_state_num = rescale_state_num
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_tau = target_tau
        self.target_update_steps = target_update_steps
        self.reward_gamma = reward_gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_rand_decay_start = epsilon_rand_decay_start
        self.epsilon_rand_decay_step = epsilon_rand_decay_step
        self.use_cuda = use_cuda
        '''
        Random Action
        '''
        self.epsilon = epsilon_start
        '''
        Device
        '''
        if self.use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        """
        Memory
        """
        self.memory = deque(maxlen=self.memory_size)
        """
        Step Counter
        """
        self.step_ita = 0
        """
        Networks and Target Networks
        """
        if Config.ACT_ARCH not in Config.ALL_ACT_ARCHS or Config.CRT_ARCH not in Config.ALL_CRT_ARCHS:
            print('***** ARCH not exist *****')
            raise NotImplementedError

        self.actor_net = globals()[Config.ACT_ARCH](obs_size=state_num, oas_size=oas_num, action_space=action_num)
        self.critic_net = globals()[Config.CRT_ARCH](obs_size=state_num, oas_size=oas_num, action_space=action_num)

        self.target_actor_net = globals()[Config.ACT_ARCH](obs_size=state_num, oas_size=oas_num, action_space=action_num)
        self.target_critic_net = globals()[Config.CRT_ARCH](obs_size=state_num, oas_size=oas_num, action_space=action_num)

        self._hard_update(self.target_actor_net, self.actor_net)
        self._hard_update(self.target_critic_net, self.critic_net)
        self.actor_net.to(self.device)
        self.critic_net.to(self.device)
        self.target_actor_net.to(self.device)
        self.target_critic_net.to(self.device)
        """
        Criterion and optimizers
        """
        self.criterion = nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.critic_lr)
        self.oas_num = oas_num
        self.other_agents_num = other_agents_num

        # self.rstate_length = 2 + Config.HOST_AGENT_OBSERVATION_LENGTH + Config.OTHER_AGENT_OBSERVATION_LENGTH * Config.AGENTS_STATES_TO_SIM
        self.rstate_length = 2 + Config.HOST_AGENT_OBSERVATION_LENGTH + 36+1
        self.state_length = self.rstate_length

    def act(self, state, explore=True, train=True):
        """
        Generate Action based on state
        :param state: current state
        :param explore: if or not do random explore
        :param train: if or not in training
        :return: action
        """
        with torch.no_grad():
            actions = self.actor_net(state).to('cpu')
            actions = actions.numpy()
        out_actions=[]
        for action in actions:
            if train:
                if self.step_ita > self.epsilon_rand_decay_start and self.epsilon > self.epsilon_end:
                    if self.step_ita % self.epsilon_rand_decay_step == 0:
                        self.epsilon = self.epsilon * self.epsilon_decay
                noise = np.random.randn(self.action_num) * self.epsilon
                action = noise + (1 - self.epsilon) * action
                # action = np.clip(action, [0., -1.], [1., 1.])
                action = np.clip(action, [0., 0.], [1., 1.])
            elif explore:
                noise = np.random.randn(self.action_num) * self.epsilon_end
                action = noise + (1 - self.epsilon_end) * action
                # action = np.clip(action, [0., -1.], [1., 1.])
                action = np.clip(action, [0., 0.], [1., 1.])
            out_actions.append(action)
        # return action.tolist()
        return out_actions

    def remember(self, states, rescale_states, actions, rewards, next_states, rescale_next_states, dones,pre_dones=None):
        num = len(dones)
        if pre_dones is  None:
            for i in range(num):     
                state = states[i]
                rescale_state =  rescale_states[i]
                next_state = next_states[i]
                rescale_next_state = rescale_next_states[i]
                action=actions[i]
                reward=rewards[i]
                done=dones[i]
                self.memory.append((state, rescale_state, action, reward, next_state, rescale_next_state, done))
        else:
            for i in range(num):
                if pre_dones[i]:
                    continue
                state = states[i]
                rescale_state =  rescale_states[i]
                next_state = next_states[i]
                rescale_next_state = rescale_next_states[i]
                action=actions[i]
                reward=rewards[i]
                done=dones[i]
                self.memory.append((state, rescale_state, action, reward, next_state, rescale_next_state, done))

    def remember_single(self, state, rescale_state, action, reward, next_state, rescale_next_state, done,pre_done=None):

        self.memory.append((state, rescale_state, action, reward, next_state, rescale_next_state, done))

    def replay(self):
        """
        Experience Replay Training
        :return: actor_loss_item, critic_loss_item
        """
        state_batch, r_state_batch, action_batch, reward_batch, nstate_batch, r_nstate_batch, done_batch = self._random_minibatch()
        '''
        Compuate Target Q Value
        '''
        with torch.no_grad():
            naction_batch = self.target_actor_net(r_nstate_batch)
            next_q = self.target_critic_net([r_nstate_batch, naction_batch])
            target_q = reward_batch + self.reward_gamma * next_q * (1. - done_batch)
        '''
        Update Critic Network
        '''
        self.critic_optimizer.zero_grad()
        current_q = self.critic_net([r_state_batch, action_batch])
        critic_loss = self.criterion(current_q, target_q)
        critic_loss_item = critic_loss.item()
        critic_loss.backward()
        self.critic_optimizer.step()
        '''
        Update Actor Network
        '''
        self.actor_optimizer.zero_grad()
        current_action = self.actor_net(r_state_batch)
        actor_loss = -self.critic_net([r_state_batch, current_action])
        actor_loss = actor_loss.mean()
        actor_loss_item = actor_loss.item()
        actor_loss.backward()
        self.actor_optimizer.step()
        '''
        Update Target Networks
        '''
        self.step_ita += 1
        if self.step_ita % self.target_update_steps == 0:
            self._soft_update(self.target_actor_net, self.actor_net)
            self._soft_update(self.target_critic_net, self.critic_net)
        return actor_loss_item, critic_loss_item

    def save(self, save_dir, episode):
        """
        Save Actor Net weights
        :param save_dir: directory for saving weights
        :param episode: number of episode
        :param run_name: name of the run
        """
        try:
            os.mkdir(save_dir)
            print("Directory ", save_dir, " Created")
        except FileExistsError:
            print("Directory", save_dir, " already exists")
        torch.save(self.actor_net.state_dict(),
                   save_dir + '/' + Config.ACT_ARCH + '_s' + str(episode) + '.pth')
        torch.save(self.critic_net.state_dict(),
                   save_dir + '/' + Config.CRT_ARCH + '_s' + str(episode) + '.pth')
        print("Episode " + str(episode) + " weights saved ...")

    def load(self, actor_file,critic_file=None):
        """
        Load Actor Net weights
        :param load_file_name: weights file name
        """
        self.actor_net.to('cpu')
        self.actor_net.load_state_dict(torch.load(actor_file))
        self.actor_net.to(self.device)

        if critic_file is not None:
            self.critic_net.to('cpu')
            self.critic_net.load_state_dict(torch.load(critic_file))
            self.critic_net.to(self.device)
        print('load {} success'.format(actor_file))

    def _random_minibatch(self):
        """
        Random select mini-batch from memory
        :return: state_batch, action_batch, reward_batch, nstate_batch, done_batch
        """

        minibatch = random.sample(self.memory, self.batch_size)

        state_batch = np.zeros((self.batch_size, self.state_length))
        rescale_state_batch = np.zeros((self.batch_size, self.rstate_length))

        action_batch = np.zeros((self.batch_size, self.action_num))
        reward_batch = np.zeros((self.batch_size, 1))

        nstate_batch = np.zeros((self.batch_size, self.state_length))
        rescale_nstate_batch = np.zeros((self.batch_size, self.rstate_length))

        done_batch = np.zeros((self.batch_size, 1))
        for num in range(self.batch_size):
            state_batch[num, :] = np.array(minibatch[num][0])
            rescale_state_batch[num, :] = np.array(minibatch[num][1])

            action_batch[num, :] = np.array(minibatch[num][2])
            reward_batch[num, 0] = minibatch[num][3]
            nstate_batch[num, :] = np.array(minibatch[num][4])

            rescale_nstate_batch[num, :] = np.array(minibatch[num][5])
            done_batch[num, 0] = minibatch[num][6]

        # state_batch=torch.Tensor(state_batch).to(self.device)
        # rescale_state_batch = torch.Tensor(rescale_state_batch).to(self.device)

        action_batch = torch.Tensor(action_batch).to(self.device)
        reward_batch = torch.Tensor(reward_batch).to(self.device)

        # nstate_batch = torch.Tensor(nstate_batch).to(self.device)
        # rescale_nstate_batch = torch.Tensor(rescale_nstate_batch).to(self.device)

        done_batch = torch.Tensor(done_batch).to(self.device)

        return state_batch, rescale_state_batch, action_batch, reward_batch, nstate_batch, rescale_nstate_batch, done_batch

    def _hard_update(self, target, source):
        """
        Hard Update Weights from source network to target network
        :param target: target network
        :param source: source network
        """
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)

    def _soft_update(self, target, source):
        """
        Soft Update weights from source network to target network
        :param target: target network
        :param source: source network
        """
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.target_tau) + param.data * self.target_tau
                )

    def action_decoder(self, actions):

        # decode_action = np.clip(actions, a_min=[0, -1], a_max=[1, 1])

        lsmax = 1.0
        lsmin = 0.05
        decode_action=[]
        for action in actions:
            lspd = action[0] * (lsmax - lsmin) + lsmin
            rspd = action[1] * (lsmax - lsmin) + lsmin
            linear=(lspd+rspd)/2
            angular=(rspd-lspd)/0.23
            decode_action.append([linear,angular])
        return decode_action

    def reset_epsilon(self, new_epsilon, new_decay):
        """
        Set Epsilon to a new value
        :param new_epsilon: new epsilon value
        :param new_decay: new epsilon decay
        """
        self.epsilon = new_epsilon
        self.epsilon_decay = new_decay

    def reset_status(self):
        self.actor_net.reset_mean_std()
        self.target_actor_net.reset_mean_std()
        self.critic_net.reset_mean_std()
        self.target_critic_net.reset_mean_std()

        # self.rstate_length = 2 + Config.HOST_AGENT_OBSERVATION_LENGTH + Config.OTHER_AGENT_OBSERVATION_LENGTH * Config.AGENTS_STATES_TO_SIM
        # self.state_length = self.rstate_length
        self.memory.clear()



