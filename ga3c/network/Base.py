import os
import re
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import math

from gym_collision_avoidance.envs import Config
# net arch
from ga3c.network.LSTMNet import LSTMNet
from ga3c.network.SpikLSTMNet import SpikLSTMNet
from ga3c.network.SpikGTrNet import SpikGTrNet


class SharedAdam(optim.Adam):
    """Implements Adam algorithm with shared states.
    """
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        # self.update_time=0

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        # self.update_time+=1
        return loss

class TorchBase:
    def __init__(self, device, model_name, num_actions):
        super(TorchBase, self).__init__()

        if Config.TRAIN_VERSION in [Config.LOAD_RL_THEN_TRAIN_RL, Config.LOAD_REGRESSION_THEN_TRAIN_RL]:
            learning_method = 'RL'
        elif Config.TRAIN_VERSION in [Config.TRAIN_ONLY_REGRESSION]:
            learning_method = 'regression'
        self.wandb_dir = os.path.dirname(os.path.realpath(__file__)) + '/checkpoints/' + learning_method

        # if training, add run to GA3C-CADRL project, add hyperparams and auto-upload checkpts
        if not Config.PLAY_MODE and not Config.EVALUATE_MODE and Config.USE_WANDB:
            raise NotImplementedError
        else:
            self.checkpoints_save_dir = 'weight/ga3c/'

        self.device = torch.device(Config.DEVICE)
        print("base device:{}".format(device))
        self.model_name = model_name
        self.num_actions = num_actions
        self.learning_rate_rl = Config.LEARNING_RATE_RL_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON

        self._create_network()

        if Config.TENSORBOARD: self._create_tensor_board()

    def _create_network(self,optimizer=None):
        if Config.NORMALIZE_INPUT:
            self.avg_vec = np.array(Config.NN_INPUT_AVG_VECTOR, dtype = np.float32)
            self.std_vec = np.array(Config.NN_INPUT_STD_VECTOR, dtype = np.float32)
        else:
            self.avg_vec = 0
            self.std_vec = 1

        self.shared_model = globals()[Config.NETWORK_NAME]().to(self.device)
        # self.shared_model = LSTMNet().to(self.device)
        self.shared_model.share_memory()
        self.local_models={}
        self.infer_models = {}

        if optimizer is None:
            self.optimizer = SharedAdam(self.shared_model.parameters(), lr=Config.LEARNING_RATE_RL_START, betas=(0.92, 0.999))
            self.optimizer.share_memory()
        else:
            self.optimizer = optimizer

    def _checkpoint_filename(self, episode, mode='save', learning_method='RL', wandb_runid_for_loading=None):
        if mode == 'save':
            d = self.checkpoints_save_dir
        elif mode == 'load':
            d = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoints", learning_method, 'wandb',
                             wandb_runid_for_loading, 'checkpoints')
        else:
            raise NotImplementedError
        
        path = os.path.join(d, '%s_%08d.pth' % (self.model_name, episode))
        return path

    def save(self, episode, learning_method='RL'):
        path=self._checkpoint_filename(episode, learning_method=learning_method, mode='save')
        torch.save(self.shared_model.state_dict(), path)
        print("Episode " + str(episode) + path+ " weights saved ...")

    def load(self, learning_method='RL'):
        weight_pth = "./weight/ga3c/{}_{}.pth".format(Config.NETWORK_NAME,Config.LOAD_EPISODE)

        if os.path.exists(weight_pth):
            self.shared_model.to('cpu')
            self.shared_model.load_state_dict(torch.load(weight_pth))
            self.shared_model.to(self.device)
            print('load {} success'.format(weight_pth))
            return int(Config.LOAD_EPISODE)
        
        return 0

    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|_|\.', filename)[-1])


    def _create_tensor_board(self):
        from torch.utils.tensorboard import SummaryWriter
        self.tb_writer = SummaryWriter()

    def __get_base_feed_dict(self):
        # return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate_rl}
        return

    def predict_p_and_v(self, x,pre_id=None):
        x = self.normalize_input(x)
        if pre_id not in self.infer_models:
            print("trainer_id:{}".format(pre_id))
            torch.manual_seed(1 + pre_id)
            self.infer_models[pre_id] = globals()[Config.NETWORK_NAME]().to(self.device)
            self.infer_models[pre_id].train()

        self.infer_models[pre_id].load_state_dict(self.shared_model.state_dict())
        with torch.no_grad():
            logits_p, logits_v = self.infer_models[pre_id](x)
        softmax_p=(F.softmax(logits_p, dim=-1)+Config.MIN_POLICY)/(1.0 + Config.MIN_POLICY * self.num_actions)
        np_softmax_p = softmax_p.detach().cpu().numpy()
        np_logits_v = logits_v.detach().cpu().numpy()
        return np_softmax_p,np_logits_v

    def train(self, x, y_r, a, trainer_id, learning_method='RL'):
        x = self.normalize_input(x)
        if learning_method == 'RL':
            max_grad_norm=50
            if trainer_id not in self.local_models:
                torch.manual_seed(1 + trainer_id)
                # self.local_models[trainer_id]=SNNNET().to(self.device)
                self.local_models[trainer_id] = globals()[Config.NETWORK_NAME]().to(self.device)
            self.local_models[trainer_id].train()
            self.local_models[trainer_id].load_state_dict(self.shared_model.state_dict())

            # with torch.autograd.set_detect_anomaly(True):
            loss = self.local_models[trainer_id].evaluate_actions(x, y_r, a)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(self.local_models[trainer_id].parameters(), max_grad_norm)
            self.ensure_shared_grads(self.local_models[trainer_id], self.shared_model)
            self.optimizer.step()
        elif learning_method == 'regression':
            raise NotImplementedError ('train with regression not implement yet')
        # return

    def log(self, x, y_r, a, reward, roll_reward, episode):

        self.tb_writer.add_scalar('reward', reward, episode)
        self.tb_writer.add_scalar('roll_reward', roll_reward, episode)
        return

    def ensure_shared_grads(self,model, shared_model):
        for param, shared_param in zip(model.parameters(),
                                       shared_model.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def normalize_input(self,x):
        x_normalized=(x - self.avg_vec) / self.std_vec
        return x_normalized



