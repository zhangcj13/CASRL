import sys

import os
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import time

import warnings
warnings.filterwarnings("ignore")

os.environ['GYM_CONFIG_CLASS'] = 'FullEvalTest'
from gym_collision_avoidance.envs import Config

import gym_collision_avoidance.envs.test_cases as tc
from Eval_test.env_utils import store_stats,run_episode

from gym_collision_avoidance.envs.wrappers import FlattenDictWrapper, MultiagentFlattenDictWrapper, MultiagentDummyVecEnv, MultiagentDictToMultiagentArrayWrapper
# from gym_world1 import gym_world
from gym_collision_avoidance.envs.agent import Agent
from gym_world import gym_world

from gym_collision_avoidance.envs.dynamics.UnicycleDynamics import UnicycleDynamics

from gym_collision_avoidance.envs.sensors.LaserScanSensor import LaserScanSensor
from gym_collision_avoidance.envs.sensors.OtherAgentsStatesSensor import OtherAgentsStatesSensor
sensor_dict = {
    'other_agents_states': OtherAgentsStatesSensor,
    'laserscan': LaserScanSensor,
    # 'other_agents_states_encoded': OtherAgentsStatesSensorEncode,
}

# Policies
from gym_collision_avoidance.envs.policies.StaticPolicy import StaticPolicy
from gym_collision_avoidance.envs.policies.NonCooperativePolicy import NonCooperativePolicy
from gym_collision_avoidance.envs.policies.DRLLongPolicy import DRLLongPolicy
try:
    from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
except:
    from gym_collision_avoidance.envs.policies.StaticPolicy import StaticPolicy as RVOPolicy
    print("RVOPolicy not installed...")
from gym_collision_avoidance.envs.policies.CADRLPolicy import CADRLPolicy
from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy
from gym_collision_avoidance.envs.policies.ExternalPolicy import ExternalPolicy
from gym_collision_avoidance.envs.policies.LearningPolicy import LearningPolicy
from gym_collision_avoidance.envs.policies.CARRLPolicy import CARRLPolicy
from gym_collision_avoidance.envs.policies.LearningPolicyGA3C import LearningPolicyGA3C
from gym_collision_avoidance.envs.policies.LearningPolicyCNNOASS import LearningPolicyCNNOASS
from gym_collision_avoidance.envs.policies.DDPGPolicy import LSTM_DDPGPolicy,SpikLSTM_DDPGPolicy,SpikGTr_DDPGPolicy,\
    SpikDDPGPolicy
from gym_collision_avoidance.envs.policies.GA3CPolicy import LSTM_GA3CPolicy,SpikLSTM_GA3CPolicy
# from gym_collision_avoidance.envs.policies.PPOCADRLPolicy import PPOCADRLPolicy
from gym_collision_avoidance.envs.policies.PPOsPolicy import LSTM_PPOsPolicy,LSTM_DPPOPolicy,DC_LSTM_DPPOPolicy,\
    DC_SpikLSTM_DPPOPolicy,DC_SpikGTr_DPPOPolicy,DC_SpikGTrAN_DPPOPolicy,DC_SpikMF_DPPOPolicy,DC_GTrMean_DPPOPolicy


policy_dict = {
    'RVO': RVOPolicy,
    'noncoop': NonCooperativePolicy,
    'carrl': CARRLPolicy,
    'external': ExternalPolicy,
    'GA3C_CADRL': GA3CCADRLPolicy,
    # 'PPO_CADRL':PPOCADRLPolicy,
    'static': StaticPolicy,
    'CADRL': CADRLPolicy,
    'DRL' : DRLLongPolicy,
    'CNNOASS':LearningPolicyCNNOASS,
    'LSTM_DDPG':LSTM_DDPGPolicy,
    'SpikLSTM_DDPG':SpikLSTM_DDPGPolicy,
    'SpikGTr_DDPG':SpikGTr_DDPGPolicy,
    'LSTM_GA3C':LSTM_GA3CPolicy,
    'SpikLSTM_GA3C':SpikLSTM_GA3CPolicy,
    'LSTM_PPOs':LSTM_PPOsPolicy,
    'LSTM_DPPO':LSTM_DPPOPolicy,
    'LSTM_DPPO_DC':DC_LSTM_DPPOPolicy,
    'SpikLSTM_DPPO_DC':DC_SpikLSTM_DPPOPolicy,
    'SpikGTr_DPPO_DC':DC_SpikGTr_DPPOPolicy,
    'SpikGTrAN_DPPO_DC':DC_SpikGTrAN_DPPOPolicy,
    'SpikMF_DPPO_DC':DC_SpikMF_DPPOPolicy,
    'GTrMean_DPPO_DC':DC_GTrMean_DPPOPolicy,
    'SpikDDPG':SpikDDPGPolicy,
}

sense_type= 'other_agents_states' # 'other_agents_states' laserscan

def reset_env(env, policy, test_case, using_npy_cases=True):
    agents=[]
    if using_npy_cases:
        for id, c in enumerate(test_case):
            agent = Agent(c[0], c[1], c[2], c[3], c[5], c[4], c[6], policy_dict[policy], UnicycleDynamics,
                            [sensor_dict[sense_type]], id)
            agents.append(agent)
    else:
        for id, c in enumerate(test_case):
            px = c[0]
            py = c[1]
            gx = c[2]
            gy = c[3]
            pref_speed = c[4]
            radius = c[5]

            vec_to_goal = np.array([gx, gy]) - np.array([px, py])
            heading = np.arctan2(vec_to_goal[1], vec_to_goal[0])

            agent = Agent(px, py, gx, gy, radius, pref_speed, heading, policy_dict[policy], UnicycleDynamics,
                            [sensor_dict[sense_type]], id)
            agents.append(agent)
    
    [agent.policy.initialize_network() for agent in agents if hasattr(agent.policy, 'initialize_network')]
    
    env.set_agents(agents)
    init_obs = env.reset()
    return init_obs

def create_env(max_num_agents=8,policy='static',test_case=None,plot_dir=None):
    if test_case is None:
        print('test_cases could not be None!')
        return

    env = gym_world()
    env = MultiagentDictToMultiagentArrayWrapper(env, dict_keys=Config.STATES_IN_OBS, max_num_agents=max_num_agents)
    if plot_dir is not None:
        env.set_plot_save_dir(plot_dir)

    agents = []
    for id, c in enumerate(test_case):
        px = c[0]
        py = c[1]
        gx = c[2]
        gy = c[3]
        pref_speed = c[4]
        radius = c[5]

        vec_to_goal = np.array([gx, gy]) - np.array([px, py])
        heading = np.arctan2(vec_to_goal[1], vec_to_goal[0])

        agent = Agent(px, py, gx, gy, radius, pref_speed, heading, policy_dict[policy], UnicycleDynamics,
                      [sensor_dict[sense_type]], id)
        agents.append(agent)

    [agent.policy.initialize_network() for agent in agents if hasattr(agent.policy, 'initialize_network')]

    env.set_agents(agents)
    init_obs = env.reset()
    return env


# test the ordinal cases's result
def main_cadrl(test_failed_flag=False,failed_ids=[]):
    np.random.seed(0)
    print("Running {test_cases} test cases for {num_agents} for policies: {policies}".format(
        test_cases=Config.NUM_TEST_CASES,
        num_agents=Config.NUM_AGENTS_TO_TEST,
        policies=Config.POLICIES_TO_TEST,
    ))

    with tqdm(total=Config.NUM_TEST_CASES * len(Config.POLICIES_TO_TEST) * len(Config.NUM_AGENTS_TO_TEST)) as pbar:
        for policy in Config.POLICIES_TO_TEST:
            for num_agent in Config.NUM_AGENTS_TO_TEST:
                df = None

                Config.set_agents_num(num_agent, policy)

                print(num_agent, Config.NN_INPUT_SIZE)

                # file = "test_cases_{}_{}.npy".format(Config.NUM_TEST_CASES, num_agent)
                file = './gym_collision_avoidance/envs/test_cases/{}_agents_{}_cases.p'.format(num_agent,Config.NUM_TEST_CASES)
                if not os.path.exists(file):
                    print('there are no file called: ',file)
                    continue
                # agents_tcs = np.load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')

                agents_tcs = pd.read_pickle(file)

                env = gym_world()
                env = MultiagentDictToMultiagentArrayWrapper(env, dict_keys=Config.STATES_IN_OBS,
                                                             max_num_agents=num_agent)
                env.set_plot_save_dir(
                    os.path.dirname(os.path.realpath(__file__)) + '/eval_output/results/{}/'.format(policy))

                df = pd.DataFrame()
                tc_cnt = 0
                fail_cnt=0
                for nc,test_case in enumerate(agents_tcs):

                    if test_failed_flag:
                        if nc not in failed_ids:
                            continue

                    # ##### Actually run the episode ##########
                    _ = reset_env(env, policy, test_case, using_npy_cases=False )
                    episode_stats, prev_agents = run_episode(env)
                    df = store_stats(df, {'test_case': Config.NUM_TEST_CASES, 'policy_id': policy}, episode_stats)
                    # ########################################
                    pbar.update(1)
                    tc_cnt += 1
                    # if not episode_stats['all_at_goal']:
                    #     fail_cnt+=1
                    #     print('id:{}, fail_cnt:{}, fail rate:{}'.format(nc,fail_cnt,fail_cnt/(tc_cnt+1e-6)))
                    if tc_cnt >= Config.NUM_TEST_CASES:
                        break


                if Config.RECORD_PICKLE_FILES:
                    file_dir = os.path.dirname(os.path.realpath(__file__)) + '/eval_output/stats_cadrl/{}_agents/'.format(
                        num_agent)
                    os.makedirs(file_dir, exist_ok=True)
                    log_filename = file_dir + '/stats_{}.p'.format(policy)
                    df.to_pickle(log_filename)


if __name__ == '__main__':

    test_failed_flag = False
    failed_ids=[]

    main_cadrl(test_failed_flag=test_failed_flag,failed_ids=failed_ids)
    print("Experiment over.")
