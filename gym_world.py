from gym_collision_avoidance.envs.collision_avoidance_env import CollisionAvoidanceEnv
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.visualize import plot_episode, animate_episode

import logging
from gym.envs.registration import register
import numpy as np
import math
import copy
logger = logging.getLogger(__name__)

from utils import creat_training_agents,gen_polygon_exterior_list,gen_goal_position_list
from gym_collision_avoidance.envs import util
import random


def to_polar(vector):
    x, y = vector[0], vector[1]
    r = math.sqrt(x ** 2 + y ** 2)
    angle = math.atan2(y,x)
    return r, angle

class gym_world(CollisionAvoidanceEnv):
    def __init__(self):
        super(gym_world, self).__init__()

        self.dis2goal_pre = np.zeros(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT)
        self.dis2goal_cur = np.zeros(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT)

        self.train_cases_cnt = {}
        if Config.USING_EXTERNAL_TRAIN_CASES:
            self.train_cases={}
            file = 'train_cases/train_cases_{}.npy'
            agents_num=[2,4,6,10]
            for n in agents_num:
                tcs = np.load(file.format(n), mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
                self.train_cases[n]=tcs
                self.train_cases_cnt[n]=0

    def step(self, actions, dt=None,step=None):
        if dt is None:
            dt = self.dt_nominal

        self.episode_step_number += 1

        # Take action
        self._take_action(actions, dt)

        # Collect rewards
        rewards = self._compute_rewards()

        # Take observation
        next_observations = self._get_obs()

        if Config.ANIMATE_EPISODES and self.episode_step_number % self.animation_period_steps == 0:
            plot_episode(self.agents, False, self.map, self.test_case_index,
                circles_along_traj=Config.PLOT_CIRCLES_ALONG_TRAJ,
                plot_save_dir=self.plot_save_dir,
                plot_policy_name=self.plot_policy_name,
                save_for_animation=True,
                limits=self.plt_limits,
                fig_size=self.plt_fig_size,
                perturbed_obs=self.perturbed_obs,
                show=True,#False,
                save=True)

        # Check which agents' games are finished (at goal/collided/out of time)
        which_agents_done, game_over = self._check_which_agents_done()

        which_agents_done_dict = {}
        which_agents_learning_dict = {}
        for i, agent in enumerate(self.agents):
            which_agents_done_dict[agent.id] = which_agents_done[i]
            which_agents_learning_dict[agent.id] = agent.policy.is_still_learning

        if Config.AGENTS_TO_Train!=Config.MAX_NUM_AGENTS_TO_SIM:
            rewards=rewards[0:Config.AGENTS_TO_Train]

        return next_observations, rewards, game_over, \
            {
                'which_agents_done': which_agents_done_dict,
                'which_agents_learning': which_agents_learning_dict,
            }

    def reset(self):
        """ Resets the environment, re-initializes agents, plots episode (if applicable) and returns an initial observation.

        Returns:
            initial observation (np array): each agent's observation given the initial configuration
        """
        # print("########################### rest env ###########################")
        self.dis2goal_pre = np.zeros(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT)
        self.dis2goal_cur = np.zeros(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT)

        if self.episode_step_number is not None and self.episode_step_number > 0 and self.plot_episodes and self.test_case_index >= 0:
            plot_episode(self.agents, self.evaluate, self.map, self.test_case_index, self.id,
                         circles_along_traj=Config.PLOT_CIRCLES_ALONG_TRAJ, plot_save_dir=self.plot_save_dir,
                         plot_policy_name=self.plot_policy_name, limits=self.plt_limits, fig_size=self.plt_fig_size,
                         show=Config.SHOW_EPISODE_PLOTS, save=Config.SAVE_EPISODE_PLOTS)
            if Config.ANIMATE_EPISODES:
                animate_episode(num_agents=len(self.agents), plot_save_dir=self.plot_save_dir, plot_policy_name=self.plot_policy_name, test_case_index=self.test_case_index, agents=self.agents)
            self.episode_number += 1
        self.begin_episode = True
        self.episode_step_number = 0
        self._init_agents()
        if Config.USE_STATIC_MAP:
            self._init_static_map()
        for state in Config.STATES_IN_OBS:
            for agent in range(Config.MAX_NUM_AGENTS_IN_ENVIRONMENT):
                self.observation[agent][state] = np.zeros((Config.STATE_INFO_DICT[state]['size']), dtype=Config.STATE_INFO_DICT[state]['dtype'])
        return self._get_obs()

    def _compute_rewards(self):
        """ Check for collisions and reaching of the goal here, and also assign the corresponding rewards based on those calculations.

        Returns:
            rewards (scalar or list): is a scalar if we are only training on a single agent, or
                      is a list of scalars if we are training on mult agents
        """

        # if nothing noteworthy happened in that timestep, reward = -0.01
        rewards = np.zeros(len(self.agents))
        collision_with_agent, collision_with_wall, entered_norm_zone, dist_btwn_nearest_agent = \
            self._check_for_collisions()

        reward_g =np.zeros(len(self.agents))
        reward_c =np.zeros(len(self.agents))
        reward_w = np.zeros(len(self.agents))

        for i, agent in enumerate(self.agents):
            if self.dis2goal_cur[i] == 0:
                self.dis2goal_pre[i] = agent.dist_to_goal
            else:
                self.dis2goal_pre[i] = copy.deepcopy(self.dis2goal_cur[i])
            self.dis2goal_cur[i] = agent.dist_to_goal
            # reward_g[i] = (self.dis2goal_pre[i] - self.dis2goal_cur[i]) * 5
            reward_g[i] = (self.dis2goal_pre[i] - self.dis2goal_cur[i])*Config.AMPL_GETING_CLOSE2GOAL
            # if reward_g[i]>0:
            #     reward_g[i]*=5

            if agent.is_at_goal:
                # reward_g[i]=self.reward_at_goal
                reward_g[i] = Config.REWARD_AT_GOAL
            else:
                # agents at their goal shouldn't be penalized if someone else
                # bumps into them
                if agent.was_in_collision_already is False:
                    if collision_with_agent[i]:
                        # reward_c[i] = self.reward_collision_with_agent
                        reward_c[i] = Config.REWARD_COLLISION_WITH_AGENT
                        agent.in_collision = True
                        # print("Agent %i: Collision with another agent!"
                        #       % agent.id)
                    elif collision_with_wall[i]:
                        # reward_c[i] = self.reward_collision_with_wall
                        reward_c[i] = Config.REWARD_COLLISION_WITH_WALL
                        agent.in_collision = True
                        # print("Agent %i: Collision with wall!"
                        # % agent.id)
                    else:
                        # There was no collision
                        if dist_btwn_nearest_agent[i] <= Config.GETTING_CLOSE_RANGE:
                            # reward_c[i] =  - dist_btwn_nearest_agent[i] * Config.AMPL_GETING_CLOSE2AGENTS
                            rewards[i] = -0.1 - dist_btwn_nearest_agent[i] / 2.
                            # print("Agent %i: Got close to another agent!"
                            #       % agent.id)
                        if abs(agent.past_actions[0, 1]) > self.wiggly_behavior_threshold:
                            # Slightly penalize wiggly behavior
                            reward_c[i] += self.reward_wiggly_behavior
                        # elif entered_norm_zone[i]:
                        #     rewards[i] = self.reward_entered_norm_zone
            [v,w]=agent.vel_ego_frame
            if np.abs(w) > 1.05:
                reward_w[i] = -0.1 * np.abs(w)

            rewards[i]=reward_g[i] + reward_c[i] +reward_w[i]

        rewards = np.clip(rewards, self.min_possible_reward,
                          self.max_possible_reward)
        if Config.TRAIN_SINGLE_AGENT:
            rewards = rewards[0]

        return rewards

    def is_terminal(self):
        max_time_step=Config.TERMINAL_STEP
        terminal_list = []
        results=[]
        for i, agent in enumerate(self.agents):
            terminate = False
            result= 'runing'
            if agent.is_at_goal:
                terminate=True
                result='Reach Goal'
            elif agent.in_collision:
                terminate = True
                result = 'Crashed'
            elif agent.step_num>max_time_step or agent.ran_out_of_time:
                terminate = True
                result = 'Time out'
            terminal_list.append(terminate)
            results.append(result)

        if Config.AGENTS_TO_Train!=Config.MAX_NUM_AGENTS_TO_SIM:
            terminal_list = terminal_list[0:Config.AGENTS_TO_Train]
            results = results[0:Config.AGENTS_TO_Train]

        return terminal_list,results

    def reset_agent(self,idxs,num_agents_in_env = None):
        """ Resets the environment, re-initializes agents, plots episode (if applicable) and returns an initial observation.

                Returns:
                    initial observation (np array): each agent's observation given the initial configuration
                """
        if self.episode_step_number is not None and self.episode_step_number > 0 and self.plot_episodes and self.test_case_index >= 0:
            plot_episode(self.agents, self.evaluate, self.map, self.test_case_index, self.id,
                         circles_along_traj=Config.PLOT_CIRCLES_ALONG_TRAJ, plot_save_dir=self.plot_save_dir,
                         plot_policy_name=self.plot_policy_name, limits=self.plt_limits, fig_size=self.plt_fig_size,
                         show=Config.SHOW_EPISODE_PLOTS, save=Config.SAVE_EPISODE_PLOTS)
            # if Config.ANIMATE_EPISODES:
            #     animate_episode(num_agents=len(self.agents), plot_save_dir=self.plot_save_dir,
            #                     plot_policy_name=self.plot_policy_name, test_case_index=self.test_case_index,
            #                     agents=self.agents)
            self.episode_number += 1

        num = len(idxs)

        if not Config.USING_EXTERNAL_TRAIN_CASES or num not in self.train_cases_cnt.keys() or num<num_agents_in_env:
            poly_raw_list = []
            th=0.5
            env_range = ((-Config.CIRCLE_RADIUS, Config.CIRCLE_RADIUS), (-Config.CIRCLE_RADIUS, Config.CIRCLE_RADIUS))
            for i, agent in enumerate(self.agents):
                if i in idxs:
                    continue
                x, y=agent.pos_global_frame
                gx, gy = agent.goal_global_frame
                poly_raw_list.append([(x - th, y - th), (x - th, y + th), (x + th, y + th), (x + th, y - th)])
                poly_raw_list.append([(gx - th, gy - th), (gx - th, gy + th), (gx + th, gy + th), (gx + th, gy - th)])

            for i in idxs:
                self.dis2goal_pre[i] = 0
                self.dis2goal_cur[i] = 0

                if Config.AGENT_RESET_TYPE=='random':
                    pose=self.generate_random_pose(rg=Config.CIRCLE_RADIUS)
                    goal=self.generate_random_goal(pose,rg=Config.CIRCLE_RADIUS)

                    agent_radius=0.2
                    if Config.AGENT_SIZE_RANGE is not  None:
                        agent_radius = np.random.uniform(Config.AGENT_SIZE_RANGE[0], Config.AGENT_SIZE_RANGE[1])

                    self.agents[i].reset(px=pose[0], py=pose[1], gx=goal[0], gy=goal[1], pref_speed=1, radius=agent_radius, heading=pose[2])
                elif Config.AGENT_RESET_TYPE == 'noCollision':
                    poly_list = gen_polygon_exterior_list(poly_raw_list)
                    coord_list = gen_goal_position_list(poly_list, env_size=env_range, obs_near_th=0.5, sample_step=0.4)

                    robot_init_pose = random.choice(coord_list)
                    goal = random.choice(coord_list)
                    distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
                    while distance < (Config.CIRCLE_RADIUS-0.5):
                        goal = random.choice(coord_list)
                        distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
                    x, y = robot_init_pose
                    gx, gy = goal
                    poly_raw_list.append([(x - th, y - th), (x - th, y + th), (x + th, y + th), (x + th, y - th)])
                    poly_raw_list.append([(gx - th, gy - th), (gx - th, gy + th), (gx + th, gy + th), (gx + th, gy - th)])

                    angle=np.random.uniform(-np.pi, np.pi)
                    agent_radius = 0.2
                    if Config.AGENT_SIZE_RANGE is not None:
                        agent_radius = np.random.uniform(Config.AGENT_SIZE_RANGE[0], Config.AGENT_SIZE_RANGE[1])

                    self.agents[i].reset(px=x, py=y, gx=gx, gy=gy, pref_speed=1.0, radius=agent_radius,
                                         heading=angle)
                else:
                    raise NotImplementedError
        else:
            cnt = self.train_cases_cnt[num]
            self.train_cases_cnt[num] += 1
            cases = self.train_cases[num][cnt]
            for i in idxs:
                c = cases[i]
                self.dis2goal_pre[i] = 0
                self.dis2goal_cur[i] = 0
                self.agents[i].reset(px=c[0], py=c[1], gx=c[2], gy=c[3], pref_speed=c[4],radius=c[5], heading=c[6])

        for state in Config.STATES_IN_OBS:
            for i in idxs:
                self.observation[i][state] = np.zeros((Config.STATE_INFO_DICT[state]['size']),
                                                          dtype=Config.STATE_INFO_DICT[state]['dtype'])
        return self._get_obs()

    def _take_action(self, actions, dt):
        num_actions_per_agent = 2  # speed, delta heading angle
        all_actions = np.zeros((len(self.agents), num_actions_per_agent), dtype=np.float32)

        if Config.USING_SINGLE_POLICY:
            all_obs = []
            agent_idxs=[]
            for agent_index, agent in enumerate(self.agents):
                if agent.is_done:
                    continue

                agent_idxs.append(agent_index)
                obs = self.observation[agent_index]
                vec_obs = np.array([])
                for state in Config.STATES_IN_OBS:
                    vec_obs = np.hstack([vec_obs, obs[state].flatten()])
                all_obs.append(vec_obs)
            all_obs = np.array(all_obs)
            # all_actions[:, :] = self.agents[0].policy.find_next_action(all_obs, self.agents, 0)
            actions=self.agents[0].policy.find_next_action(all_obs, self.agents, agent_idxs)
            for sn,agent_index in enumerate(agent_idxs):
                all_actions[agent_index, :] = actions[sn]
        else:
            # Agents set their action (either from external or w/ find_next_action)
            for agent_index, agent in enumerate(self.agents):
                if agent.is_done:
                    continue
                elif agent.policy.is_external:
                    all_actions[agent_index, :] = agent.policy.external_action_to_action(agent, actions[agent_index])
                else:
                    dict_obs = self.observation[agent_index]
                    all_actions[agent_index, :] = agent.policy.find_next_action(dict_obs, self.agents, agent_index)

        # After all agents have selected actions, run one dynamics update
        for i, agent in enumerate(self.agents):
            agent.take_action(all_actions[i,:], dt)

    def generate_random_pose(self,rg=9):
        x = np.random.uniform(-rg, rg)
        y = np.random.uniform(-rg, rg)
        dis = np.sqrt(x ** 2 + y ** 2)
        while (dis > rg) :
            x = np.random.uniform(-rg, rg)
            y = np.random.uniform(-rg, rg)
            dis = np.sqrt(x ** 2 + y ** 2)
        theta = np.random.uniform(-np.pi, np.pi)
        return [x, y, theta]

    def generate_random_goal(self,init_pose,rg=9):
        x = np.random.uniform(-rg, rg)
        y = np.random.uniform(-rg, rg)
        dis_origin = np.sqrt(x ** 2 + y ** 2)
        dis_goal = np.sqrt((x - init_pose[0]) ** 2 + (y - init_pose[1]) ** 2)
        # while (dis_origin > rg or dis_goal > (rg+1) or dis_goal < (rg-1)):
        while (dis_origin > rg or  dis_goal < (rg - 1)):
            x = np.random.uniform(-rg, rg)
            y = np.random.uniform(-rg, rg)
            dis_origin = np.sqrt(x ** 2 + y ** 2)
            dis_goal = np.sqrt((x - init_pose[0]) ** 2 + (y - init_pose[1]) ** 2)
        return [x, y]

    def generate_random_range_pose(self,rg=((-1, 1), (-1, 1))):
        x = np.random.uniform(rg[0][0], rg[0][1])
        y = np.random.uniform(rg[1][0], rg[1][1])
        theta = np.random.uniform(-np.pi, np.pi)
        return [x, y, theta]

    def saved2gif(self):
        if Config.ANIMATE_EPISODES:
            animate_episode(num_agents=len(self.agents), plot_save_dir=self.plot_save_dir,
                            plot_policy_name=self.plot_policy_name, test_case_index=self.test_case_index,
                            agents=self.agents)
        return

# register(
#     id='CollisionAvoidance-vs',
#     entry_point='gym_collision_avoidance.envs.collision_avoidance_env:CollisionAvoidanceEnv',
# )