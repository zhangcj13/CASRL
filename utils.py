import time
import copy
import numpy as np

import random
from shapely.geometry import Point, Polygon
import math
import torch
import logging
import os
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

from gym_collision_avoidance.envs.agent import Agent
# Policies
from gym_collision_avoidance.envs.policies.StaticPolicy import StaticPolicy
from gym_collision_avoidance.envs.policies.NonCooperativePolicy import NonCooperativePolicy
from gym_collision_avoidance.envs.policies.DRLLongPolicy import DRLLongPolicy
try:
    from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
except:
    from gym_collision_avoidance.envs.policies.StaticPolicy import StaticPolicy as RVOPolicy
    print("Torch not installed...")
from gym_collision_avoidance.envs.policies.CADRLPolicy import CADRLPolicy
from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy
from gym_collision_avoidance.envs.policies.ExternalPolicy import ExternalPolicy
from gym_collision_avoidance.envs.policies.LearningPolicy import LearningPolicy
from gym_collision_avoidance.envs.policies.CARRLPolicy import CARRLPolicy
from gym_collision_avoidance.envs.policies.LearningPolicyGA3C import LearningPolicyGA3C
from gym_collision_avoidance.envs.policies.LearningPolicyCNNOASS import LearningPolicyCNNOASS
policy_dict = {
    'RVO': RVOPolicy,
    'noncoop': NonCooperativePolicy,
    'carrl': CARRLPolicy,
    'external': ExternalPolicy,
    'GA3C_CADRL': GA3CCADRLPolicy,
    'learning': LearningPolicy,
    'learning_ga3c': LearningPolicyGA3C,
    'static': StaticPolicy,
    'CADRL': CADRLPolicy,
    'DRL' : DRLLongPolicy,
    'CNNOASS':LearningPolicyCNNOASS,
}
# Dynamics
from gym_collision_avoidance.envs.dynamics.UnicycleDynamics import UnicycleDynamics
# Sensors
from gym_collision_avoidance.envs.sensors.LaserScanSensor import LaserScanSensor
from gym_collision_avoidance.envs.sensors.OtherAgentsStatesSensor import OtherAgentsStatesSensor
sensor_dict = {
    'other_agents_states': OtherAgentsStatesSensor,
    'laserscan': LaserScanSensor,
    # 'other_agents_states_encoded': OtherAgentsStatesSensorEncode,
}



def generate_random_pose():
    x = np.random.uniform(-9, 9)
    y = np.random.uniform(-9, 9)
    dis = np.sqrt(x ** 2 + y ** 2)
    while (dis > 9):# and not rospy.is_shutdown():
        x = np.random.uniform(-9, 9)
        y = np.random.uniform(-9, 9)
        dis = np.sqrt(x ** 2 + y ** 2)
    theta = np.random.uniform(0, 2 * np.pi)
    return [x, y, theta]

def gen_goal_position_list(poly_list, env_size=((-6, 6), (-6, 6)), obs_near_th=0.5, sample_step=0.1):
    """
    Generate list of goal positions
    :param poly_list: list of obstacle polygon
    :param env_size: size of the environment
    :param obs_near_th: Threshold for near an obstacle
    :param sample_step: sample step for goal generation
    :return: goal position list
    """
    goal_pos_list = []
    x_pos, y_pos = np.mgrid[env_size[0][0]:env_size[0][1]:sample_step, env_size[1][0]:env_size[1][1]:sample_step]
    for x in range(x_pos.shape[0]):
        for y in range(x_pos.shape[1]):
            tmp_pos = [x_pos[x, y], y_pos[x, y]]
            tmp_point = Point(tmp_pos[0], tmp_pos[1])
            near_obstacle = False
            for poly in poly_list:
                tmp_dis = tmp_point.distance(poly)
                if tmp_dis < obs_near_th:
                    near_obstacle = True
            if near_obstacle is False:
                goal_pos_list.append(tmp_pos)
    return goal_pos_list

def gen_polygon_exterior_list(poly_point_list):
    """
    Generate list of obstacle in the environment as polygon exterior list
    :param poly_point_list: list of points of polygon (with first always be the out wall)
    :return: polygon exterior list
    """
    poly_list = []
    for i, points in enumerate(poly_point_list, 0):
        tmp_poly = Polygon(points)
        if i > 0:
            poly_list.append(tmp_poly)
        else:
            poly_list.append(tmp_poly.exterior)
    return poly_list

def creat_training_agents(num_agents,navi_type,
                          radius=0.2,pref_speed=1.0,
                          env_range = ((-2, 2), (-2, 2)),poly_raw_list=[],goal_raw_list=[],diff=1.0,
                          r_circle=1,
                          policy='CNNOASS',sensor='other_agents_states'):
    tc = np.zeros((num_agents, 7))

    c_radius=None

    if navi_type=='circle':
        for i in range(num_agents):
            tc[i, 4] = 1.0
            tc[i, 5] = 0.5
            theta_start = (2 * np.pi / num_agents) * i
            theta_end = theta_start + np.pi
            tc[i, 0] = r_circle * np.cos(theta_start)
            tc[i, 1] = r_circle * np.sin(theta_start)
            tc[i, 2] = r_circle * np.cos(theta_end)
            tc[i, 3] = r_circle * np.sin(theta_end)

            tc[i, 6] = theta_end  # theta_start
            # if i==0:
            #     tc[i, 0] = r_circle * np.cos(theta_start)
            #     tc[i, 1] = r_circle * np.sin(theta_start)+1

    elif navi_type=='random_goals':

        for i in range(num_agents):
            tc[i, 4] = 1.0
            tc[i, 5] = 0.5
            theta_start = (2 * np.pi / num_agents) * i
            theta_end = theta_start + np.pi
            tc[i, 0] = 0#r_circle * np.cos(theta_start)
            tc[i, 1] = 0#r_circle * np.sin(theta_start)

            gx = 2#np.random.uniform(env_range[0][0], env_range[0][1])
            gy = 2#np.random.uniform(env_range[1][0], env_range[1][1])
            distance = math.sqrt((gx - tc[i, 0]) ** 2 + (gy - tc[i, 1]) ** 2)
            while distance < diff:
                gx = np.random.uniform(env_range[0][0], env_range[0][1])
                gy = np.random.uniform(env_range[1][0], env_range[1][1])
                distance = math.sqrt((gx - tc[i, 0]) ** 2 + (gy - tc[i, 1]) ** 2)

            # tc[i, 2] = r_circle * np.cos(theta_end)
            # tc[i, 3] = r_circle * np.sin(theta_end)
            tc[i, 2] = gx
            tc[i, 3] = gy

            tc[i, 6] = np.random.uniform(-math.pi, math.pi)#,theta_end  # theta_start
    elif navi_type=='single':

        for i in range(num_agents):
            tc[i, 4] = 1.0
            tc[i, 5] = 0.5
            tc[i, 0] = 0
            tc[i, 1] = 0
            tc[i, 2] = 5
            tc[i, 3] = 5

            tc[i, 6] = 0  # theta_start

    elif navi_type=='test1':
        x=2
        poses=[[x*1.0,0.0,np.pi],[2.0*x,0.0,np.pi],[3.0*x,0.0,np.pi],
                   [-1.0*x,0.0,0],[-2.0*x,0.0,0],[-3.0*x,0.0,0]]
        goals=[[-1.0*x,0.0],[-2.0*x,0.0],[-3.0*x,0.0],
              [1.0*x,0.0],[2.0*x,0.0],[3.0*x,0.0]]
        c_radius=[0.2,0.2,0.2,
                  0.2,0.2,0.2]
        for i in range(num_agents):
            tc[i, 0] = poses[i][0]
            tc[i, 1] = poses[i][1]
            tc[i, 2] = goals[i][0]
            tc[i, 3] = goals[i][1]
            tc[i, 6] = poses[i][2]
    elif navi_type=='test2':
        x = 2
        poses=[[1.0*x,0.0,np.pi],[1.0*x,1.0*x,np.pi],[1.0*x,-1.0*x,np.pi],
                   [-1.0*x,0.0,0],[-1.0*x,1.0*x,0],[-1.0*x,-1.0*x,0]]
        goals=[[-1.0*x,0.0],[-1.0*x,1.0*x],[-1.0*x,-1.0*x],
              [1.0*x,0.0],[1.0*x,1.0*x],[1.0*x,-1.0*x]]
        c_radius=np.asarray([0.2,0.2,0.2,
                  0.2,0.2,0.2])*1
        for i in range(num_agents):
            tc[i, 0] = poses[i][0]
            tc[i, 1] = poses[i][1]
            tc[i, 2] = goals[i][0]
            tc[i, 3] = goals[i][1]
            tc[i, 6] = poses[i][2]
    elif navi_type=='test3':
        x=2.0
        poses = [[x, 2*x, np.pi],
                 [2 * x, x, np.pi], [2 * x, 2 * x, np.pi], [2 * x, 3 * x, np.pi],
                 [3 * x, x, np.pi], [3 * x, 2 * x, np.pi], [3 * x, 3 * x, np.pi],
                 [4 * x, x, np.pi], [4 * x, 2 * x, np.pi], [4 * x, 3 * x, np.pi],

                 [-x, 2 * x, 0],
                 [-2 * x, x, 0], [-2 * x, 2 * x, 0], [-2 * x, 3 * x, 0],
                 [-3 * x, x, 0], [-3 * x, 2 * x, 0], [-3 * x, 3 * x, 0],
                 [-4 * x, x, 0], [-4 * x, 2 * x, 0], [-4 * x, 3 * x, 0],
                 ]
        goals=[
            [-x, -x],
            [-2 * x, 2 * x], [-2 * x, 1 * x], [-2 * x, 0 * x],
            [-3 * x, 2 * x], [-3 * x, 1 * x], [-3 * x, 0 * x],
            [-4 * x, 2 * x], [-4 * x, 1 * x], [-4 * x, 0 * x],

            [x, -x],
            [2 * x, 2 * x], [2 * x, 1 * x], [2 * x, 0 * x],
            [3 * x, 2 * x], [3 * x, 1 * x], [3 * x, 0 * x],
            [4 * x, 2 * x], [4 * x, 1 * x], [4 * x, 0 * x],
        ]
        for i in range(num_agents):
            tc[i, 0] = poses[i][0]
            tc[i, 1] = poses[i][1]
            tc[i, 2] = goals[i][0]
            tc[i, 3] = goals[i][1]
            tc[i, 6] = poses[i][2]

    else:
        poly_raw_list = []
        c_radius = []
        th=0.5
        for i in range(num_agents):
            poly_list = gen_polygon_exterior_list(poly_raw_list)
            coord_list = gen_goal_position_list(poly_list, env_size=env_range,obs_near_th=0.5, sample_step=0.1)

            robot_init_pose = random.choice(coord_list)
            goal = random.choice(coord_list)

            distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)
            while distance < diff:
                goal = random.choice(coord_list)
                distance = math.sqrt((goal[0] - robot_init_pose[0]) ** 2 + (goal[1] - robot_init_pose[1]) ** 2)

            x,y=robot_init_pose
            gx,gy=goal
            poly_raw_list.append([(x - th, y - th), (x - th, y + th), (x + th, y + th), (x + th, y - th)])
            poly_raw_list.append([(gx - th, gy - th), (gx - th, gy + th), (gx + th, gy + th), (gx + th, gy - th)])

            tc[i, 4] = 1.0
            tc[i, 5] = 0.5
            tc[i, 0] = x
            tc[i, 1] = y
            tc[i, 2] = gx
            tc[i, 3] = gy
            tc[i, 6] = np.random.uniform(-np.pi, np.pi)
            c_radius.append(np.random.uniform(0.1, 0.5))

        # poly_raw_list=[[(env_range[0][0], env_range[1][0]), (env_range[0][0], env_range[1][1]), (env_range[0][1], env_range[1][1]), (env_range[0][1], env_range[1][0])]]
        #
        # rand_goal_list = []
        # rand_robot_init_pose_list = []
        #
        # c_radius=[]
        # for i in range(num_agents):
        #     poly_list = gen_polygon_exterior_list(poly_raw_list)
        #     coord_list = gen_goal_position_list(poly_list, env_size=env_range)
        #
        #     c_radius.append(np.random.uniform(0.1, 0.5))
        #     x = np.random.uniform(env_range[0][0], env_range[0][1])
        #     y = np.random.uniform(env_range[1][0], env_range[1][1])
        #
        #     gx = np.random.uniform(env_range[0][0], env_range[0][1])
        #     gy = np.random.uniform(env_range[1][0], env_range[1][1])
        #
        #     distance = math.sqrt((x - gx) ** 2 + (y - gy) ** 2)
        #     while distance < diff:
        #         gx = np.random.uniform(env_range[0][0], env_range[0][1])
        #         gy = np.random.uniform(env_range[1][0], env_range[1][1])
        #         distance = math.sqrt((x - gx) ** 2 + (y - gy) ** 2)
        #     tc[i, 4] = 1.0
        #     tc[i, 5] = 0.5
        #     tc[i, 0] = x
        #     tc[i, 1] = y
        #     tc[i, 2] = gx
        #     tc[i, 3] = gy
        #     tc[i, 6] = np.random.uniform(-np.pi, np.pi)
    test_case = tc
    agents = []

    for id, c in enumerate(test_case):
        if c_radius is not None:
            radius=c_radius[id]
        agent = Agent(c[0], c[1], c[2], c[3], radius, pref_speed, c[6], policy_dict[policy], UnicycleDynamics,
                      [sensor_dict[sensor]], id)
        agents.append(agent)
    return agents

def get_filter_index(d_list):
    filter_index = []
    filter_flag = 0
    step = d_list.shape[0]
    num_env = d_list.shape[1]
    for i in range(num_env):
        for j in range(step):
            if d_list[j, i] == True:
                filter_flag += 1
            else:
                filter_flag = 0
            if filter_flag >= 2:
                filter_index.append(num_env*j + i)
    return filter_index

if __name__ == '__main__':
    poly_raw_list = []
    c_radius = []
    th = 0.5
    poly_list = gen_polygon_exterior_list(poly_raw_list)
    coord_list = gen_goal_position_list(poly_list, env_size=((-5, 5), (-5, 5)), obs_near_th=0.5, sample_step=0.5)

    robot_init_pose = random.choice(coord_list)

    poses = random.sample(coord_list,5)

    print(robot_init_pose)




