import numpy as np


class Config(object):
    def __init__(self):
        #########################################################################
        # GENERAL PARAMETERS
        num_agents = 4
        self.train_plat = 'torch'
        self.COLLISION_AVOIDANCE = True
        self.continuous, self.discrete = range(2)  # Initialize game types as enum
        self.ACTION_SPACE_TYPE = self.continuous

        ### DISPLAY
        self.ANIMATE_EPISODES = False  # False  True
        self.SHOW_EPISODE_PLOTS = False  # False
        self.SAVE_EPISODE_PLOTS = False  # False
        if not hasattr(self, "PLOT_CIRCLES_ALONG_TRAJ"):
            self.PLOT_CIRCLES_ALONG_TRAJ = True
        self.ANIMATION_PERIOD_STEPS = 1  # 5 # plot every n-th DT step (if animate mode on)
        self.PLT_LIMITS = None
        self.PLT_FIG_SIZE = (10, 8)

        if not hasattr(self, "USE_STATIC_MAP"):
            self.USE_STATIC_MAP = False

        ### TRAIN / PLAY / EVALUATE
        self.TRAIN_MODE = True  # Enable to see the trained agent in action (for testing)
        self.PLAY_MODE = False  # Enable to see the trained agent in action (for testing)
        self.EVALUATE_MODE = False  # Enable to see the trained agent in action (for testing)

        ### REWARDS
        self.REWARD_AT_GOAL = 1.0  # reward given when agent reaches goal position
        self.REWARD_COLLISION_WITH_AGENT = -0.25  # reward given when agent collides with another agent
        self.REWARD_COLLISION_WITH_WALL = -0.25  # reward given when agent collides with wall
        self.REWARD_GETTING_CLOSE = -0.1  # reward when agent gets close to another agent (unused?)
        self.REWARD_ENTERED_NORM_ZONE = -0.05  # reward when agent enters another agent's social zone
        self.REWARD_TIME_STEP = 0.0  # default reward given if none of the others apply (encourage speed)
        self.REWARD_WIGGLY_BEHAVIOR = 0.0
        self.WIGGLY_BEHAVIOR_THRESHOLD = np.inf
        self.COLLISION_DIST = 0.0  # meters between agents' boundaries for collision
        self.GETTING_CLOSE_RANGE = 0.2  # meters between agents' boundaries for collision
        # self.SOCIAL_NORMS = "right"
        # self.SOCIAL_NORMS = "left"
        self.SOCIAL_NORMS = "none"

        ### SIMULATION
        self.DT = 0.2  # seconds between simulation time steps
        self.NEAR_GOAL_THRESHOLD = 0.2
        self.MAX_TIME_RATIO = 2.  # agent has this number times the straight-line-time to reach its goal before "timing out"

        ### TEST CASE SETTINGS
        self.TEST_CASE_FN = "get_testcase_random"
        self.TEST_CASE_ARGS = {
            'policy_to_ensure': 'learning_ga3c',
            'policies': ['noncoop', 'learning_ga3c', 'static'],
            'policy_distr': [0.05, 0.9, 0.05],
            'speed_bnds': [0.5, 2.0],
            'radius_bnds': [0.2, 0.8],
            'side_length': [
                {'num_agents': [0, 5], 'side_length': [4, 5]},
                {'num_agents': [5, np.inf], 'side_length': [6, 8]},
            ],
            # 'agents_sensors': ['other_agents_states_encoded'],
        }

        if not hasattr(self, "MAX_NUM_AGENTS_IN_ENVIRONMENT"):
            self.MAX_NUM_AGENTS_IN_ENVIRONMENT = num_agents  # 4
        if not hasattr(self, "MAX_NUM_AGENTS_TO_SIM"):
            self.MAX_NUM_AGENTS_TO_SIM = num_agents  # 4
        self.MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
        if not hasattr(self, "MAX_NUM_OTHER_AGENTS_OBSERVED"):
            self.MAX_NUM_OTHER_AGENTS_OBSERVED = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1

        ### EXPERIMENTS
        self.PLOT_EVERY_N_EPISODES = 100  # for tensorboard visualization

        ### SENSORS
        self.SENSING_HORIZON = np.inf
        # self.SENSING_HORIZON  = 3.0
        self.LASERSCAN_LENGTH = 36 #512  # num range readings in one scan
        self.LASERSCAN_NUM_PAST = 1  # 3 # num range readings in one scan
        self.NUM_STEPS_IN_OBS_HISTORY = 1  # number of time steps to store in observation vector
        self.NUM_PAST_ACTIONS_IN_STATE = 0

        ### RVO AGENTS
        self.RVO_TIME_HORIZON = 5.0
        self.RVO_COLLAB_COEFF = 0.5
        self.RVO_ANTI_COLLAB_T = 1.0

        ### OBSERVATION VECTOR
        self.TRAIN_SINGLE_AGENT = False
        self.STATE_INFO_DICT = {
            'dist_to_goal': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("dist_to_goal")',
                'std': np.array([5.], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
            },
            'radius': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("radius")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([0.5], dtype=np.float32)
            },
            'heading_ego_frame': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [-np.pi, np.pi],
                'attr': 'get_agent_data("heading_ego_frame")',
                'std': np.array([3.14], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
            },
            'speed_ego_frame': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("speed_ego_frame")',
                'std': np.array([6], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
            },
            'vel_ego_frame': {
                'dtype': np.float32,
                'size': 2,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("vel_ego_frame")',
                'std': np.array([6], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
            },
            'pref_speed': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("pref_speed")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([1.0], dtype=np.float32)
            },
            'num_other_agents': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("num_other_agents_observed")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([1.0], dtype=np.float32)
            },
            'other_agent_states': {
                'dtype': np.float32,
                'size': 7,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("other_agent_states")',
                'std': np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32),
                'mean': np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32)
            },
            'other_agents_states': {
                'dtype': np.float32,
                'size': (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 7),
                'bounds': [-np.inf, np.inf],
                'attr': 'get_sensor_data("other_agents_states")',
                'std': np.tile(np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32),
                               (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
                'mean': np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32),
                                (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
            },
            'laserscan': {
                'dtype': np.float32,
                'size': (self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH),
                'bounds': [0., 6.],
                'attr': 'get_sensor_data("laserscan")',
                'std': 5. * np.ones((self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH), dtype=np.float32),
                'mean': 5. * np.ones((self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH), dtype=np.float32)
            },
            'is_learning': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0., 1.],
                'attr': 'get_agent_data_equiv("policy.str", "learning")'
            },
            'other_agents_states_encoded': {
                'dtype': np.float32,
                'size': 100,
                'bounds': [0., 1.],
                'attr': 'get_sensor_data("other_agents_states_encoded")'
            }
        }
        self.setup_obs()

        # self.AGENT_SORTING_METHOD = "closest_last"
        self.AGENT_SORTING_METHOD = "closest_first"
        # self.AGENT_SORTING_METHOD = "time_to_impact"

        self.NEW_REWARD_TYPE = False

        self.OTHER_AGENTS_INIT_STATE = None
        self.USING_MAX_HEADCHANGE = False
        self.USING_SINGLE_POLICY = False
        self.USING_EXTERNAL_TRAIN_CASES = False
        self.AMPL_GETING_CLOSE2GOAL = 0.0
        self.FIXED_AGENTS_NUM = False
        self.DISCRETE_CONTROL_FLAG=False

        self.USE_ENERGY = False
        self.COLLABORATIVE = False
        self.COLLABORATIVE_TYPE=''
        self.AGENT_STOP_IF_DONE = {
            'is_at_goal': True,
            'ran_out_of_time': True,
            'in_collision': True,
            'ran_out_of_energy': True,
        }

    def setup_obs(self):
        if not hasattr(self, "STATES_IN_OBS"):
            self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed',
                                  'radius', 'other_agents_states']
            # STATES_IN_OBS = ['dist_to_goal', 'radius', 'heading_ego_frame', 'pref_speed', 'other_agent_states', 'use_ppo', 'laserscan']
            # STATES_IN_OBS = ['dist_to_goal', 'radius', 'heading_ego_frame', 'pref_speed', 'other_agent_states', 'use_ppo'] # 2-agent net
            # STATES_IN_OBS = ['dist_to_goal', 'radius', 'heading_ego_frame', 'pref_speed', 'other_agents_states', 'use_ppo', 'num_other_agents', 'laserscan'] # LSTM
        if not hasattr(self, "STATES_NOT_USED_IN_POLICY"):
            self.STATES_NOT_USED_IN_POLICY = ['is_learning']

        self.MEAN_OBS = {};
        self.STD_OBS = {}
        for state in self.STATES_IN_OBS:
            if 'mean' in self.STATE_INFO_DICT[state]:
                self.MEAN_OBS[state] = self.STATE_INFO_DICT[state]['mean']
            if 'std' in self.STATE_INFO_DICT[state]:
                self.STD_OBS[state] = self.STATE_INFO_DICT[state]['std']

    def set_display(self, animate=False):
        ### DISPLAY
        self.ANIMATE_EPISODES = animate  # False True
        self.SHOW_EPISODE_PLOTS = True
        self.SAVE_EPISODE_PLOTS = True
        self.PLOT_CIRCLES_ALONG_TRAJ = True
        # self.PLT_LIMITS = ((-5, 5), (-5, 5))
        self.PLT_FIG_SIZE = (8, 8)
        self.MAX_TIME_RATIO = 8

        # self.EVALUATE_MODE=True


class EvaluateConfig(Config):
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 19
        Config.__init__(self)
        self.EVALUATE_MODE = True
        self.TRAIN_MODE = False
        self.DT = 0.1
        self.MAX_TIME_RATIO = 8.


class Example(EvaluateConfig):
    def __init__(self):
        EvaluateConfig.__init__(self)
        self.SAVE_EPISODE_PLOTS = True
        self.PLOT_CIRCLES_ALONG_TRAJ = True
        self.ANIMATE_EPISODES = True
        # self.SENSING_HORIZON = 4
        # self.PLT_LIMITS = [[-20, 20], [-20, 20]]
        # self.PLT_FIG_SIZE = (10,10)


class Formations(EvaluateConfig):
    def __init__(self):
        EvaluateConfig.__init__(self)
        self.SAVE_EPISODE_PLOTS = True
        self.SHOW_EPISODE_PLOTS = False
        self.ANIMATE_EPISODES = True
        self.NEAR_GOAL_THRESHOLD = 0.2
        self.PLT_LIMITS = [[-5, 6], [-2, 7]]
        self.PLT_FIG_SIZE = (10, 10)
        self.PLOT_CIRCLES_ALONG_TRAJ = False
        self.NUM_AGENTS_TO_TEST = [6]
        self.POLICIES_TO_TEST = ['GA3C-CADRL-10']
        self.NUM_TEST_CASES = 2
        self.LETTERS = ['C', 'A', 'D', 'R', 'L']


class SmallTestSuite(EvaluateConfig):
    def __init__(self):
        EvaluateConfig.__init__(self)
        self.SAVE_EPISODE_PLOTS = True
        self.SHOW_EPISODE_PLOTS = False
        self.ANIMATE_EPISODES = False
        self.PLOT_CIRCLES_ALONG_TRAJ = True
        self.NUM_TEST_CASES = 4


class FullTestSuite(EvaluateConfig):
    def __init__(self):
        self.MAX_NUM_OTHER_AGENTS_OBSERVED = 19
        EvaluateConfig.__init__(self)
        self.SAVE_EPISODE_PLOTS = True
        self.SHOW_EPISODE_PLOTS = False
        self.ANIMATE_EPISODES = False
        self.PLOT_CIRCLES_ALONG_TRAJ = True

        self.NUM_TEST_CASES = 4
        self.NUM_AGENTS_TO_TEST = [2, 3, 4]
        self.RECORD_PICKLE_FILES = False

        # # DRLMACA
        # self.FIXED_RADIUS_AND_VPREF = True
        # self.NEAR_GOAL_THRESHOLD = 0.8

        # Normal
        self.POLICIES_TO_TEST = [
            'CADRL', 'RVO', 'GA3C-CADRL-10'
            # 'GA3C-CADRL-4-WS-4-1', 'GA3C-CADRL-4-WS-4-2', 'GA3C-CADRL-4-WS-4-3', 'GA3C-CADRL-4-WS-4-4', 'GA3C-CADRL-4-WS-4-5',
            # 'GA3C-CADRL-4-WS-6-1', 'GA3C-CADRL-4-WS-6-2', 'GA3C-CADRL-4-WS-6-3', 'GA3C-CADRL-4-WS-6-4',
            # 'GA3C-CADRL-4-WS-8-1', 'GA3C-CADRL-4-WS-8-2', 'GA3C-CADRL-4-WS-8-3', 'GA3C-CADRL-4-WS-8-4',
            # 'GA3C-CADRL-4-LSTM-1', 'GA3C-CADRL-4-LSTM-2', 'GA3C-CADRL-4-LSTM-3', 'GA3C-CADRL-4-LSTM-4', 'GA3C-CADRL-4-LSTM-5',
            # 'GA3C-CADRL-10-WS-4-1', 'GA3C-CADRL-10-WS-4-2', 'GA3C-CADRL-10-WS-4-3', 'GA3C-CADRL-10-WS-4-4', 'GA3C-CADRL-10-WS-4-5',
            # 'GA3C-CADRL-10-WS-6-1', 'GA3C-CADRL-10-WS-6-2', 'GA3C-CADRL-10-WS-6-3', 'GA3C-CADRL-10-WS-6-4',
            # 'GA3C-CADRL-10-WS-8-1', 'GA3C-CADRL-10-WS-8-2', 'GA3C-CADRL-10-WS-8-3', 'GA3C-CADRL-10-WS-8-4',
            # 'GA3C-CADRL-10-LSTM-1', 'GA3C-CADRL-10-LSTM-2', 'GA3C-CADRL-10-LSTM-3', 'GA3C-CADRL-10-LSTM-4', 'GA3C-CADRL-10-LSTM-5',
            # 'CADRL', 'RVO'
        ]
        self.FIXED_RADIUS_AND_VPREF = False
        self.NEAR_GOAL_THRESHOLD = 0.2


class CollectRegressionDataset(EvaluateConfig):
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 4
        self.MAX_NUM_AGENTS_TO_SIM = 4
        self.DATASET_NAME = ""

        # # Laserscan mode
        # self.USE_STATIC_MAP = True
        # self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed', 'radius', 'laserscan']
        # self.DATASET_NAME = "laserscan_"

        EvaluateConfig.__init__(self)
        self.TEST_CASE_ARGS['policies'] = 'CADRL'
        self.AGENT_SORTING_METHOD = "closest_first"

        # # Laserscan mode
        # self.TEST_CASE_ARGS['agents_sensors'] = ['laserscan', 'other_agents_states']


class VisConfig(Config):
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 50
        Config.__init__(self)
        self.EVALUATE_MODE = True
        self.TRAIN_MODE = False
        self.DT = 0.1
        self.MAX_TIME_RATIO = 8.

        self.ANIMATE_EPISODES = True
        self.SHOW_EPISODE_PLOTS = True
        self.SAVE_EPISODE_PLOTS = True
        self.PLOT_CIRCLES_ALONG_TRAJ = False

        self.NEAR_GOAL_THRESHOLD = 0.2
        self.USE_STATIC_MAP = True
        self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed',
                              'radius', 'other_agents_states', 'laserscan']

        # self.STATES_IN_OBS = ['dist_to_goal', 'radius', 'heading_ego_frame', 'pref_speed', 'other_agents_states', 'use_ppo', 'num_other_agents', 'laserscan'] # LSTM


class Train(Config):
    def __init__(self):
        self.TRAIN_SINGLE_AGENT = False

        # self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed', 'radius', 'other_agents_states_encoded']

        # self.STATES_IN_OBSX = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed', 'radius', 'other_agents_states_encoded']

        self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed',
                              'radius', 'other_agents_states']
        self.STATES_NOT_USED_IN_POLICY = ['is_learning']

        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 6
        self.MAX_NUM_AGENTS_TO_SIM = 6

        Config.__init__(self)

        # self.TEST_CASE_ARGS['num_agents'] = 2
        # self.TEST_CASE_ARGS['policy_to_ensure'] = 'LearningPolicyGA3C'
        # self.TEST_CASE_ARGS['policies'] = ['LearningPolicyGA3C', 'RVO']
        # self.TEST_CASE_ARGS['policy_distr'] = [0.9, 0.1]


class FullEvalTest(Config):
    def __init__(self):

        self.MAX_NUM_OTHER_AGENTS_OBSERVED=50

        self.Train_Flag = True # True  False

        self.USING_LASER = False #True

        self.NUM_TEST_CASES = 500
        self.NUM_AGENTS_TO_TEST = [2,3,4,6,8,10]#[6,8,10,16,18,20]#[6,8,10,16,18,20]
        self.RECORD_PICKLE_FILES = True

        self.POLICIES_TO_TEST = [
            # 'carrl',
            # 'SpikDDPG',
            # 'DRL',
            'LSTM_DPPO_DC',
            # 'GTrMean_DPPO_DC',
            # 'SpikLSTM_DPPO_DC',
            # 'SpikGTr_DPPO_DC',
            # 'SpikGTrAN_DPPO_DC',
            # 'SpikMF_DPPO_DC',
            # 'GA3C_CADRL',
            # 'RVO',
            # 'CADRL',
            # 'LSTM_GA3C',
            # 'SpikLSTM_GA3C',
            # 'LSTM_PPOs',
            # 'LSTM_DPPO',
            # 'PPO_CADRL',
            # 'SpikGTr_DDPG',
            # 'SpikLSTM_DDPG',
            # 'LSTM_DDPG',
            # 'SpikLSTM_GA3C',
        ]
        self.RESCALE_POLICIES = ['x']
        self.INITSTATE_POLICIES = ['LSTM_DDPG', 'SpikGTr_DDPG', 'SpikLSTM_DDPG',
                                   'SpikLSTM_GA3C',
                                   'LSTM_PPOs','LSTM_DPPO',
                                   'LSTM_DPPO_DC','SpikLSTM_DPPO_DC','SpikGTr_DPPO_DC','SpikGTrAN_DPPO_DC','SpikMF_DPPO_DC',]
        self.SINGLE_POLICIES = ['LSTM_DDPG', 'SpikGTr_DDPG', 'SpikLSTM_DDPG',
                                'LSTM_GA3C','SpikLSTM_GA3C',
                                'LSTM_PPOs','LSTM_DPPO',
                                'LSTM_DPPO_DC','SpikLSTM_DPPO_DC','SpikGTr_DPPO_DC','SpikGTrAN_DPPO_DC','SpikMF_DPPO_DC',]
        if True: #GA3C
            self.DEVICE = 'cuda'  # 'cpu'
            self.BETA_START = 1e-4  # Entropy regularization hyper-parameter
            self.LOG_EPSILON = 1e-6

        self.TRAIN_SINGLE_AGENT = False
        self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed',
                              'radius', 'other_agents_states']
        self.STATES_NOT_USED_IN_POLICY = ['is_learning']

        TAN = self.NUM_AGENTS_TO_TEST[0]
        self.init_agents_type = ''

        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = TAN
        self.MAX_NUM_AGENTS_TO_SIM = TAN
        self.AGENTS_STATES_TO_SIM = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
        self.AGENTS_TO_Train = TAN

        if self.USING_LASER:
            self.STATES_IN_OBS = ['is_learning', 'dist_to_goal', 'heading_ego_frame', 'vel_ego_frame', 'pref_speed',
                                  'radius', 'laserscan']
            self.USE_STATIC_MAP = True

        Config.__init__(self)
        self.USING_MAX_HEADCHANGE = True

        # Reward setting
        self.REWARD_AT_GOAL = 5  # 15
        self.REWARD_COLLISION_WITH_AGENT = -5  # -15 -0.25
        self.REWARD_COLLISION_WITH_WALL = -5  # -15 -0.25
        self.REWARD_GETTING_CLOSE = -1.0  # -0.1  # reward when agent gets close to another agent (unused?)
        self.REWARD_ENTERED_NORM_ZONE = -0.5  # -0.05

        self.AMPL_GETING_CLOSE2GOAL = 1.0  # 1.0 #1.5
        self.AMPL_GETING_CLOSE2AGENTS = 1.2

        self.COLLISION_DIST = 0.0  # meters between agents' boundaries for collision
        self.GETTING_CLOSE_RANGE = 0.2  # meters between agents' boundaries for collision

        self.EVALUATE_MODE = True
        self.TRAIN_MODE = False
        # self.DT = 0.1
        self.MAX_TIME_RATIO = 5#10
        self.CIRCLE_RADIUS = 6  # 9 # self.MAX_NUM_AGENTS_IN_ENVIRONMENT*0.4+2.0
        self.TERMINAL_STEP = 150  # 150 # int(self.CIRCLE_RADIUS*15)#150

        ### DISPLAY
        self.ANIMATE_EPISODES = False  # False True
        self.SHOW_EPISODE_PLOTS = False
        self.SAVE_EPISODE_PLOTS = False
        self.PLOT_CIRCLES_ALONG_TRAJ = False
        # self.PLT_LIMITS = ((-10,10),(-10,10))
        lrn = self.CIRCLE_RADIUS * 1.2
        self.PLT_LIMITS = ((-lrn, lrn), (-lrn, lrn))
        self.PLT_FIG_SIZE = (5, 5)
        # self.PLT_FIG_SIZE = (10, 10)
        self.NEAR_GOAL_THRESHOLD = 0.2

        self.AGENT_SIZE_RANGE = (0.1, 0.5)
        # self.SENSING_HORIZON = 6

        self.NORMALIZE_INPUT = True
        self.FIRST_STATE_INDEX = 1 if not self.USING_LASER else 0
        self.HOST_AGENT_OBSERVATION_LENGTH = 4  # dist to goal, heading to goal, pref speed, radius
        self.OTHER_AGENT_OBSERVATION_LENGTH = 7  # other px, other py, other vx, other vy, other radius, combined radius, distance between
        self.OTHER_AGENT_FULL_OBSERVATION_LENGTH = self.OTHER_AGENT_OBSERVATION_LENGTH
        self.HOST_AGENT_STATE_SIZE = self.HOST_AGENT_OBSERVATION_LENGTH

        self.NN_INPUT_AVG_VECTOR = np.array([])
        self.NN_INPUT_STD_VECTOR = np.array([])
        self.NN_INPUT_SIZE = 0
        for state in self.STATES_IN_OBS:
            if state not in self.STATES_NOT_USED_IN_POLICY:
                self.NN_INPUT_SIZE += np.product(self.STATE_INFO_DICT[state]['size'])
                self.NN_INPUT_AVG_VECTOR = np.hstack(
                    [self.NN_INPUT_AVG_VECTOR, self.STATE_INFO_DICT[state]['mean'].flatten()])
                self.NN_INPUT_STD_VECTOR = np.hstack(
                    [self.NN_INPUT_STD_VECTOR, self.STATE_INFO_DICT[state]['std'].flatten()])


        if not self.Train_Flag:
            self.CIRCLE_RADIUS = 10  # 9 # self.MAX_NUM_AGENTS_IN_ENVIRONMENT*0.4+2.0
            self.TERMINAL_STEP = 150  # 150 # int(self.CIRCLE_RADIUS*15)#150

            ### DISPLAY
            self.ANIMATE_EPISODES = True  # False True
            self.SHOW_EPISODE_PLOTS = True
            self.SAVE_EPISODE_PLOTS = True
            self.PLOT_CIRCLES_ALONG_TRAJ = False
            lrn = self.CIRCLE_RADIUS * 1.2
            self.PLT_LIMITS = ((-lrn, lrn), (-lrn, lrn))
            self.PLT_FIG_SIZE = (8, 8)



    def set_agents_num(self, num, policy):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = num
        self.MAX_NUM_AGENTS_TO_SIM = num
        self.AGENTS_STATES_TO_SIM = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
        self.AGENTS_TO_Train = num

        self.MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
        self.MAX_NUM_OTHER_AGENTS_OBSERVED = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1

        self.STATE_INFO_DICT = {
            'dist_to_goal': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("dist_to_goal")',
                'std': np.array([5.], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
            },
            'radius': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("radius")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([0.5], dtype=np.float32)
            },
            'heading_ego_frame': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [-np.pi, np.pi],
                'attr': 'get_agent_data("heading_ego_frame")',
                'std': np.array([3.14], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
            },
            'speed_ego_frame': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("speed_ego_frame")',
                'std': np.array([6], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
            },
            'vel_ego_frame': {
                'dtype': np.float32,
                'size': 2,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("vel_ego_frame")',
                'std': np.array([6], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
            },
            'pref_speed': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("pref_speed")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([1.0], dtype=np.float32)
            },
            'num_other_agents': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("num_other_agents_observed")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([1.0], dtype=np.float32)
            },
            'other_agent_states': {
                'dtype': np.float32,
                'size': 7,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("other_agent_states")',
                'std': np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32),
                'mean': np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32)
            },
            'other_agents_states': {
                'dtype': np.float32,
                'size': (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 7),
                'bounds': [-np.inf, np.inf],
                'attr': 'get_sensor_data("other_agents_states")',
                'std': np.tile(np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32),
                               (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
                'mean': np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32),
                                (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
            },
            'laserscan': {
                'dtype': np.float32,
                'size': (self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH),
                'bounds': [0., 6.],
                'attr': 'get_sensor_data("laserscan")',
                'std': 5. * np.ones((self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH), dtype=np.float32),
                'mean': 5. * np.ones((self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH), dtype=np.float32)
            },
            'is_learning': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0., 1.],
                'attr': 'get_agent_data_equiv("policy.str", "learning")'
            },
            'other_agents_states_encoded': {
                'dtype': np.float32,
                'size': 100,
                'bounds': [0., 1.],
                'attr': 'get_sensor_data("other_agents_states_encoded")'
            }
        }



        # self.STATE_INFO_DICT = {
        #     'dist_to_goal': {
        #         'dtype': np.float32,
        #         'size': 1,
        #         'bounds': [-np.inf, np.inf],
        #         'attr': 'get_agent_data("dist_to_goal")',
        #         'std': np.array([5.], dtype=np.float32),
        #         'mean': np.array([0.], dtype=np.float32)
        #     },
        #     'radius': {
        #         'dtype': np.float32,
        #         'size': 1,
        #         'bounds': [0, np.inf],
        #         'attr': 'get_agent_data("radius")',
        #         'std': np.array([1.0], dtype=np.float32),
        #         'mean': np.array([0.5], dtype=np.float32)
        #     },
        #     'heading_ego_frame': {
        #         'dtype': np.float32,
        #         'size': 1,
        #         'bounds': [-np.pi, np.pi],
        #         'attr': 'get_agent_data("heading_ego_frame")',
        #         'std': np.array([3.14], dtype=np.float32),
        #         'mean': np.array([0.], dtype=np.float32)
        #     },
        #     'pref_speed': {
        #         'dtype': np.float32,
        #         'size': 1,
        #         'bounds': [0, np.inf],
        #         'attr': 'get_agent_data("pref_speed")',
        #         'std': np.array([1.0], dtype=np.float32),
        #         'mean': np.array([1.0], dtype=np.float32)
        #     },
        #     'num_other_agents': {
        #         'dtype': np.float32,
        #         'size': 1,
        #         'bounds': [0, np.inf],
        #         'attr': 'get_agent_data("num_other_agents_observed")',
        #         'std': np.array([1.0], dtype=np.float32),
        #         'mean': np.array([1.0], dtype=np.float32)
        #     },
        #     'other_agent_states': {
        #         'dtype': np.float32,
        #         'size': 7,
        #         'bounds': [-np.inf, np.inf],
        #         'attr': 'get_agent_data("other_agent_states")',
        #         'std': np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32),
        #         'mean': np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32)
        #     },
        #     'other_agents_states': {
        #         'dtype': np.float32,
        #         'size': (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 7),
        #         'bounds': [-np.inf, np.inf],
        #         'attr': 'get_sensor_data("other_agents_states")',
        #         'std': np.tile(np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32),
        #                        (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
        #         'mean': np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32),
        #                         (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
        #     },
        #     'laserscan': {
        #         'dtype': np.float32,
        #         'size': (self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH),
        #         'bounds': [0., 6.],
        #         'attr': 'get_sensor_data("laserscan")',
        #         'std': 5. * np.ones((self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH), dtype=np.float32),
        #         'mean': 5. * np.ones((self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH), dtype=np.float32)
        #     },
        #     'is_learning': {
        #         'dtype': np.float32,
        #         'size': 1,
        #         'bounds': [0., 1.],
        #         'attr': 'get_agent_data_equiv("policy.str", "learning")'
        #     },
        #     'other_agents_states_encoded': {
        #         'dtype': np.float32,
        #         'size': 100,
        #         'bounds': [0., 1.],
        #         'attr': 'get_sensor_data("other_agents_states_encoded")'
        #     }
        # }
        if policy in self.RESCALE_POLICIES:
            self.STATE_INFO_DICT['other_agent_states'] = {
                'dtype': np.float32,
                'size': 7,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("other_agent_states")',
                'std': np.array([5.0, 5.0, 1.0, 1.0, 0.5, 5.0, 5.0], dtype=np.float32),
                'mean': np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32)
            }
            self.STATE_INFO_DICT['other_agents_states'] = {
                'dtype': np.float32,
                'size': (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 7),
                'bounds': [-np.inf, np.inf],
                'attr': 'get_sensor_data("other_agents_states")',
                'std': np.tile(np.array([5.0, 5.0, 1.0, 1.0, 0.5, 5.0, 5.0], dtype=np.float32),
                               (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
                'mean': np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32),
                                (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
            }
            self.STATE_INFO_DICT['radius'] = {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("radius")',
                'std': np.array([0.5], dtype=np.float32),
                'mean': np.array([0.25], dtype=np.float32)
            }
        if policy in self.INITSTATE_POLICIES:
            self.OTHER_AGENTS_INIT_STATE = np.array([[-10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 10.0]])
        else:
            self.OTHER_AGENTS_INIT_STATE = None
        self.setup_obs()

        if policy in self.SINGLE_POLICIES:
            self.USING_SINGLE_POLICY = True

        self.NN_INPUT_AVG_VECTOR = np.array([])
        self.NN_INPUT_STD_VECTOR = np.array([])
        self.NN_INPUT_SIZE = 0
        for state in self.STATES_IN_OBS:
            if state not in self.STATES_NOT_USED_IN_POLICY:
                self.NN_INPUT_SIZE += np.product(self.STATE_INFO_DICT[state]['size'])
                self.NN_INPUT_AVG_VECTOR = np.hstack(
                    [self.NN_INPUT_AVG_VECTOR, self.STATE_INFO_DICT[state]['mean'].flatten()])
                self.NN_INPUT_STD_VECTOR = np.hstack(
                    [self.NN_INPUT_STD_VECTOR, self.STATE_INFO_DICT[state]['std'].flatten()])

class TrainPhase_DDPG(Config):
    def __init__(self):
        self.RESCALE_STATE = False

        self.USING_LASER = True  # True False

        self.IS_FILTER_INDEX = False

        self.Train_Flag = True  # True False

        self.TRAIN_SINGLE_AGENT = False
        self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed',
                              'radius', 'other_agents_states']
        self.SENSING_TYPR = np.inf
        if self.USING_LASER:
            self.STATES_IN_OBS = ['is_learning', 'dist_to_goal', 'heading_ego_frame', 'vel_ego_frame','pref_speed',
                                  'radius', 'laserscan']
            self.USE_STATIC_MAP = True

        self.STATES_NOT_USED_IN_POLICY = ['is_learning']

        num_agents = 10
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = num_agents  # 4
        self.MAX_NUM_AGENTS_TO_SIM = num_agents  # 4
        self.AGENTS_STATES_TO_SIM = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
        self.AGENTS_TO_Train = num_agents

        Config.__init__(self)


        self.USE_STATIC_MAP = True

        self.USING_MAX_HEADCHANGE = True
        self.USING_EXTERNAL_TRAIN_CASES = True

        self.ALL_ACT_ARCHS = ['LSTM_ActorNet',
                              'SpikLSTM_ActorNet',
                              'GTr_ActorNet',
                              'SpikGTr_ActorNet',
                              'GO_SpikGTr_ActorNet',
                              'SDDPG_ActorNet',
                              ]

        self.ALL_CRT_ARCHS = ['LSTM_CriticNet',
                              'GTr_CriticNet',
                              'DDPG_CriticNet',]

        SCAN_ACT_ARCH ='SDDPG_ActorNet'   # None 'ScanConvPolicy' 'SDDPG_ActorNet'
        SCAN_CRT_ARCH ='DDPG_CriticNet'   # None 'ScanConvPolicy' 'DDPG_CriticNet'
        self.ACT_ARCH = 'LSTM_ActorNet' if not self.USING_LASER else SCAN_ACT_ARCH
        self.CRT_ARCH = 'LSTM_CriticNet' if not self.USING_LASER else SCAN_CRT_ARCH

        # Reward setting
        # self.REWARD_TIME_STEP = 0.0  # default reward given if none of the others apply (encourage speed)
        # self.REWARD_WIGGLY_BEHAVIOR = 0.0
        # self.WIGGLY_BEHAVIOR_THRESHOLD = np.inf

        self.REWARD_AT_GOAL = 1.0  # 10 #15
        self.REWARD_COLLISION_WITH_AGENT = -0.25  # -10#-10#-15 -0.25
        self.REWARD_COLLISION_WITH_WALL = -0.25  # -10#-10 #-15 -0.25
        self.REWARD_GETTING_CLOSE = -0.1  # -0.1  # reward when agent gets close to another agent (unused?)
        self.REWARD_ENTERED_NORM_ZONE = -0.05  # -0.05

        self.AMPL_GETING_CLOSE2GOAL = 0.2  # 1.0 #1.5
        # self.AMPL_GETING_CLOSE2AGENTS = 1.2

        self.COLLISION_DIST = 0.0  # meters between agents' boundaries for collision
        self.GETTING_CLOSE_RANGE = 0.2  # meters between agents' boundaries for collision

        # SIM step length setting
        self.MAX_TIME_RATIO = 25
        self.TERMINAL_STEP = 1000  # 1000

        # env range  setting
        self.CIRCLE_RADIUS = 5
        self.AGENT_RESET_TYPE = 'noCollision'  # noCollision random

        ### DISPLAY setting
        self.ANIMATE_EPISODES = False  # False True
        self.SHOW_EPISODE_PLOTS = False
        self.SAVE_EPISODE_PLOTS = False
        self.PLOT_CIRCLES_ALONG_TRAJ = False
        lrn = self.CIRCLE_RADIUS * 1.8  # 1.2
        self.PLT_LIMITS = ((-lrn, lrn), (-lrn, lrn))
        self.PLT_FIG_SIZE = (8, 8)

        self.NEAR_GOAL_THRESHOLD = 0.2
        self.AGENT_SIZE_RANGE = (0.1, 0.5)
        self.SENSING_HORIZON = 6

        self.NORMALIZE_INPUT = True
        self.FIRST_STATE_INDEX = 1 if not self.USING_LASER else 0
        self.HOST_AGENT_OBSERVATION_LENGTH = 4  # dist to goal, heading to goal, pref speed, radius
        self.OTHER_AGENT_OBSERVATION_LENGTH = 7  # other px, other py, other vx, other vy, other radius, combined radius, distance between
        self.OTHER_AGENT_FULL_OBSERVATION_LENGTH = self.OTHER_AGENT_OBSERVATION_LENGTH
        self.HOST_AGENT_STATE_SIZE = self.HOST_AGENT_OBSERVATION_LENGTH

        if self.RESCALE_STATE:
            self.STATE_INFO_DICT['other_agent_states'] = {
                'dtype': np.float32,
                'size': 7,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("other_agent_states")',
                'std': np.array([5.0, 5.0, 1.0, 1.0, 0.5, 5.0, 5.0], dtype=np.float32),
                'mean': np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32)
            }
            self.STATE_INFO_DICT['other_agents_states'] = {
                'dtype': np.float32,
                'size': (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 7),
                'bounds': [-np.inf, np.inf],
                'attr': 'get_sensor_data("other_agents_states")',
                'std': np.tile(np.array([5.0, 5.0, 1.0, 1.0, 0.5, 5.0, 5.0], dtype=np.float32),
                               (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
                'mean': np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32),
                                (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
            }
            self.STATE_INFO_DICT['radius'] = {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("radius")',
                'std': np.array([0.5], dtype=np.float32),
                'mean': np.array([0.25], dtype=np.float32)
            }
            self.setup_obs()
        self.OTHER_AGENTS_INIT_STATE = np.array([[-10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 10.0]])

        self.NN_INPUT_AVG_VECTOR = np.array([])
        self.NN_INPUT_STD_VECTOR = np.array([])
        self.NN_INPUT_SIZE = 0
        for state in self.STATES_IN_OBS:
            if state not in self.STATES_NOT_USED_IN_POLICY:
                self.NN_INPUT_SIZE += np.product(self.STATE_INFO_DICT[state]['size'])
                self.NN_INPUT_AVG_VECTOR = np.hstack(
                    [self.NN_INPUT_AVG_VECTOR, self.STATE_INFO_DICT[state]['mean'].flatten()])
                self.NN_INPUT_STD_VECTOR = np.hstack(
                    [self.NN_INPUT_STD_VECTOR, self.STATE_INFO_DICT[state]['std'].flatten()])

        if not self.Train_Flag:
            self.CIRCLE_RADIUS = 6  # 9 # self.MAX_NUM_AGENTS_IN_ENVIRONMENT*0.4+2.0
            self.TERMINAL_STEP = 150  # 150 # int(self.CIRCLE_RADIUS*15)#150
            ### DISPLAY
            self.ANIMATE_EPISODES = True  # False True
            self.SHOW_EPISODE_PLOTS = True
            self.SAVE_EPISODE_PLOTS = True
            self.PLOT_CIRCLES_ALONG_TRAJ = False
            lrn = self.CIRCLE_RADIUS * 1.2
            self.PLT_LIMITS = ((-lrn, lrn), (-lrn, lrn))
            self.PLT_FIG_SIZE = (8, 8)

    def set_agents_num(self, num):
        self.AMPL_GETING_CLOSE2GOAL = self.AMPL_GETING_CLOSE2GOAL * 0.5

        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = num
        self.MAX_NUM_AGENTS_TO_SIM = num
        self.AGENTS_STATES_TO_SIM = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
        self.AGENTS_TO_Train = num

        self.MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
        self.MAX_NUM_OTHER_AGENTS_OBSERVED = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1

        self.STATE_INFO_DICT = {
            'dist_to_goal': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("dist_to_goal")',
                'std': np.array([5.], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
            },
            'radius': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("radius")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([0.5], dtype=np.float32)
            },
            'heading_ego_frame': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [-np.pi, np.pi],
                'attr': 'get_agent_data("heading_ego_frame")',
                'std': np.array([3.14], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
            },
            'speed_ego_frame': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("speed_ego_frame")',
                'std': np.array([6], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
            },
            'vel_ego_frame': {
                'dtype': np.float32,
                'size': 2,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("vel_ego_frame")',
                'std': np.array([6], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
            },
            'pref_speed': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("pref_speed")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([1.0], dtype=np.float32)
            },
            'num_other_agents': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("num_other_agents_observed")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([1.0], dtype=np.float32)
            },
            'other_agent_states': {
                'dtype': np.float32,
                'size': 7,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("other_agent_states")',
                'std': np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32),
                'mean': np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32)
            },
            'other_agents_states': {
                'dtype': np.float32,
                'size': (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 7),
                'bounds': [-np.inf, np.inf],
                'attr': 'get_sensor_data("other_agents_states")',
                'std': np.tile(np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32),
                               (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
                'mean': np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32),
                                (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
            },
            'laserscan': {
                'dtype': np.float32,
                'size': (self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH),
                'bounds': [0., 6.],
                'attr': 'get_sensor_data("laserscan")',
                'std': 5. * np.ones((self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH), dtype=np.float32),
                'mean': 5. * np.ones((self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH), dtype=np.float32)
            },
            'is_learning': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0., 1.],
                'attr': 'get_agent_data_equiv("policy.str", "learning")'
            },
            'other_agents_states_encoded': {
                'dtype': np.float32,
                'size': 100,
                'bounds': [0., 1.],
                'attr': 'get_sensor_data("other_agents_states_encoded")'
            }
        }
        if self.RESCALE_STATE:
            self.STATE_INFO_DICT['other_agent_states'] = {
                'dtype': np.float32,
                'size': 7,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("other_agent_states")',
                'std': np.array([5.0, 5.0, 1.0, 1.0, 0.5, 5.0, 5.0], dtype=np.float32),
                'mean': np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32)
            }
            self.STATE_INFO_DICT['other_agents_states'] = {
                'dtype': np.float32,
                'size': (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 7),
                'bounds': [-np.inf, np.inf],
                'attr': 'get_sensor_data("other_agents_states")',
                'std': np.tile(np.array([5.0, 5.0, 1.0, 1.0, 0.5, 5.0, 5.0], dtype=np.float32),
                               (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
                'mean': np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32),
                                (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
            }
            self.STATE_INFO_DICT['radius'] = {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("radius")',
                'std': np.array([0.5], dtype=np.float32),
                'mean': np.array([0.25], dtype=np.float32)
            }
        self.setup_obs()

        self.NN_INPUT_AVG_VECTOR = np.array([])
        self.NN_INPUT_STD_VECTOR = np.array([])
        self.NN_INPUT_SIZE = 0
        for state in self.STATES_IN_OBS:
            if state not in self.STATES_NOT_USED_IN_POLICY:
                self.NN_INPUT_SIZE += np.product(self.STATE_INFO_DICT[state]['size'])
                self.NN_INPUT_AVG_VECTOR = np.hstack(
                    [self.NN_INPUT_AVG_VECTOR, self.STATE_INFO_DICT[state]['mean'].flatten()])
                self.NN_INPUT_STD_VECTOR = np.hstack(
                    [self.NN_INPUT_STD_VECTOR, self.STATE_INFO_DICT[state]['std'].flatten()])


class TrainPhase_PPO(Config):
    def __init__(self):
        self.RESCALE_STATE = True

        self.USING_LASER = False  # True False

        self.IS_FILTER_INDEX = False

        self.Train_Flag = True  # True False

        self.TRAIN_SINGLE_AGENT = False
        self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed',
                              'radius', 'other_agents_states']
        self.SENSING_TYPR = np.inf
        if self.USING_LASER:
            self.STATES_IN_OBS = ['is_learning', 'dist_to_goal', 'heading_ego_frame', 'pref_speed',
                                  'radius', 'laserscan']
            self.USE_STATIC_MAP = True

        self.STATES_NOT_USED_IN_POLICY = ['is_learning']

        num_agents = 4
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = num_agents  # 4
        self.MAX_NUM_AGENTS_TO_SIM = num_agents  # 4
        self.AGENTS_STATES_TO_SIM = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
        self.AGENTS_TO_Train = num_agents

        Config.__init__(self)

        self.ALL_ARCHS=['LSTMPolicy',
                        'GTrPolicy',
                        'FixLengthPolicy',
                        'ScanPolicy',

                        'ScanConvPolicy',
                        'ScanGTrPolicy',]

        SCAN_ARCH= 'ScanGTrPolicy'
        self.NET_ARCH = 'FixLengthPolicy' if not self.USING_LASER else SCAN_ARCH

        #Reward setting
        self.REWARD_AT_GOAL = 10 #15
        self.REWARD_COLLISION_WITH_AGENT = -10#-15 -0.25
        self.REWARD_COLLISION_WITH_WALL = -10 #-15 -0.25
        self.REWARD_GETTING_CLOSE = -1.0#-0.1  # reward when agent gets close to another agent (unused?)
        self.REWARD_ENTERED_NORM_ZONE = -0.5#-0.05

        self.AMPL_GETING_CLOSE2GOAL=1.0 #1.0 #1.5
        self.AMPL_GETING_CLOSE2AGENTS = 1.2

        self.COLLISION_DIST = 0.0  # meters between agents' boundaries for collision
        self.GETTING_CLOSE_RANGE = 0.2  # meters between agents' boundaries for collision

        # SIM step length setting
        self.MAX_TIME_RATIO = 10 #8
        self.TERMINAL_STEP = 200 #150

        # env range  setting
        self.CIRCLE_RADIUS = 5
        self.AGENT_RESET_TYPE = 'random' # noCollision random

        ### DISPLAY setting
        self.ANIMATE_EPISODES = False  # False True
        self.SHOW_EPISODE_PLOTS = False
        self.SAVE_EPISODE_PLOTS = False
        self.PLOT_CIRCLES_ALONG_TRAJ = False
        lrn = self.CIRCLE_RADIUS * 1.8  # 1.2
        self.PLT_LIMITS = ((-lrn, lrn), (-lrn, lrn))
        self.PLT_FIG_SIZE = (8, 8)

        self.NEAR_GOAL_THRESHOLD = 0.2
        self.AGENT_SIZE_RANGE = (0.1, 0.5)
        self.SENSING_HORIZON = 6

        self.NORMALIZE_INPUT = True
        self.FIRST_STATE_INDEX = 1 if not self.USING_LASER else 0
        self.HOST_AGENT_OBSERVATION_LENGTH = 4  # dist to goal, heading to goal, pref speed, radius
        self.OTHER_AGENT_OBSERVATION_LENGTH = 7  # other px, other py, other vx, other vy, other radius, combined radius, distance between
        self.OTHER_AGENT_FULL_OBSERVATION_LENGTH = self.OTHER_AGENT_OBSERVATION_LENGTH
        self.HOST_AGENT_STATE_SIZE = self.HOST_AGENT_OBSERVATION_LENGTH

        if self.RESCALE_STATE:
            self.STATE_INFO_DICT['other_agent_states'] = {
                'dtype': np.float32,
                'size': 7,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("other_agent_states")',
                'std': np.array([5.0, 5.0, 1.0, 1.0, 0.5, 5.0, 5.0], dtype=np.float32),
                'mean': np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32)
            }
            self.STATE_INFO_DICT['other_agents_states'] = {
                'dtype': np.float32,
                'size': (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 7),
                'bounds': [-np.inf, np.inf],
                'attr': 'get_sensor_data("other_agents_states")',
                'std': np.tile(np.array([5.0, 5.0, 1.0, 1.0, 0.5, 5.0, 5.0], dtype=np.float32),
                               (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
                'mean': np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32),
                                (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
            }
            self.STATE_INFO_DICT['radius'] = {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("radius")',
                'std': np.array([0.5], dtype=np.float32),
                'mean': np.array([0.25], dtype=np.float32)
            }
            self.setup_obs()
            self.OTHER_AGENTS_INIT_STATE = np.array([[-10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 10.0]])

        self.NN_INPUT_AVG_VECTOR = np.array([])
        self.NN_INPUT_STD_VECTOR = np.array([])
        self.NN_INPUT_SIZE = 0
        for state in self.STATES_IN_OBS:
            if state not in self.STATES_NOT_USED_IN_POLICY:
                self.NN_INPUT_SIZE += np.product(self.STATE_INFO_DICT[state]['size'])
                self.NN_INPUT_AVG_VECTOR = np.hstack(
                    [self.NN_INPUT_AVG_VECTOR, self.STATE_INFO_DICT[state]['mean'].flatten()])
                self.NN_INPUT_STD_VECTOR = np.hstack(
                    [self.NN_INPUT_STD_VECTOR, self.STATE_INFO_DICT[state]['std'].flatten()])

        if not self.Train_Flag:
            self.CIRCLE_RADIUS = 5  # 9 # self.MAX_NUM_AGENTS_IN_ENVIRONMENT*0.4+2.0
            self.TERMINAL_STEP = 150  # 150 # int(self.CIRCLE_RADIUS*15)#150
            ### DISPLAY
            self.ANIMATE_EPISODES = True  # False True
            self.SHOW_EPISODE_PLOTS = True
            self.SAVE_EPISODE_PLOTS = True
            self.PLOT_CIRCLES_ALONG_TRAJ = False
            lrn = self.CIRCLE_RADIUS * 1.2
            self.PLT_LIMITS = ((-lrn, lrn), (-lrn, lrn))
            self.PLT_FIG_SIZE = (8, 8)


class Ga3cConfig(Config):
    def __init__(self):
        ### PARAMETERS THAT OVERWRITE/IMPACT THE ENV'S PARAMETERS
        if not hasattr(self, "MAX_NUM_AGENTS_IN_ENVIRONMENT"):
            self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 4
        if not hasattr(self, "MAX_NUM_AGENTS_TO_SIM"):
            self.MAX_NUM_AGENTS_TO_SIM = 4

        # self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed', 'radius', 'laserscan']
        self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed',
                              'radius', 'other_agents_states']
        self.STATES_NOT_USED_IN_POLICY = ['is_learning']

        self.MULTI_AGENT_ARCH_RNN, self.MULTI_AGENT_ARCH_WEIGHT_SHARING, self.MULTI_AGENT_ARCH_LASERSCAN = range(3)
        self.MULTI_AGENT_ARCH = self.MULTI_AGENT_ARCH_RNN

        if self.MULTI_AGENT_ARCH == self.MULTI_AGENT_ARCH_WEIGHT_SHARING:
            self.MAX_NUM_OTHER_AGENTS_OBSERVED = 7
        elif self.MULTI_AGENT_ARCH in [self.MULTI_AGENT_ARCH_RNN, self.MULTI_AGENT_ARCH_LASERSCAN]:
            self.MAX_NUM_OTHER_AGENTS_OBSERVED = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1

        ### INITIALIZE THE ENVIRONMENT'S PARAMETERS
        Config.__init__(self)

        ### GENERAL PARAMETERS
        self.game_grid, self.game_ale, self.game_collision_avoidance = range(3)  # Initialize game types as enum
        self.GAME_CHOICE = self.game_collision_avoidance  # Game choice: Either "game_grid" or "game_ale" or "game_collision_avoidance"
        self.USE_WANDB = False
        self.WANDB_PROJECT_NAME = "ga3c_cadrl"
        self.DEBUG = False  # Enable debug (prints more information for debugging purpose)
        self.RANDOM_SEED_1000 = 0  # np.random.seed(this * 1000 + env_id)

        ### OBSERVATIONS
        self.USE_IMAGE = False  # Enable image input
        self.NN_INPUT_AVG_VECTOR = np.array([])
        self.NN_INPUT_STD_VECTOR = np.array([])
        self.NN_INPUT_SIZE = 0
        for state in self.STATES_IN_OBS:
            if state not in self.STATES_NOT_USED_IN_POLICY:
                self.NN_INPUT_SIZE += np.product(self.STATE_INFO_DICT[state]['size'])
                self.NN_INPUT_AVG_VECTOR = np.hstack(
                    [self.NN_INPUT_AVG_VECTOR, self.STATE_INFO_DICT[state]['mean'].flatten()])
                self.NN_INPUT_STD_VECTOR = np.hstack(
                    [self.NN_INPUT_STD_VECTOR, self.STATE_INFO_DICT[state]['std'].flatten()])
        self.FIRST_STATE_INDEX = 1
        self.HOST_AGENT_OBSERVATION_LENGTH = 4  # dist to goal, heading to goal, pref speed, radius
        self.OTHER_AGENT_OBSERVATION_LENGTH = 7  # other px, other py, other vx, other vy, other radius, combined radius, distance between
        self.OTHER_AGENT_FULL_OBSERVATION_LENGTH = self.OTHER_AGENT_OBSERVATION_LENGTH
        self.HOST_AGENT_STATE_SIZE = self.HOST_AGENT_OBSERVATION_LENGTH

        ### ACTIONS
        self.NUM_ACTIONS = 11

        self.LOAD_RL_THEN_TRAIN_RL, self.TRAIN_ONLY_REGRESSION, self.LOAD_REGRESSION_THEN_TRAIN_RL = range(3)

        ### NETWORK
        self.NET_ARCH = 'NetworkVP_rnn'  # Neural net architecture
        self.ALL_ARCHS = ['NetworkVP_rnn']  # Can add more model types here
        self.NORMALIZE_INPUT = True
        self.USE_DROPOUT = False
        self.USE_REGULARIZATION = True

        #########################################################################
        # NUMBER OF AGENTS, PREDICTORS, TRAINERS, AND OTHER SYSTEM SETTINGS
        # IF THE DYNAMIC CONFIG IS ON, THESE ARE THE INITIAL VALUES
        self.AGENTS = 32  # Number of Agents
        self.PREDICTORS = 2  # Number of Predictors
        self.TRAINERS = 2  # Number of Trainers
        self.DEVICE = '/cpu:0'  # '/cpu:0' # Device '/gpu:0'
        self.DYNAMIC_SETTINGS = False  # Enable the dynamic adjustment (+ waiting time to start it)
        self.DYNAMIC_SETTINGS_STEP_WAIT = 20
        self.DYNAMIC_SETTINGS_INITIAL_WAIT = 10

        #########################################################################
        # ALGORITHM PARAMETER
        self.DISCOUNT = 0.97  # Discount factor
        self.TIME_MAX = int(4 / self.DT)  # Tmax
        self.MAX_QUEUE_SIZE = 100  # Max size of the queue
        self.PREDICTION_BATCH_SIZE = 128
        self.MIN_POLICY = 0.0  # Minimum policy

        # OPTIMIZER PARAMETERS
        self.OPT_RMSPROP, self.OPT_ADAM = range(2)  # Initialize optimizer types as enum
        self.OPTIMIZER = self.OPT_ADAM  # Game choice: Either "game_grid" or "game_ale"
        self.LEARNING_RATE_RL_START = 2e-5  # Learning rate
        self.LEARNING_RATE_RL_END = 2e-5  # Learning rate
        self.RMSPROP_DECAY = 0.99
        self.RMSPROP_MOMENTUM = 0.0
        self.RMSPROP_EPSILON = 0.1
        self.BETA_START = 1e-4  # Entropy regularization hyper-parameter
        self.BETA_END = 1e-4
        self.USE_GRAD_CLIP = False  # Gradient clipping
        self.GRAD_CLIP_NORM = 40.0
        self.LOG_EPSILON = 1e-6  # Epsilon (regularize policy lag in GA3C)
        self.TRAINING_MIN_BATCH_SIZE = 100  # Training min batch size - increasing the batch size increases the stability of the algorithm, but make learning slower

        #########################################################################
        # LOG AND SAVE
        self.TENSORBOARD = True  # Enable TensorBoard
        self.TENSORBOARD_UPDATE_FREQUENCY = 100  # Update TensorBoard every X training steps
        self.SAVE_MODELS = True  # Enable to save models every SAVE_FREQUENCY episodes
        self.SAVE_FREQUENCY = 10000  # 50000 # Save every SAVE_FREQUENCY episodes
        self.SPECIAL_EPISODES_TO_SAVE = []  # Save these episode numbers, in addition to ad SAVE_FREQUENCY
        self.PRINT_STATS_FREQUENCY = 1  # Print stats every PRINT_STATS_FREQUENCY episodes
        self.STAT_ROLLING_MEAN_WINDOW = 1000  # The window to average stats
        self.RESULTS_FILENAME = 'results.txt'  # Results filename
        self.NETWORK_NAME = 'network'  # Network checkpoint name

        self.train_plat = 'torch'

        self.AMPL_2GOAL_DICT = {0: 0.00,}
    def set_ampl_2GOAL(self,v):
        self.AMPL_GETING_CLOSE2GOAL = v
        print(" Reset Reward dis to goal ampl as:{}".format(self.AMPL_GETING_CLOSE2GOAL))


class TrainPhase1_GA3C(Ga3cConfig):
    def __init__(self):
        num_agents = 4
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = num_agents
        self.MAX_NUM_AGENTS_TO_SIM = num_agents
        Ga3cConfig.__init__(self)
        self.TRAIN_VERSION = self.LOAD_REGRESSION_THEN_TRAIN_RL

        if self.MULTI_AGENT_ARCH == self.MULTI_AGENT_ARCH_RNN:
            self.LOAD_FROM_WANDB_RUN_ID = 'run-rnn'
        elif self.MULTI_AGENT_ARCH == self.MULTI_AGENT_ARCH_WEIGHT_SHARING:
            self.LOAD_FROM_WANDB_RUN_ID = 'run-ws-' + str(self.MAX_NUM_OTHER_AGENTS_OBSERVED + 1)
        self.EPISODE_NUMBER_TO_LOAD = 0

        self.EPISODES = 1500000  # Total number of episodes and annealing frequency
        self.ANNEALING_EPISODE_COUNT = 1500000

        self.SPECIAL_EPISODES_TO_SAVE = [1490000, 1500000]
        self.LOAD_EPISODE = "00000000"

        self.ALL_ARCHS = ['NetworkVP_rnn', 'TorchBase']
        self.NET_ARCH = 'TorchBase'
        self.NETWORK_NAME = "SpikGTrNet"
        self.AGENTS = 32  # Number of Agents
        self.SAVE_FREQUENCY = 50000
        self.DEVICE = 'cuda:0'  # 'cpu'

        self.LEARNING_RATE_RL_START = 2e-5  # 2e-5 Learning rate
        self.LEARNING_RATE_RL_END = 2e-5  # 2e-5 Learning rate

        self.AMPL_2GOAL_DICT = {0: 0.00,
                                1: 0.20,
                                10000: 0.15,
                                20000: 0.10,
                                30000: 0.05,
                                40000: 0.00, }

        self.OTHER_AGENTS_INIT_STATE = np.array([[-10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 10.0]])


        # self.AMPL_2GOAL_DICT = {0:     0.00,
        #                         1:     0.20,
        #                         10000: 0.15,
        #                         20000: 0.10,
        #                         40000: 0.05,
        #                         50000: 0.01,
        #                         60000: 0.00,}

class TrainPhase2_GA3C(Ga3cConfig):
    def __init__(self):
        num_agents = 10
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = num_agents
        self.MAX_NUM_AGENTS_TO_SIM = num_agents
        Ga3cConfig.__init__(self)
        self.EPISODES                = 2500000 #2000000
        self.ANNEALING_EPISODE_COUNT = 2500000 #2000000
        self.TRAIN_VERSION = self.LOAD_RL_THEN_TRAIN_RL

        self.LOAD_FROM_WANDB_RUN_ID = 'run-20200324_221727-2tz70xqi'
        self.EPISODE_NUMBER_TO_LOAD        = 1490000
        self.SPECIAL_EPISODES_TO_SAVE = [1990000, 2000000]

        self.SPECIAL_EPISODES_TO_SAVE = [1990000, 2000000]
        self.LOAD_EPISODE = "01490000"
        # self.LOAD_EPISODE = "01990000"


        self.ALL_ARCHS = ['NetworkVP_rnn', 'TorchBase']
        self.NET_ARCH = 'TorchBase'
        self.NETWORK_NAME = "LSTMNet"
        self.AGENTS = 32  # Number of Agents
        self.SAVE_FREQUENCY = 50000
        self.DEVICE = 'cuda'  # 'cpu'

        self.LEARNING_RATE_RL_START = 2e-5  # 2e-5 Learning rate
        self.LEARNING_RATE_RL_END = 2e-5  # 2e-5 Learning rate

        self.OTHER_AGENTS_INIT_STATE = np.array([[-10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 10.0]])


class TrainPhase_PPOS(Ga3cConfig):
    def __init__(self):
        num_agents = 10 #4 finished then change to 10
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = num_agents
        self.MAX_NUM_AGENTS_TO_SIM = num_agents
        Ga3cConfig.__init__(self)

        self.SAVE_FREQUENCY = 10000
        self.DEVICE = 'cuda:0'  # 'cpu'

        self.LEARNING_RATE_RL_START = 2e-5  # 2e-5 Learning rate
        self.LEARNING_RATE_RL_END = 2e-5  # 2e-5 Learning rate

        self.USING_MAX_HEADCHANGE = True

        self.ALL_ACT_ARCHS = ['LSTMActor',
                              'SpikLSTMActor',
                              'GTrActor',
                              'SpikGTrActor',
                              ]
        self.ALL_CRT_ARCHS = ['LSTMCritic',
                              'GTrCritic',
                              ]
        self.ACT_ARCH = 'SpikLSTMActor'#'SpikLSTMActor'
        self.CRT_ARCH = 'LSTMCritic'


        self.AGENTS_STATES_TO_SIM = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
        self.OTHER_AGENTS_INIT_STATE = np.array([[-10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 10.0]])
        Config.USING_LASER=False

        self.TEST_CASE_ARGS = {
            'policy_to_ensure': 'learning_ppos',
            'policies': ['noncoop', 'learning_ppos', 'static'],
            'policy_distr': [0.05, 0.9, 0.05],
            'speed_bnds': [0.5, 2.0],
            'radius_bnds': [0.2, 0.8],
            'side_length': [
                {'num_agents': [0,5], 'side_length': [4,5]},
                {'num_agents': [5,np.inf], 'side_length': [6,8]},
                ],
            # 'agents_sensors': ['other_agents_states_encoded'],
        }

        self.TRAIN_SINGLE_AGENT = False
        self.MAX_TIME_RATIO = 2.5

        self.NEW_REWARD_TYPE = True
        self.REWARD_AT_GOAL = 1.0 #15
        self.REWARD_COLLISION_WITH_AGENT = -1.0 #-15 -0.25
        self.REWARD_COLLISION_WITH_WALL = -1.0 #-15 -0.25
        self.GETTING_CLOSE_RANGE = 0.1
        self.DISCRETE_CONTROL_FLAG = True

class TrainPhase_PPOS_STAGE1(Ga3cConfig):
    def __init__(self):
        num_agents = 4 #4 finished then change to 10
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = num_agents
        self.MAX_NUM_AGENTS_TO_SIM = num_agents
        Ga3cConfig.__init__(self)

        self.SAVE_FREQUENCY = 10000
        self.DEVICE = 'cuda:0'  # 'cpu'

        self.LEARNING_RATE_RL_START = 2e-5  # 2e-5 Learning rate
        self.LEARNING_RATE_RL_END = 2e-5  # 2e-5 Learning rate

        self.USING_MAX_HEADCHANGE = True

        self.ALL_ACT_ARCHS = ['LSTMActor',
                              'SpikLSTMActor',
                              'GTrActor',
                              'SpikGTrActor',
                              ]
        self.ALL_CRT_ARCHS = ['LSTMCritic',
                              'GTrCritic',
                              ]
        self.ACT_ARCH = 'LSTMActor'  # SpikGMFActor SpikGTrANActor SpikGTrMeanActor GTrMeanActor,
        self.CRT_ARCH = 'LSTMCritic'  # LSTMCritic GTrMeanCritic

        self.AGENTS_STATES_TO_SIM = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
        self.OTHER_AGENTS_INIT_STATE = np.array([[-10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 10.0]])
        Config.USING_LASER=False

        self.TEST_CASE_ARGS = {
            'policy_to_ensure': 'learning_ppos',
            'policies': ['noncoop', 'learning_ppos', 'static'],
            'policy_distr': [0.05, 0.9, 0.05],
            'speed_bnds': [0.5, 2.0],
            'radius_bnds': [0.2, 0.8],
            'side_length': [
                {'num_agents': [0,5], 'side_length': [4,5]},
                {'num_agents': [5,np.inf], 'side_length': [6,8]},
                ],
            # 'agents_sensors': ['other_agents_states_encoded'],
        }

        self.TRAIN_SINGLE_AGENT = False
        self.MAX_TIME_RATIO = 2.5

        self.NEW_REWARD_TYPE = True
        self.REWARD_AT_GOAL = 1.0 #15
        self.REWARD_COLLISION_WITH_AGENT = -1.0 #-15 -0.25
        self.REWARD_COLLISION_WITH_WALL = -1.0 #-15 -0.25
        self.GETTING_CLOSE_RANGE = 0.1
        self.DISCRETE_CONTROL_FLAG = True

class TrainPhase_PPOS_STAGE2(Ga3cConfig):
    def __init__(self):
        num_agents = 10 #4 finished then change to 10
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = num_agents
        self.MAX_NUM_AGENTS_TO_SIM = num_agents
        Ga3cConfig.__init__(self)

        self.SAVE_FREQUENCY = 10000
        self.DEVICE = 'cuda:0'  # 'cpu'

        self.LEARNING_RATE_RL_START = 2e-5  # 2e-5 Learning rate
        self.LEARNING_RATE_RL_END = 2e-5  # 2e-5 Learning rate

        self.USING_MAX_HEADCHANGE = True

        self.ALL_ACT_ARCHS = ['LSTMActor',
                              'SpikLSTMActor',
                              'GTrActor',
                              'SpikGTrActor',
                              ]
        self.ALL_CRT_ARCHS = ['LSTMCritic',
                              'GTrCritic',
                              ]

        self.ACT_ARCH = 'LSTMActor'  # SpikGMFActor SpikGTrANActor SpikGTrMeanActor GTrMeanActor,
        self.CRT_ARCH = 'LSTMCritic'  # LSTMCritic GTrMeanCritic

        self.AGENTS_STATES_TO_SIM = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
        self.OTHER_AGENTS_INIT_STATE = np.array([[-10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 10.0]])
        Config.USING_LASER=False

        self.TEST_CASE_ARGS = {
            'policy_to_ensure': 'learning_ppos',
            'policies': ['noncoop', 'learning_ppos', 'static'],
            'policy_distr': [0.05, 0.9, 0.05],
            'speed_bnds': [0.5, 2.0],
            'radius_bnds': [0.2, 0.8],
            'side_length': [
                {'num_agents': [0,5], 'side_length': [4,5]},
                {'num_agents': [5,np.inf], 'side_length': [6,8]},
                ],
            # 'agents_sensors': ['other_agents_states_encoded'],
        }

        self.TRAIN_SINGLE_AGENT = False
        self.MAX_TIME_RATIO = 2.5

        self.NEW_REWARD_TYPE = True
        self.REWARD_AT_GOAL = 1.0 #15
        self.REWARD_COLLISION_WITH_AGENT = -1.0 #-15 -0.25
        self.REWARD_COLLISION_WITH_WALL = -1.0 #-15 -0.25
        self.GETTING_CLOSE_RANGE = 0.1
        self.DISCRETE_CONTROL_FLAG = True





# continuous_action_space
class TrainPhase_PPOS_CAS(Ga3cConfig):
    def __init__(self):
        num_agents = 4 #4 finished then change to 10
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = num_agents
        self.MAX_NUM_AGENTS_TO_SIM = num_agents
        Ga3cConfig.__init__(self)

        self.SAVE_FREQUENCY = 10000
        self.DEVICE = 'cuda:0'  # 'cpu'

        self.LEARNING_RATE_RL_START = 2e-5  # 2e-5 Learning rate
        self.LEARNING_RATE_RL_END = 2e-5  # 2e-5 Learning rate

        self.USING_MAX_HEADCHANGE = True

        self.ALL_ACT_ARCHS = ['LSTMActor',
                              'SpikLSTMActor',
                              'GTrActor',
                              'SpikGTrActor',
                              ]
        self.ALL_CRT_ARCHS = ['LSTMCritic',
                              'GTrCritic',
                              ]
        self.ACT_ARCH = 'LSTMActor'#'SpikLSTMActor'
        self.CRT_ARCH = 'LSTMCritic'


        self.AGENTS_STATES_TO_SIM = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
        self.OTHER_AGENTS_INIT_STATE = np.array([[-10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 10.0]])
        Config.USING_LASER=False

        self.TEST_CASE_ARGS = {
            'policy_to_ensure': 'learning_ppos',
            'policies': ['noncoop', 'learning_ppos', 'static'],
            'policy_distr': [0.05, 0.9, 0.05],
            'speed_bnds': [0.5, 2.0],
            'radius_bnds': [0.2, 0.8],
            'side_length': [
                {'num_agents': [0,5], 'side_length': [4,5]},
                {'num_agents': [5,np.inf], 'side_length': [6,8]},
                ],
            # 'agents_sensors': ['other_agents_states_encoded'],
        }

        self.TRAIN_SINGLE_AGENT = False
        self.MAX_TIME_RATIO = 2.5

        self.NEW_REWARD_TYPE = True
        self.REWARD_AT_GOAL = 1.0 #15
        self.REWARD_COLLISION_WITH_AGENT = -1.0 #-15 -0.25
        self.REWARD_COLLISION_WITH_WALL = -1.0 #-15 -0.25
        self.GETTING_CLOSE_RANGE = 0.1
        self.DISCRETE_CONTROL_FLAG = False

# Maze Navigation
class TrainPhase_Maze_PPO(Config):
    def __init__(self):
        self.RESCALE_STATE = True

        self.USING_LASER = True

        self.IS_FILTER_INDEX = False

        self.Train_Flag = False  # True False

        self.TRAIN_SINGLE_AGENT = False
        self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed',
                              'radius', 'other_agents_states']
        self.SENSING_TYPR = np.inf
        if self.USING_LASER:
            self.STATES_IN_OBS = ['is_learning', 'dist_to_goal', 'heading_ego_frame', 'pref_speed',
                                  'radius', 'laserscan']
            self.USE_STATIC_MAP = True

        self.STATES_NOT_USED_IN_POLICY = ['is_learning']

        num_agents = 4
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = num_agents  # 4
        self.MAX_NUM_AGENTS_TO_SIM = num_agents  # 4
        self.AGENTS_STATES_TO_SIM = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
        self.AGENTS_TO_Train = num_agents

        Config.__init__(self)

        self.ALL_ARCHS=['LSTMPolicy',
                        'GTrPolicy',
                        'FixLengthPolicy',
                        'ScanPolicy',

                        'ScanConvPolicy',
                        'ScanGTrPolicy',]

        SCAN_ARCH= 'ScanGTrPolicy'
        self.NET_ARCH = 'FixLengthPolicy' if not self.USING_LASER else SCAN_ARCH

        #Reward setting
        self.REWARD_AT_GOAL = 10 #15
        self.REWARD_COLLISION_WITH_AGENT = -10#-15 -0.25
        self.REWARD_COLLISION_WITH_WALL = -10 #-15 -0.25
        self.REWARD_GETTING_CLOSE = -1.0#-0.1  # reward when agent gets close to another agent (unused?)
        self.REWARD_ENTERED_NORM_ZONE = -0.5#-0.05

        self.AMPL_GETING_CLOSE2GOAL=1.0 #1.0 #1.5
        self.AMPL_GETING_CLOSE2AGENTS = 1.2

        self.COLLISION_DIST = 0.0  # meters between agents' boundaries for collision
        self.GETTING_CLOSE_RANGE = 0.2  # meters between agents' boundaries for collision

        # SIM step length setting
        self.MAX_TIME_RATIO = 10 #8
        self.TERMINAL_STEP = 200 #150

        # env range  setting
        self.CIRCLE_RADIUS = 5
        self.AGENT_RESET_TYPE = 'random' # noCollision random

        ### DISPLAY setting
        self.ANIMATE_EPISODES = False  # False True
        self.SHOW_EPISODE_PLOTS = False
        self.SAVE_EPISODE_PLOTS = False
        self.PLOT_CIRCLES_ALONG_TRAJ = False
        lrn = self.CIRCLE_RADIUS * 1.8  # 1.2
        self.PLT_LIMITS = ((-lrn, lrn), (-lrn, lrn))
        self.PLT_FIG_SIZE = (8, 8)

        self.NEAR_GOAL_THRESHOLD = 0.2
        self.AGENT_SIZE_RANGE = (0.1, 0.5)
        self.SENSING_HORIZON = 6

        self.NORMALIZE_INPUT = True
        self.FIRST_STATE_INDEX = 1 if not self.USING_LASER else 0
        self.HOST_AGENT_OBSERVATION_LENGTH = 4  # dist to goal, heading to goal, pref speed, radius
        self.OTHER_AGENT_OBSERVATION_LENGTH = 7  # other px, other py, other vx, other vy, other radius, combined radius, distance between
        self.OTHER_AGENT_FULL_OBSERVATION_LENGTH = self.OTHER_AGENT_OBSERVATION_LENGTH
        self.HOST_AGENT_STATE_SIZE = self.HOST_AGENT_OBSERVATION_LENGTH

        if self.RESCALE_STATE:
            self.STATE_INFO_DICT['other_agent_states'] = {
                'dtype': np.float32,
                'size': 7,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("other_agent_states")',
                'std': np.array([5.0, 5.0, 1.0, 1.0, 0.5, 5.0, 5.0], dtype=np.float32),
                'mean': np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32)
            }
            self.STATE_INFO_DICT['other_agents_states'] = {
                'dtype': np.float32,
                'size': (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 7),
                'bounds': [-np.inf, np.inf],
                'attr': 'get_sensor_data("other_agents_states")',
                'std': np.tile(np.array([5.0, 5.0, 1.0, 1.0, 0.5, 5.0, 5.0], dtype=np.float32),
                               (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
                'mean': np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32),
                                (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
            }
            self.STATE_INFO_DICT['radius'] = {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("radius")',
                'std': np.array([0.5], dtype=np.float32),
                'mean': np.array([0.25], dtype=np.float32)
            }
            self.setup_obs()
            self.OTHER_AGENTS_INIT_STATE = np.array([[-10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 10.0]])

        self.NN_INPUT_AVG_VECTOR = np.array([])
        self.NN_INPUT_STD_VECTOR = np.array([])
        self.NN_INPUT_SIZE = 0
        for state in self.STATES_IN_OBS:
            if state not in self.STATES_NOT_USED_IN_POLICY:
                self.NN_INPUT_SIZE += np.product(self.STATE_INFO_DICT[state]['size'])
                self.NN_INPUT_AVG_VECTOR = np.hstack(
                    [self.NN_INPUT_AVG_VECTOR, self.STATE_INFO_DICT[state]['mean'].flatten()])
                self.NN_INPUT_STD_VECTOR = np.hstack(
                    [self.NN_INPUT_STD_VECTOR, self.STATE_INFO_DICT[state]['std'].flatten()])

        if not self.Train_Flag:
            self.CIRCLE_RADIUS = 5  # 9 # self.MAX_NUM_AGENTS_IN_ENVIRONMENT*0.4+2.0
            self.TERMINAL_STEP = 150  # 150 # int(self.CIRCLE_RADIUS*15)#150
            ### DISPLAY
            self.ANIMATE_EPISODES = True  # False True
            self.SHOW_EPISODE_PLOTS = True
            self.SAVE_EPISODE_PLOTS = True
            self.PLOT_CIRCLES_ALONG_TRAJ = False
            lrn = self.CIRCLE_RADIUS * 1.2
            self.PLT_LIMITS = ((-lrn, lrn), (-lrn, lrn))
            self.PLT_FIG_SIZE = (8, 8)

class TrainPhase_DDPG_Maze(Config):
    def __init__(self):
        self.RESCALE_STATE = False

        self.USING_LASER = True  # True False

        self.IS_FILTER_INDEX = False

        self.Train_Flag = True  # True False

        self.TRAIN_SINGLE_AGENT = False
        self.STATES_IN_OBS = ['is_learning', 'num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed',
                              'radius', 'other_agents_states']
        self.SENSING_TYPR = np.inf
        if self.USING_LASER:
            self.STATES_IN_OBS = ['is_learning', 'dist_to_goal', 'heading_ego_frame', 'vel_ego_frame','pref_speed',
                                  'radius', 'laserscan']
            self.USE_STATIC_MAP = True

        self.STATES_NOT_USED_IN_POLICY = ['is_learning']

        num_agents = 10
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = num_agents  # 4
        self.MAX_NUM_AGENTS_TO_SIM = num_agents  # 4
        self.AGENTS_STATES_TO_SIM = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
        self.AGENTS_TO_Train = num_agents

        Config.__init__(self)


        self.USE_STATIC_MAP = True

        self.USING_MAX_HEADCHANGE = True
        self.USING_EXTERNAL_TRAIN_CASES = True

        self.ALL_ACT_ARCHS = ['LSTM_ActorNet',
                              'SpikLSTM_ActorNet',
                              'GTr_ActorNet',
                              'SpikGTr_ActorNet',
                              'GO_SpikGTr_ActorNet',
                              'SDDPG_ActorNet',
                              ]

        self.ALL_CRT_ARCHS = ['LSTM_CriticNet',
                              'GTr_CriticNet',
                              'DDPG_CriticNet',]

        SCAN_ACT_ARCH ='SDDPG_ActorNet'   # None 'ScanConvPolicy' 'SDDPG_ActorNet'
        SCAN_CRT_ARCH ='DDPG_CriticNet'   # None 'ScanConvPolicy' 'DDPG_CriticNet'
        self.ACT_ARCH = 'LSTM_ActorNet' if not self.USING_LASER else SCAN_ACT_ARCH
        self.CRT_ARCH = 'LSTM_CriticNet' if not self.USING_LASER else SCAN_CRT_ARCH

        # Reward setting
        # self.REWARD_TIME_STEP = 0.0  # default reward given if none of the others apply (encourage speed)
        # self.REWARD_WIGGLY_BEHAVIOR = 0.0
        # self.WIGGLY_BEHAVIOR_THRESHOLD = np.inf

        self.REWARD_AT_GOAL = 1.0  # 10 #15
        self.REWARD_COLLISION_WITH_AGENT = -0.25  # -10#-10#-15 -0.25
        self.REWARD_COLLISION_WITH_WALL = -0.25  # -10#-10 #-15 -0.25
        self.REWARD_GETTING_CLOSE = -0.1  # -0.1  # reward when agent gets close to another agent (unused?)
        self.REWARD_ENTERED_NORM_ZONE = -0.05  # -0.05

        self.AMPL_GETING_CLOSE2GOAL = 0.2  # 1.0 #1.5
        # self.AMPL_GETING_CLOSE2AGENTS = 1.2

        self.COLLISION_DIST = 0.0  # meters between agents' boundaries for collision
        self.GETTING_CLOSE_RANGE = 0.2  # meters between agents' boundaries for collision

        # SIM step length setting
        self.MAX_TIME_RATIO = 25
        self.TERMINAL_STEP = 1000  # 1000

        # env range  setting
        self.CIRCLE_RADIUS = 5
        self.AGENT_RESET_TYPE = 'noCollision'  # noCollision random

        ### DISPLAY setting
        self.ANIMATE_EPISODES = False  # False True
        self.SHOW_EPISODE_PLOTS = False
        self.SAVE_EPISODE_PLOTS = False
        self.PLOT_CIRCLES_ALONG_TRAJ = False
        lrn = self.CIRCLE_RADIUS * 1.8  # 1.2
        self.PLT_LIMITS = ((-lrn, lrn), (-lrn, lrn))
        self.PLT_FIG_SIZE = (8, 8)

        self.NEAR_GOAL_THRESHOLD = 0.2
        self.AGENT_SIZE_RANGE = (0.1, 0.5)
        self.SENSING_HORIZON = 6

        self.NORMALIZE_INPUT = True
        self.FIRST_STATE_INDEX = 1 if not self.USING_LASER else 0
        self.HOST_AGENT_OBSERVATION_LENGTH = 4  # dist to goal, heading to goal, pref speed, radius
        self.OTHER_AGENT_OBSERVATION_LENGTH = 7  # other px, other py, other vx, other vy, other radius, combined radius, distance between
        self.OTHER_AGENT_FULL_OBSERVATION_LENGTH = self.OTHER_AGENT_OBSERVATION_LENGTH
        self.HOST_AGENT_STATE_SIZE = self.HOST_AGENT_OBSERVATION_LENGTH

        if self.RESCALE_STATE:
            self.STATE_INFO_DICT['other_agent_states'] = {
                'dtype': np.float32,
                'size': 7,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("other_agent_states")',
                'std': np.array([5.0, 5.0, 1.0, 1.0, 0.5, 5.0, 5.0], dtype=np.float32),
                'mean': np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32)
            }
            self.STATE_INFO_DICT['other_agents_states'] = {
                'dtype': np.float32,
                'size': (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 7),
                'bounds': [-np.inf, np.inf],
                'attr': 'get_sensor_data("other_agents_states")',
                'std': np.tile(np.array([5.0, 5.0, 1.0, 1.0, 0.5, 5.0, 5.0], dtype=np.float32),
                               (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
                'mean': np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32),
                                (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
            }
            self.STATE_INFO_DICT['radius'] = {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("radius")',
                'std': np.array([0.5], dtype=np.float32),
                'mean': np.array([0.25], dtype=np.float32)
            }
            self.setup_obs()
        self.OTHER_AGENTS_INIT_STATE = np.array([[-10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 10.0]])

        self.NN_INPUT_AVG_VECTOR = np.array([])
        self.NN_INPUT_STD_VECTOR = np.array([])
        self.NN_INPUT_SIZE = 0
        for state in self.STATES_IN_OBS:
            if state not in self.STATES_NOT_USED_IN_POLICY:
                self.NN_INPUT_SIZE += np.product(self.STATE_INFO_DICT[state]['size'])
                self.NN_INPUT_AVG_VECTOR = np.hstack(
                    [self.NN_INPUT_AVG_VECTOR, self.STATE_INFO_DICT[state]['mean'].flatten()])
                self.NN_INPUT_STD_VECTOR = np.hstack(
                    [self.NN_INPUT_STD_VECTOR, self.STATE_INFO_DICT[state]['std'].flatten()])

        if not self.Train_Flag:
            self.CIRCLE_RADIUS = 6  # 9 # self.MAX_NUM_AGENTS_IN_ENVIRONMENT*0.4+2.0
            self.TERMINAL_STEP = 150  # 150 # int(self.CIRCLE_RADIUS*15)#150
            ### DISPLAY
            self.ANIMATE_EPISODES = True  # False True
            self.SHOW_EPISODE_PLOTS = True
            self.SAVE_EPISODE_PLOTS = True
            self.PLOT_CIRCLES_ALONG_TRAJ = False
            lrn = self.CIRCLE_RADIUS * 1.2
            self.PLT_LIMITS = ((-lrn, lrn), (-lrn, lrn))
            self.PLT_FIG_SIZE = (8, 8)

    def set_agents_num(self, num):
        self.AMPL_GETING_CLOSE2GOAL = self.AMPL_GETING_CLOSE2GOAL * 0.5

        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = num
        self.MAX_NUM_AGENTS_TO_SIM = num
        self.AGENTS_STATES_TO_SIM = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
        self.AGENTS_TO_Train = num

        self.MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
        self.MAX_NUM_OTHER_AGENTS_OBSERVED = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1

        self.STATE_INFO_DICT = {
            'dist_to_goal': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("dist_to_goal")',
                'std': np.array([5.], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
            },
            'radius': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("radius")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([0.5], dtype=np.float32)
            },
            'heading_ego_frame': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [-np.pi, np.pi],
                'attr': 'get_agent_data("heading_ego_frame")',
                'std': np.array([3.14], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
            },
            'speed_ego_frame': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("speed_ego_frame")',
                'std': np.array([6], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
            },
            'vel_ego_frame': {
                'dtype': np.float32,
                'size': 2,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("vel_ego_frame")',
                'std': np.array([6], dtype=np.float32),
                'mean': np.array([0.], dtype=np.float32)
            },
            'pref_speed': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("pref_speed")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([1.0], dtype=np.float32)
            },
            'num_other_agents': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("num_other_agents_observed")',
                'std': np.array([1.0], dtype=np.float32),
                'mean': np.array([1.0], dtype=np.float32)
            },
            'other_agent_states': {
                'dtype': np.float32,
                'size': 7,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("other_agent_states")',
                'std': np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32),
                'mean': np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32)
            },
            'other_agents_states': {
                'dtype': np.float32,
                'size': (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 7),
                'bounds': [-np.inf, np.inf],
                'attr': 'get_sensor_data("other_agents_states")',
                'std': np.tile(np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32),
                               (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
                'mean': np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32),
                                (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
            },
            'laserscan': {
                'dtype': np.float32,
                'size': (self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH),
                'bounds': [0., 6.],
                'attr': 'get_sensor_data("laserscan")',
                'std': 5. * np.ones((self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH), dtype=np.float32),
                'mean': 5. * np.ones((self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH), dtype=np.float32)
            },
            'is_learning': {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0., 1.],
                'attr': 'get_agent_data_equiv("policy.str", "learning")'
            },
            'other_agents_states_encoded': {
                'dtype': np.float32,
                'size': 100,
                'bounds': [0., 1.],
                'attr': 'get_sensor_data("other_agents_states_encoded")'
            }
        }
        if self.RESCALE_STATE:
            self.STATE_INFO_DICT['other_agent_states'] = {
                'dtype': np.float32,
                'size': 7,
                'bounds': [-np.inf, np.inf],
                'attr': 'get_agent_data("other_agent_states")',
                'std': np.array([5.0, 5.0, 1.0, 1.0, 0.5, 5.0, 5.0], dtype=np.float32),
                'mean': np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32)
            }
            self.STATE_INFO_DICT['other_agents_states'] = {
                'dtype': np.float32,
                'size': (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 7),
                'bounds': [-np.inf, np.inf],
                'attr': 'get_sensor_data("other_agents_states")',
                'std': np.tile(np.array([5.0, 5.0, 1.0, 1.0, 0.5, 5.0, 5.0], dtype=np.float32),
                               (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
                'mean': np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0], dtype=np.float32),
                                (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
            }
            self.STATE_INFO_DICT['radius'] = {
                'dtype': np.float32,
                'size': 1,
                'bounds': [0, np.inf],
                'attr': 'get_agent_data("radius")',
                'std': np.array([0.5], dtype=np.float32),
                'mean': np.array([0.25], dtype=np.float32)
            }
        self.setup_obs()

        self.NN_INPUT_AVG_VECTOR = np.array([])
        self.NN_INPUT_STD_VECTOR = np.array([])
        self.NN_INPUT_SIZE = 0
        for state in self.STATES_IN_OBS:
            if state not in self.STATES_NOT_USED_IN_POLICY:
                self.NN_INPUT_SIZE += np.product(self.STATE_INFO_DICT[state]['size'])
                self.NN_INPUT_AVG_VECTOR = np.hstack(
                    [self.NN_INPUT_AVG_VECTOR, self.STATE_INFO_DICT[state]['mean'].flatten()])
                self.NN_INPUT_STD_VECTOR = np.hstack(
                    [self.NN_INPUT_STD_VECTOR, self.STATE_INFO_DICT[state]['std'].flatten()])


