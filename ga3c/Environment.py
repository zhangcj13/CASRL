# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import sys
import time
if sys.version_info >= (3,0):
    from queue import Queue
else:
    from Queue import Queue
import numpy as np
import scipy.misc as misc
from gym_collision_avoidance.envs import Config

class Environment:
    def __init__(self, id, lft_train_status=False):
        self._set_env(id)
        self.index=id
        
        self.nb_frames    = 1
        self.frame_q      = Queue(maxsize=self.nb_frames)
        self.total_reward = 0

        self.previous_state = self.current_state = None
        self.lft_train_status=lft_train_status
        # self.reset()

    def _set_env(self, id):
        if Config.GAME_CHOICE == Config.game_grid:
            self.game = Gridworld(id, Config.ENV_ROW, Config.ENV_COL, Config.PIXEL_SIZE, Config.MAX_ITER, Config.AGENT_COLOR, Config.TARGET_COLOR,
                                  Config.DISPLAY_SCREEN, Config.TIMER_DURATION, Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, Config.STACKED_FRAMES, Config.DEBUG)
        elif Config.GAME_CHOICE == Config.game_collision_avoidance:
            from gym_collision_avoidance.experiments.src.env_utils import run_episode, create_env, store_stats
            env, one_env = create_env()
            self.game = env
        else: 
            raise ValueError("[ ERROR ] Invalid choice of game. Check Config.py for choices")

    def _get_current_state(self):
        if Config.DEBUG: print('[ DEBUG ] Environment::_get_current_state()')

        if Config.GAME_CHOICE == Config.game_collision_avoidance:
            agent_states_ = np.array(self.frame_q.queue)
            return agent_states_

        else:
            if not self.frame_q.full():
                return None

            image_ = np.array(self.frame_q.queue)

            return image_

    def _update_frame_q(self, frame):
        if self.frame_q.full():
            self.frame_q.get()# Pop oldest frame
        self.frame_q.put(frame)
        if Config.DEBUG: print('[ DEBUG ] Environment::frame_q size is): {}'.format(self.frame_q.qsize()))

    def _process_obs(self, observations):
        # print("[_process_obs]")
        # print("observations: {}".format(observations))
        if observations.ndim > 2:
            observations_ = observations[0] # undo vecenv
        else:
            observations_=observations
        if observations_.ndim == 3:
            observations_ = observations_[0] # undo multiagentvecenv wrapper
        # print("observations_: {}".format(observations_))
        self.latest_observations = observations_
        if self.lft_train_status:
            self._update_frame_q(observations_[:, :])
        else:
            self._update_frame_q(observations_[:, 1:])
        self.previous_state = self.current_state
        self.current_state = self._get_current_state()
        # print("self.previous_state: {}".format(self.previous_state))
        # print("self.current_state: {}".format(self.current_state))
        # assert(0)

        # for agent_observation in observations:
        #     # only use host agent's observations for training
        #     if agent_observation[0] == 0:
        #         self._update_frame_q(agent_observation[1:])

    def reset(self):
        if Config.DEBUG: print('[ DEBUG ] Environment::reset()')
        self.total_reward = 0
        self.frame_q.queue.clear()

        observations = self.game.reset()
        self._process_obs(observations)
        # self.previous_state = self.current_state = None

    def step(self, action, pid, count,ampl2goal=None):

        if Config.DEBUG: print('[ DEBUG ] Environment::step()')

        # actions={'action':action,'ampl2goal':ampl2goal}
        if ampl2goal is not None:
            action[0]['ampl2goal'] = ampl2goal
        observations, rewards, game_over, info = self.game.step(action)

        # observations, rewards, game_over, info = self.game.step(action)
        self.total_reward += np.sum(rewards)
        self._process_obs(observations)

        return rewards, game_over, info

    def print_frame_q(self):
        return self.frame_q.qsize()
    def get_observation_list(self):
        observation_list=[]
        for i, agent_observation in enumerate(self.latest_observations):
            state = np.asarray([agent_observation])
            observation_list.append(state)
        return observation_list

