# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SimulatedBatchEnv in a Gym-like interface."""

#TODO(pm): do we really need these
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from tensor2tensor.utils import trainer_lib
from tensor2tensor.rl.envs.simulated_batch_env import SimulatedBatchEnv
import tensorflow as tf
from gym import Env


class Dumper(object):

  def __init__(self, batch_env, quiet_exit_on_step=None, dump_lambda=None):
    self.batch_env = batch_env
    self.batch_size = batch_env.batch_size
    self.action_space = self.batch_env.action_space
    self.observation_space = self.batch_env.observation_space
    self._index = 0
    self._quiet_exit_on_step = quiet_exit_on_step if quiet_exit_on_step is not None else -1 #-1 is effectively +infty
    self._dump_lambda = dump_lambda if dump_lambda is not None else lambda index: True

  def _send_image_to_neptune(self, obs):
    import neptune
    from PIL import Image

    pil_image = Image.fromarray(obs[0, ...])
    ctx = neptune.Context()
    ctx.channel_send('dumper_image', x=self._index,
                     y=neptune.Image(name='Frame: {}'.format(self._index),
                                   description='', data=pil_image))

  def step(self, action):
    import numpy as np
    ret = self.batch_env.step(action)
    obs, rewards, dones = ret
    if self._dump_lambda(self._index):
      np.savez("save_{}".format(self._index), obs=obs, rewards=rewards, dones=dones)
      self._send_image_to_neptune(obs)


    self._index += 1
    if self._quiet_exit_on_step is not None and self._index==self._quiet_exit_on_step:
      print("Exiting!!!!")
      exit(0)
    return ret

  def reset(self, **kwargs):
    print("Dumper reset at step:{}".format(self._index))
    return self.batch_env.reset()


class FlatBatchEnv(Env):
  def __init__(self, batch_env):
    if batch_env.batch_size != 1:
      raise ValueError("Number of environments in batch must be equal to one")
    self.batch_env = batch_env
    self.action_space = self.batch_env.action_space
    self.observation_space = self.batch_env.observation_space

  def step(self, action):
    obs, rewards, dones = self.batch_env.step([action])
    return obs[0], rewards[0], dones[0], {}

  def reset(self, **kwargs):
    print("FlatBatchenv:reset")
    return self.batch_env.reset()[0]


class SimulatedBatchGymEnv(object):
  """ SimulatedBatchEnv in a Gym-like interface.

  The environments are  batched.
  """
  def __init__(self, environment_spec, batch_size,
               model_dir=None, sess=None):
    self.batch_size = batch_size

    with tf.Graph().as_default():
      self._batch_env = SimulatedBatchEnv(environment_spec,
                                          self.batch_size)

      self.action_space = self._batch_env.action_space
      # TODO(KC): check for the stack wrapper and correct number of channels in
      # observation_space
      self.observation_space = self._batch_env.observ_space
      self._sess = sess if sess is not None else tf.Session()
      self._to_initialize = [self._batch_env]

      environment_wrappers = environment_spec.wrappers
      wrappers = copy.copy(environment_wrappers) if environment_wrappers else []

      for w in wrappers:
        self._batch_env = w[0](self._batch_env, **w[1])
        self._to_initialize.append(self._batch_env)

      self._sess.run(tf.global_variables_initializer())
      for wrapped_env in self._to_initialize:
        wrapped_env.initialize(self._sess)

      self._actions_t = tf.placeholder(shape=(1,), dtype=tf.int32)
      self._rewards_t, self._dones_t = self._batch_env.simulate(self._actions_t)
      self._obs_t = self._batch_env.observ
      self._reset_op = self._batch_env.reset(tf.constant([0], dtype=tf.int32))

      env_model_loader = tf.train.Saver(
          var_list=tf.global_variables(scope="next_frame*"))  # pylint:disable=unexpected-keyword-arg
      trainer_lib.restore_checkpoint(model_dir, saver=env_model_loader,
                                     sess=self._sess, must_restore=True)

  def render(self, mode="human"):
    raise NotImplementedError()

  def reset(self, indicies=None):
    if indicies:
      raise NotImplementedError()
    obs = self._sess.run(self._reset_op)
    # TODO(pmilos): remove if possible
    obs[:, 0, 0, 0] = 0
    obs[:, 0, 0, 1] = 255
    return obs

  def step(self, actions):
    obs, rewards, dones = self._sess.run(
        [self._obs_t, self._rewards_t, self._dones_t],
        feed_dict={self._actions_t: actions})
    return obs, rewards, dones
