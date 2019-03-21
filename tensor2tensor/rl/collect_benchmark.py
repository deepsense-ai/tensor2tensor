# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""Collect benchmark."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver

from tensor2tensor.models.research.rl import PolicyBase
from tensor2tensor.rl import ppo
from tensor2tensor.rl import tf_new_collect
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow as tf

import functools
from gym.spaces import Discrete
from munch import Munch
from time import time


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_bool("use_tpu", False, "Whether to use TPU.")
tf.flags.DEFINE_integer("batch_size", 16, "Batch size.")
tf.flags.DEFINE_integer("epoch_length", 50, "Epoch length.")
tf.flags.DEFINE_integer("num_tpus", 8, "Number of TPUs.")

BATCH_SIZES = [16, 32, 64, 128]
EPOCH_LENGTHS = [50, 100, 200]
if FLAGS.use_tpu:
  NUMS_TPUS = [8, 4, 2, 1]
else:
  NUMS_TPUS = [0]
HISTORY = 4
TPU_NAME = "ng-tpu-01"



def run_config(num_tpus, batch_size, epoch_length):
  if FLAGS.use_tpu:
    target = TPUClusterResolver(tpu=[TPU_NAME]).get_master()
  else:
    target = ""
  if FLAGS.use_tpu:
    batch_size //= num_tpus

  ppo_hparams = trainer_lib.create_hparams("ppo_original_params")
  ppo_hparams.policy_network = "feed_forward_cnn_small_categorical_policy"
  ppo_hparams.epoch_length = epoch_length
  action_space = Discrete(2)
  with tf.variable_scope(
      "collect_{}_{}_{}".format(num_tpus, batch_size, epoch_length)
  ):
    batch_env = tf_new_collect.NewSimulatedBatchEnv(
        batch_size,
        "next_frame_basic_stochastic_discrete",
        trainer_lib.create_hparams("next_frame_basic_stochastic_discrete_long")
    )
    batch_env = tf_new_collect.NewStackWrapper(batch_env, HISTORY)
    memory = tf_new_collect.new_define_collect(
        batch_env, ppo_hparams, action_space, force_beginning_resets=True
    )
  with tf.Session(target) as sess:
    if FLAGS.use_tpu:
      topology = sess.run(tpu.initialize_system())
      memory = tpu.replicate(
          lambda: memory,
          inputs=([()] * num_tpus),
          device_assignment=tf.contrib.tpu.device_assignment(
              topology, num_replicas=num_tpus
          )
      )

    sess.run(tf.global_variables_initializer())
    # First trial is for warmup.
    sess.run(memory)
    t = time()
    sess.run(memory)

    if FLAGS.use_tpu:
      sess.run(tpu.shutdown_system())

  return time() - t


def run_configs(run_fn, **ranges):
  results = {}
  for (name, range) in ranges.items():
    print("looping over", name)
    results[name] = {}
    for value in range:
      kwargs = {
          arg_name: value if arg_name == name else ranges[arg_name][0]
          for arg_name in ranges
      }
      t = run_fn(**kwargs)
      print("{}: {}".format(kwargs, t))
      results[name][value] = t
  return results


def main(_):
    t = run_config(FLAGS.num_tpus, FLAGS.batch_size, FLAGS.epoch_length)
    print(">>>>> time:", t)


if __name__ == "__main__":
  tf.app.run()
