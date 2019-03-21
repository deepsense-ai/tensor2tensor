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

r"""Training of RL agent with PPO algorithm.

Example invocation:

python -m tensor2tensor.rl.trainer_model_free \
    --output_dir=$HOME/t2t/rl_v1 \
    --hparams_set=pong_model_free \
    --hparams='batch_size=15'

Example invocation with EnvProblem interface:

python -m tensor2tensor.rl.trainer_model_free \
  --env_problem_name=tic_tac_toe_env_problem \
  --hparams_set=rlmf_tictactoe \
  --output_dir=${OUTPUTDIR} \
  --log_dir=${LOGDIR} \
  --alsologtostderr \
  --vmodule=*/tensor2tensor/*=2 \
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
_time = time.time()
#
# import os
# import pprint
#
# import tensorflow as tf
# import reprlib
# import socket
# import codecs
# import xmlrpc
# import os
# import google_auth_httplib2
# import pkgutil
# import difflib
# import stringprep
# import struct
# import random
# import html
# import signal
# import importlib
# import fractions
# import gc
# import gym
# import ast
# import keyword
# import json
# import time
# import string
# import cryptography
# import calendar
# import dis
# import oauth2client
# import inspect
# import pwd
# import threading
# import heapq
# import getpass
# import posix
# import uritemplate
# import encodings
# import numbers
# import multiprocessing
# import gettext
# import array
# import timeit
# import pyasn1
# import asn1crypto
# import PIL
# import ctypes
# import subprocess
# import sys
# import fnmatch
# import tempfile
# import asyncio
# import urllib
# import tokenize
# import queue
# import httplib2
# import pkg_resources
# import operator
# import http
# import errno
# import mesh_tensorflow
# import getopt
# import pydoc
# import posixpath
# import mmap
# import genericpath
# import decimal
# import tensorflow_probability
# import csv
# import mimetypes
# import tensorflow
# import sre_parse
# import sre_constants
# import pyexpat
# import uuid
# import pprint
# import abc
# import zlib
# import copyreg
# import token
# import traceback
# import yaml
# import fcntl
# import bisect
# import _decimal
# import ssl
# import logging
# import pathlib
# import cython_runtime
# import copy
# import zipimport
# import astor
# import lzma
# import textwrap
# import pickle
# import idna
# import quopri
# import socketserver
# import plistlib
# import functools
# import math
# import zipfile
# import cffi
# import marshal
# import hmac
# import grp
# import gast
# import rsa
# import codeop
# import tensorboard
# import googleapiclient
# import xml
# import cv2
# import itertools
# import stat
# import glob
# import code
# import binascii
# import unicodedata
# import numpy
# import email
# import contextlib
# import select
# import selectors
# import atexit
# import re
# import chardet
# import tarfile
# import certifi
# import argparse
# import concurrent
# import six
# import sysconfig
# import pyasn1_modules
# import bz2
# import locale
# import site
# import shutil
# import io
# import termios
# import h5py
# import requests
# import sre_compile
# import enum
# import scipy
# import distutils
# import ntpath
# import urllib3
# import hashlib
# import opcode
# import datetime
# import warnings
# import types
# import uu
# import collections
# import platform
# import linecache
# import termcolor
# import base64
# import trace
# import gzip
# import ipaddress
# import absl
# import unittest
import os
# print("t0:{}".format(time.time() - _time))
# from tensor2tensor.data_generators import all_problems
# all_problems.ALL_MODULES = ['tensor2tensor.data_generators.gym_env']
#
# import sys
# blacklist = []
# blacklist.extend([
#     'tensor2tensor.models.' + module
#     for module in [
#         'basic', 'bytenet', 'distillation', 'evolved_transformer', 'image_transformer', 'image_transformer_2d', 'lstm', 'mtf_image_transformer', 'mtf_resnet', 'mtf_transformer', 'mtf_transformer2', 'neural_gpu', 'resnet', 'revnet', 'shake_shake', 'slicenet', 'text_cnn', 'transformer', 'vanilla_gan', 'xception',
#     ]
# ])
# blacklist.extend([
#     'tensor2tensor.models.research.' + module
#     for module in [
#         'adafactor_experiments', 'aligned', 'attention_lm', 'attention_lm_moe', 'autoencoders', 'cycle_gan', 'gene_expression', 'glow', 'lm_experiments', 'moe_experiments', 'multiquery_paper', 'similarity_transformer', 'super_lm', 'transformer_moe', 'transformer_nat', 'transformer_parallel', 'transformer_revnet', 'transformer_sketch', 'transformer_symshard', 'transformer_vae', 'universal_transformer', 'vqa_attention', 'vqa_recurrent_self_attention', 'vqa_self_attention',
#     ]
# ])
# blacklist.extend([
#     'tensor2tensor.models.video.' + module
#     for module in [
#        'epva', 'next_frame_glow', 'savp',
#     ]
# ])
# blacklist.append('tensor2tensor.layers.common_audio')
# for module in blacklist:
#   sys.modules[module] = object()
#
# class fake_text_encoder(object):
#     def TextEncoder(*args, **kwargs):
#         pass
# sys.modules['tensor2tensor.data_generators.text_encoder'] = fake_text_encoder()
#
# class fake_bleu_hook(object):
#     bleu_score = None
# sys.modules['tensor2tensor.utils.bleu_hook'] = fake_bleu_hook()
from tensor2tensor.models.research import rl
from tensor2tensor.rl import rl_utils
from tensor2tensor.utils import flags as t2t_flags  # pylint: disable=unused-import
from tensor2tensor.utils import misc_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


flags.DEFINE_string("env_problem_name", "",
                    "Which registered env_problem do we want?")

# To maintain compatibility with some internal libs, we guard against these flag
# definitions possibly erring. Apologies for the ugliness.
try:
  flags.DEFINE_string("output_dir", "", "Base output directory for run.")
except:  # pylint: disable=bare-except
  pass


def initialize_env_specs(hparams, env_problem_name):
  """Initializes env_specs using the appropriate env."""
  if env_problem_name:
    env = registry.env_problem(env_problem_name, hparams.batch_size)
  else:
    env = rl_utils.setup_env(hparams, hparams.batch_size,
                             hparams.eval_max_num_noops,
                             hparams.rl_env_max_episode_steps,
                             env_name=hparams.rl_env_name)
    env.start_new_epoch(0)

  return rl.make_real_env_fn(env)


step = 0


def train(hparams, output_dir, env_problem_name, report_fn=None):
  """Train."""
  env_fn = initialize_env_specs(hparams, env_problem_name)

  tf.logging.vlog(1, "HParams in trainer_model_free.train : %s",
                  misc_utils.pprint_hparams(hparams))

  tf.logging.vlog(1, "Using hparams.base_algo: %s", hparams.base_algo)
  learner = rl_utils.LEARNERS[hparams.base_algo](
      hparams.frame_stack_size, output_dir, output_dir, total_num_epochs=1
  )

  policy_hparams = trainer_lib.create_hparams(hparams.base_algo_params)

  rl_utils.update_hparams_from_hparams(
      policy_hparams, hparams, hparams.base_algo + "_"
  )

  tf.logging.vlog(1, "Policy HParams : %s",
                  misc_utils.pprint_hparams(policy_hparams))

  total_steps = policy_hparams.epochs_num
  tf.logging.vlog(2, "total_steps: %d", total_steps)

  eval_every_epochs = policy_hparams.eval_every_epochs
  tf.logging.vlog(2, "eval_every_epochs: %d", eval_every_epochs)

  if eval_every_epochs == 0:
    eval_every_epochs = total_steps
  policy_hparams.eval_every_epochs = 0

  metric_name = rl_utils.get_metric_name(
      sampling_temp=hparams.eval_sampling_temps[0],
      max_num_noops=hparams.eval_max_num_noops,
      clipped=False
  )

  tf.logging.vlog(1, "metric_name: %s", metric_name)

  eval_metrics_dir = os.path.join(output_dir, "eval_metrics")
  eval_metrics_dir = os.path.expanduser(eval_metrics_dir)
  tf.gfile.MakeDirs(eval_metrics_dir)
  eval_metrics_writer = tf.summary.FileWriter(eval_metrics_dir)

  def evaluate_on_new_model(model_dir_path):
    global step
    eval_metrics = rl_utils.evaluate_all_configs(hparams, model_dir_path)
    tf.logging.info(
        "Agent eval metrics:\n{}".format(pprint.pformat(eval_metrics)))
    rl_utils.summarize_metrics(eval_metrics_writer, eval_metrics, step)
    if report_fn:
      report_fn(eval_metrics[metric_name], step)
    step += 1

  policy_hparams.epochs_num = total_steps
  policy_hparams.save_models_every_epochs = eval_every_epochs
  learner.train(env_fn,
                policy_hparams,
                simulated=False,
                save_continuously=True,
                epoch=0,
                model_save_fn=evaluate_on_new_model)


def main(_):
  hparams = trainer_lib.create_hparams(FLAGS.hparams_set, FLAGS.hparams)

  tf.logging.info("Starting model free training.")
  train(hparams, FLAGS.output_dir, FLAGS.env_problem_name)
  tf.logging.info("Ended model free training.")


if __name__ == "__main__":
  tf.app.run()
