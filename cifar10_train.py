# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""
Trains on CIFAR-10.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import os
import math

import tensorflow as tf

import cifar10

import variational_dropout as vd

parser = cifar10.parser

parser.add_argument('--train_dir', type=str, default=os.environ.get('SCRATCH', '/tmp/cifar10')+'/tf-models',
                    help='Directory where to write event logs and checkpoint.')

parser.add_argument('--max_steps', type=int, default=1000000,
                    help='Number of batches to run.')

parser.add_argument('--lr', type=float, default=0.1,
                    help='Initial learning rate.')

parser.add_argument('--log_device_placement', type=bool, default=False,
                    help='Whether to log device placement.')

parser.add_argument('--clean', action='store_true',
                    help='Whether to start from clean (WILL DELETE OLD FILES).')

parser.add_argument('--vanilla', action='store_true',
                    help='Run without variational dropout.')

parser.add_argument('--log_frequency', type=int, default=10,
                    help='How often to log results to the console.')


def train(train_dir):
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = cifar10.distorted_inputs()
      #phase = tf.placeholder(bool, name='is_train')
      phase = tf.Variable(True, name='is_train', dtype=bool, trainable=False)
      #learning_rate = tf.placeholder(tf.float32, name='learning_rate')
      learning_rate = tf.Variable(0.1, name='learning_rate', trainable=False)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    if not FLAGS.vanilla:
      logits = cifar10.inference(images, phase, vd.conv2d)
    else:
      logits = cifar10.inference(images, phase, None)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step, learning_rate)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._start_time = time.time()

      def before_run(self, run_context):
        return tf.train.SessionRunArgs((loss,global_step))  # Asks for loss value.

      def after_run(self, run_context, run_values):
        loss_value, step = run_values.results
        self._step = step
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    class _ScheduleHook(tf.train.SessionRunHook):
      """Controls learning rate schedule."""

      def begin(self):
        self.lr = FLAGS.lr
        self.decay_rate = 0.1
        self.N = 50000 # dataset size, known beforehand
        self.minibatch_size = FLAGS.batch_size

      def before_run(self, run_context):
        return tf.train.SessionRunArgs(global_step)

      def after_run(self, run_context, run_values):
        step = self.minibatch_size*(run_values.results + 1)
        period = 60*self.N
        self.lr = FLAGS.lr*(self.decay_rate**math.floor(step/period))
        #format_str = '%s: step %d, lr = %.6f'
        #print(format_str%(datetime.now(), step, self.lr))

    schedule_hook = _ScheduleHook()

    saver = tf.train.Saver()
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook(),
               schedule_hook],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      ckpt = tf.train.get_checkpoint_state(train_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(mon_sess, ckpt.model_checkpoint_path)
      else:
        print("Starting clean in %s"%train_dir)
      global_step = tf.contrib.framework.get_or_create_global_step()
        
      while not mon_sess.should_stop():
        mon_sess.run(train_op, feed_dict={learning_rate:schedule_hook.lr})


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if FLAGS.vanilla:
    train_dir = FLAGS.train_dir + '/vanilla' 
  else:
    train_dir = FLAGS.train_dir + '/vardrop'
  if tf.gfile.Exists(train_dir) and FLAGS.clean:
    tf.gfile.DeleteRecursively(train_dir)
  if not tf.gfile.Exists(train_dir):
    tf.gfile.MakeDirs(train_dir)
  train(train_dir)


if __name__ == '__main__':
  FLAGS = parser.parse_args()
  tf.app.run()
