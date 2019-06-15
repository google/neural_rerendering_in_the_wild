# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from options import FLAGS as opts
import layers
import os.path as osp
import tensorflow as tf
import vgg16


def gradient_penalty_loss(y_xy, xy, iwass_target=1, iwass_lambda=10):
  grad = tf.gradients(tf.reduce_sum(y_xy), [xy])[0]
  grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]) + 1e-8)
  loss_gp = tf.reduce_mean(
      tf.square(grad_norm - iwass_target)) * iwass_lambda / iwass_target**2
  return loss_gp


def KL_loss(mean, logvar):
  loss = 0.5 * tf.reduce_sum(tf.square(mean) + tf.exp(logvar) - 1. - logvar,
                             axis=-1)
  return tf.reduce_sum(loss)  # just to match DRIT implementation


def l2_regularize(x):
  return tf.reduce_mean(tf.square(x))


def L1_loss(x, y):
  return tf.reduce_mean(tf.abs(x - y))


class PerceptualLoss:
  def __init__(self, x, y, image_shape, layers, w_layers, w_act=0.1):
    """
    Builds vgg16 network and computes the perceptual loss.
    """
    assert len(image_shape) == 3 and image_shape[-1] == 3
    assert osp.exists(opts.vgg16_path), 'Cannot find %s' % opts.vgg16_path

    self.w_act = w_act
    self.vgg_layers = layers
    self.w_layers = w_layers
    batch_shape = [None] + image_shape  # [None, H, W, 3]

    vgg_net = vgg16.Vgg16(opts.vgg16_path)
    self.x_acts = vgg_net.get_vgg_activations(x, layers)
    self.y_acts = vgg_net.get_vgg_activations(y, layers)
    loss = 0
    for w, act1, act2 in zip(self.w_layers, self.x_acts, self.y_acts):
      loss += w * tf.reduce_mean(tf.square(self.w_act * (act1 - act2)))
    self.loss = loss

  def __call__(self):
    return self.loss


def lsgan_appearance_E_loss(disc_response):
  disc_response = tf.squeeze(disc_response)
  gt_label = 0.5
  loss = tf.reduce_mean(tf.square(disc_response - gt_label))
  return loss


def lsgan_loss(disc_response, is_real):
  gt_label = 1 if is_real else 0
  disc_response = tf.squeeze(disc_response)
  # The following works for both regular and patchGAN discriminators
  loss = tf.reduce_mean(tf.square(disc_response - gt_label))
  return loss


def multiscale_discriminator_loss(Ds_responses, is_real):
  num_D = len(Ds_responses)
  loss = 0
  for i in range(num_D):
    curr_response = Ds_responses[i][-1][-1]
    loss += lsgan_loss(curr_response, is_real)
  return loss
