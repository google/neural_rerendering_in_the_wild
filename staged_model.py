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

"""Neural re-rerendering in the wild.

Implementation of the staged training pipeline.
"""

from options import FLAGS as opts
import losses
import networks
import tensorflow as tf
import utils


def create_computation_graph(x_in, x_gt, x_app=None, arch_type='pggan',
                             use_appearance=True):
  """Create the models and the losses.

  Args:
    x_in: 4D tensor, batch of conditional input images in NHWC format.
    x_gt: 2D tensor, batch ground-truth images in NHWC format.
    x_app: 4D tensor, batch of input appearance images.

  Returns:
    Dictionary of placeholders and TF graph functions.
  """
  # ---------------------------------------------------------------------------
  # Build models/networks
  # ---------------------------------------------------------------------------

  rerenderer = networks.RenderingModel(arch_type, use_appearance)
  app_enc = rerenderer.get_appearance_encoder()
  discriminator = networks.MultiScaleDiscriminator(
      'd_model', opts.appearance_nc, num_scales=3, nf=64, n_layers=3,
      get_fmaps=False)

  # ---------------------------------------------------------------------------
  # Forward pass
  # ---------------------------------------------------------------------------

  if opts.use_appearance:
    z_app, _, _ = app_enc(x_app)
  else:
    z_app = None

  y = rerenderer(x_in, z_app)

  # ---------------------------------------------------------------------------
  # Losses
  # ---------------------------------------------------------------------------

  w_loss_gan = opts.w_loss_gan
  w_loss_recon = opts.w_loss_vgg if opts.use_vgg_loss else opts.w_loss_l1

  # compute discriminator logits
  disc_real_featmaps = discriminator(x_gt, x_in)
  disc_fake_featmaps = discriminator(y, x_in)

  # discriminator loss
  loss_d_real = losses.multiscale_discriminator_loss(disc_real_featmaps, True)
  loss_d_fake = losses.multiscale_discriminator_loss(disc_fake_featmaps, False)
  loss_d = loss_d_real + loss_d_fake

  # generator loss
  loss_g_gan = losses.multiscale_discriminator_loss(disc_fake_featmaps, True)
  if opts.use_vgg_loss:
    vgg_layers = ['conv%d_2' % i for i in range(1, 6)]  # conv1 through conv5
    vgg_layer_weights = [1./32, 1./16, 1./8, 1./4, 1.]
    vgg_loss = losses.PerceptualLoss(y, x_gt, [256, 256, 3], vgg_layers,
                                     vgg_layer_weights)  # NOTE: shouldn't hardcode image size!
    loss_g_recon = vgg_loss()
  else:
    loss_g_recon = losses.L1_loss(y, x_gt)
  loss_g = w_loss_gan * loss_g_gan + w_loss_recon * loss_g_recon

  # ---------------------------------------------------------------------------
  # Tensorboard visualizations
  # ---------------------------------------------------------------------------

  x_in_render = tf.slice(x_in, [0, 0, 0, 0], [-1, -1, -1, 3])
  if opts.use_semantic:
    x_in_semantic = tf.slice(x_in, [0, 0, 0, 4], [-1, -1, -1, 3])
    tb_visualization = tf.concat([x_in_render, x_in_semantic, y, x_gt], axis=2)
  else:
    tb_visualization = tf.concat([x_in_render, y, x_gt], axis=2)
  tf.summary.image('rendered-semantic-generated-gt tuple', tb_visualization)

  # Show input appearance images
  if opts.use_appearance:
    x_app_rgb = tf.slice(x_app, [0, 0, 0, 0], [-1, -1, -1, 3])
    x_app_sem = tf.slice(x_app, [0, 0, 0, 7], [-1, -1, -1, -1])
    tb_app_visualization = tf.concat([x_app_rgb, x_app_sem], axis=2)
    tf.summary.image('input appearance image', tb_app_visualization)

  # Loss summaries
  with tf.name_scope('Discriminator_Loss'):
    tf.summary.scalar('D_real_loss', loss_d_real)
    tf.summary.scalar('D_fake_loss', loss_d_fake)
    tf.summary.scalar('D_total_loss', loss_d)
  with tf.name_scope('Generator_Loss'):
    tf.summary.scalar('G_GAN_loss', w_loss_gan * loss_g_gan)
    tf.summary.scalar('G_reconstruction_loss', w_loss_recon * loss_g_recon)
    tf.summary.scalar('G_total_loss', loss_g)

  # ---------------------------------------------------------------------------
  # Optimizers
  # ---------------------------------------------------------------------------

  def get_optimizer(lr, loss, var_list):
    optimizer = tf.train.AdamOptimizer(lr, opts.adam_beta1, opts.adam_beta2)
    # optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
    return optimizer.minimize(loss, var_list=var_list)

  # Training ops.
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    with tf.variable_scope('optimizers'):
      d_vars = utils.model_vars('d_model')[0]
      g_vars_all = utils.model_vars('g_model')[0]
      train_d = [get_optimizer(opts.d_lr, loss_d, d_vars)]
      train_g = [get_optimizer(opts.g_lr, loss_g, g_vars_all)]

      train_app_encoder = []
      if opts.train_app_encoder:
        lr_app = opts.ez_lr
        app_enc_vars = utils.model_vars('appearance_net')[0]
        train_app_encoder.append(get_optimizer(lr_app, loss_g, app_enc_vars))

  ema = tf.train.ExponentialMovingAverage(decay=0.999)
  with tf.control_dependencies(train_g + train_app_encoder):
    inference_vars_all = g_vars_all
    if opts.use_appearance:
      app_enc_vars = utils.model_vars('appearance_net')[0]
      inference_vars_all += app_enc_vars
    ema_op = ema.apply(inference_vars_all)

  print('***************************************************')
  print('len(g_vars_all) = %d' % len(g_vars_all))
  for ii, v in enumerate(g_vars_all):
    print('%03d) %s' % (ii, str(v)))
  print('-------------------------------------------------------')
  print('len(d_vars) = %d' % len(d_vars))
  for ii, v in enumerate(d_vars):
    print('%03d) %s' % (ii, str(v)))
  if opts.train_app_encoder:
    print('-------------------------------------------------------')
    print('len(app_enc_vars) = %d' % len(app_enc_vars))
    for ii, v in enumerate(app_enc_vars):
      print('%03d) %s' % (ii, str(v)))
  print('***************************************************\n\n')

  return {
      'train_disc_op': tf.group(train_d),
      'train_renderer_op': ema_op,
      'total_loss_d': loss_d,
      'loss_d_real': loss_d_real,
      'loss_d_fake': loss_d_fake,
      'loss_g_gan': w_loss_gan * loss_g_gan,
      'loss_g_recon': w_loss_recon * loss_g_recon,
      'total_loss_g': loss_g}
