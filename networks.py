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
import functools
import layers
import tensorflow as tf


class RenderingModel(object):

  def __init__(self, model_name, use_appearance=True):

    if model_name == 'pggan':
      self._model = ModelPGGAN(use_appearance)
    else:
      raise ValueError('Model %s not implemented!' % model_name)

  def __call__(self, x_in, z_app=None):
    return self._model(x_in, z_app)

  def get_appearance_encoder(self):
    return self._model._appearance_encoder

  def get_generator(self):
    return self._model._generator

  def get_content_encoder(self):
    return self._model._content_encoder


# "Progressive Growing of GANs (PGGAN)"-inspired architecture. Implementation is
# based on the implementation details in their paper, but code is not taken from
# the authors' released code.
# Main changes are:
#  - conditional GAN setup by introducting an encoder + skip connections.
#  - no progressive growing during training.
class ModelPGGAN(RenderingModel):

  def __init__(self, use_appearance=True):
    self._use_appearance = use_appearance
    self._content_encoder = None
    self._generator = GeneratorPGGAN(appearance_vec_size=opts.app_vector_size)
    if use_appearance:
      self._appearance_encoder = DRITAppearanceEncoderConcat(
          'appearance_net', opts.appearance_nc, opts.normalize_drit_Ez)
    else:
      self._appearance_encoder = None

  def __call__(self, x_in, z_app=None):
    y = self._generator(x_in, z_app)
    return y

  def get_appearance_encoder(self):
    return self._appearance_encoder

  def get_generator(self):
    return self._generator

  def get_content_encoder(self):
    return self._content_encoder


class PatchGANDiscriminator(object):

  def __init__(self, name_scope, input_nc, nf=64, n_layers=3, get_fmaps=False):
    """Constructor for a patchGAN discriminators.

    Args:
      name_scope: str - tf name scope.
      input_nc: int - number of input channels.
      nf: int - starting number of discriminator filters.
      n_layers: int - number of layers in the discriminator.
      get_fmaps: bool - return intermediate feature maps for FeatLoss.
    """
    self.get_fmaps = get_fmaps
    self.n_layers = n_layers
    kw = 4  # kernel width for convolution

    activation = functools.partial(tf.nn.leaky_relu, alpha=0.2)
    norm_layer = functools.partial(layers.LayerInstanceNorm)
    conv2d = functools.partial(layers.LayerConv, use_scaling=opts.use_scaling,
                               relu_slope=0.2)

    def minibatch_stats(x):
      return layers.scalar_concat(x, layers.minibatch_mean_variance(x))

    # Create layers.
    self.blocks = []
    with tf.variable_scope(name_scope, tf.AUTO_REUSE):
      with tf.variable_scope('block_0'):
        self.blocks.append([
            conv2d('conv0', w=kw, n=[input_nc, nf], stride=2),
            activation
        ])
      for ii_block in range(1, n_layers):
        nf_prev = nf
        nf = min(nf * 2, 512)
        with tf.variable_scope('block_%d' % ii_block):
          self.blocks.append([
              conv2d('conv%d' % ii_block, w=kw, n=[nf_prev, nf], stride=2),
              norm_layer(),
              activation
          ])
      # Add minibatch_stats (from PGGAN) and do a stride1 convolution.
      nf_prev = nf
      nf = min(nf * 2, 512)
      with tf.variable_scope('block_%d' % (n_layers + 1)):
        self.blocks.append([
            minibatch_stats,  # this is improvised by @meshry
            conv2d('conv%d' % (n_layers + 1), w=kw, n=[nf_prev + 1, nf],
                   stride=1),
            norm_layer(),
            activation
        ])
      # Get 1-channel patchGAN logits
      with tf.variable_scope('patchGAN_logits'):
        self.blocks.append([
            conv2d('conv%d' % (n_layers + 2), w=kw, n=[nf, 1], stride=1)
        ])

  def __call__(self, x, x_cond=None):
    # Concatenate extra conditioning input, if any.
    if x_cond is not None:
      x = tf.concat([x, x_cond], axis=3)

    if self.get_fmaps:
      # Dummy addition of x to D_fmaps, which will be removed before returing
      D_fmaps = [[x]]
      for i_block in range(len(self.blocks)):
        # Apply layer #0 in the current block
        block_fmaps = [self.blocks[i_block][0](D_fmaps[-1][-1])]
        # Apply the remaining layers of this block
        for i_layer in range(1, len(self.blocks[i_block])):
          block_fmaps.append(self.blocks[i_block][i_layer](block_fmaps[-1]))
        # Append the feature maps of this block to D_fmaps
        D_fmaps.append(block_fmaps)
      return D_fmaps[1:]  # exclude the input x which we added initially
    else:
      y = x
      for i_block in range(len(self.blocks)):
        for i_layer in range(len(self.blocks[i_block])):
          y = self.blocks[i_block][i_layer](y)
      return [[y]]


class MultiScaleDiscriminator(object):

  def __init__(self, name_scope, input_nc, num_scales=3, nf=64, n_layers=3,
               get_fmaps=False):
    self.get_fmaps = get_fmaps
    discs = []
    with tf.variable_scope(name_scope):
      for i in range(num_scales):
        discs.append(PatchGANDiscriminator(
            'D_scale%d' % i, input_nc, nf=nf, n_layers=n_layers,
            get_fmaps=get_fmaps))
    self.discriminators = discs

  def __call__(self, x, x_cond=None, params=None):
    del params
    if x_cond is not None:
      x = tf.concat([x, x_cond], axis=3)

    responses = []
    for ii, D in enumerate(self.discriminators):
      responses.append(D(x, x_cond=None))  # x_cond is already concatenated
      if ii != len(self.discriminators) - 1:
        x = layers.downscale(x, n=2)
    return responses


class GeneratorPGGAN(object):
  def __init__(self, appearance_vec_size=8, use_scaling=True,
               num_blocks=5, input_nc=7,
               fmap_base=8192, fmap_decay=1.0, fmap_max=512):
    """Generator model.
  
    Args:
      appearance_vec_size: int, size of the latent appearance vector.
      use_scaling: bool, whether to use weight scaling.
      resolution: int, width of the images (assumed to be square).
      input_nc: int, number of input channles.
      fmap_base: int, base number of channels.
      fmap_decay: float, decay rate of channels with respect to depth.
      fmap_max: int, max number of channels (supersedes fmap_base).
  
    Returns:
      function of the model.
    """
    def _num_filters(fmap_base, fmap_decay, fmap_max, stage):
      if opts.g_nf == 32:
        return min(int(2**(10 - stage)), fmap_max)  # nf32
      elif opts.g_nf == 64:
        return min(int(2**(11 - stage)), fmap_max)  # nf64
      else:
        raise ValueError('Currently unsupported num filters')

    nf = functools.partial(_num_filters, fmap_base, fmap_decay, fmap_max)
    self.num_blocks = num_blocks
    activation = functools.partial(tf.nn.leaky_relu, alpha=0.2)
    conv2d_stride1 = functools.partial(
        layers.LayerConv, stride=1, use_scaling=use_scaling, relu_slope=0.2)
    conv2d_rgb = functools.partial(layers.LayerConv, w=1, stride=1,
                                   use_scaling=use_scaling)
  
    # Create encoder layers.
    with tf.variable_scope('g_model_enc', tf.AUTO_REUSE):
      self.enc_stage = []
      self.from_rgb = []

      if opts.use_appearance and opts.inject_z == 'to_encoder':
        input_nc += appearance_vec_size
  
      for i in range(num_blocks, -1, -1):
        with tf.variable_scope('res_%d' % i):
          self.from_rgb.append(
              layers.LayerPipe([
                  conv2d_rgb('from_rgb', n=[input_nc, nf(i + 1)]),
                  activation,
              ])
          )
          self.enc_stage.append(
              layers.LayerPipe([
                  functools.partial(layers.downscale, n=2),
                  conv2d_stride1('conv0', w=3, n=[nf(i + 1), nf(i)]),
                  activation,
                  layers.pixel_norm,
                  conv2d_stride1('conv1', w=3, n=[nf(i), nf(i)]),
                  activation,
                  layers.pixel_norm
              ])
          )
  
    # Create decoder layers.
    with tf.variable_scope('g_model_dec', tf.AUTO_REUSE):
      self.dec_stage = []
      self.to_rgb = []
  
      nf_bottleneck = nf(0)  # num input filters at the bottleneck
      if opts.use_appearance and opts.inject_z == 'to_bottleneck':
        nf_bottleneck += appearance_vec_size

      with tf.variable_scope('res_0'):
        self.dec_stage.append(
          layers.LayerPipe([
            functools.partial(layers.upscale, n=2),
            conv2d_stride1('conv0', w=3, n=[nf_bottleneck, nf(1)]),
            activation,
            layers.pixel_norm,
            conv2d_stride1('conv1', w=3, n=[nf(1), nf(1)]),
            activation,
            layers.pixel_norm
          ])
        )
        self.to_rgb.append(conv2d_rgb('to_rgb', n=[nf(1), opts.output_nc]))
  
      multiply_factor = 2 if opts.concatenate_skip_layers else 1
      for i in range(1, num_blocks + 1):
        with tf.variable_scope('res_%d' % i):
          self.dec_stage.append(
              layers.LayerPipe([
                  functools.partial(layers.upscale, n=2),
                  conv2d_stride1('conv0', w=3,
                                 n=[multiply_factor * nf(i), nf(i + 1)]),
                  activation,
                  layers.pixel_norm,
                  conv2d_stride1('conv1', w=3, n=[nf(i + 1), nf(i + 1)]),
                  activation,
                  layers.pixel_norm
              ])
          )
          self.to_rgb.append(conv2d_rgb('to_rgb',
                                        n=[nf(i + 1), opts.output_nc]))

  def __call__(self, x, appearance_embedding=None, encoder_fmaps=None):
    """Generator function.

    Args:
      x: 2D tensor (batch, latents), the conditioning input batch of images.
      appearance_embedding: float tensor: latent appearance vector.
    Returns:
      4D tensor of images (NHWC), the generated images.
    """
    del encoder_fmaps
    enc_st_idx = 0
    if opts.use_appearance and opts.inject_z == 'to_encoder':
      x = layers.tile_and_concatenate(x, appearance_embedding,
                                      opts.app_vector_size)
    y = self.from_rgb[enc_st_idx](x)

    enc_responses = []
    for i in range(enc_st_idx, len(self.enc_stage)):
      y = self.enc_stage[i](y)
      enc_responses.insert(0, y)

    # Concatenate appearance vector to y
    if opts.use_appearance and opts.inject_z == 'to_bottleneck':
      appearance_tensor = tf.tile(appearance_embedding,
                                  [1, tf.shape(y)[1], tf.shape(y)[2], 1])
      y = tf.concat([y, appearance_tensor], axis=3)

    y_list = []
    for i in range(self.num_blocks + 1):
      if i > 0:
        y_skip = enc_responses[i]  # skip layer
        if opts.concatenate_skip_layers:
          y = tf.concat([y, y_skip], axis=3)
        else:
          y = y + y_skip
      y = self.dec_stage[i](y)
      y_list.append(y)

    return self.to_rgb[self.num_blocks](y_list[-1])


class DRITAppearanceEncoderConcat(object):

  def __init__(self, name_scope, input_nc, normalize_encoder):
    self.blocks = []
    activation = functools.partial(tf.nn.leaky_relu, alpha=0.2)
    conv2d = functools.partial(layers.LayerConv, use_scaling=opts.use_scaling,
                               relu_slope=0.2, padding='SAME')
    with tf.variable_scope(name_scope, tf.AUTO_REUSE):
      if normalize_encoder:
        self.blocks.append(layers.LayerPipe([
            conv2d('conv0', w=4, n=[input_nc, 64], stride=2),
            layers.BasicBlock('BB0', n=[64, 128], use_scaling=opts.use_scaling),
            layers.pixel_norm,
            layers.BasicBlock('BB1', n=[128, 192], use_scaling=opts.use_scaling),
            layers.pixel_norm,
            layers.BasicBlock('BB2', n=[192, 256], use_scaling=opts.use_scaling),
            layers.pixel_norm,
            activation,
            layers.global_avg_pooling
        ]))
      else:
        self.blocks.append(layers.LayerPipe([
            conv2d('conv0', w=4, n=[input_nc, 64], stride=2),
            layers.BasicBlock('BB0', n=[64, 128], use_scaling=opts.use_scaling),
            layers.BasicBlock('BB1', n=[128, 192], use_scaling=opts.use_scaling),
            layers.BasicBlock('BB2', n=[192, 256], use_scaling=opts.use_scaling),
            activation,
            layers.global_avg_pooling
        ]))
      # FC layers to get the mean and logvar
      self.fc_mean = layers.FullyConnected(opts.app_vector_size, 'FC_mean')
      self.fc_logvar = layers.FullyConnected(opts.app_vector_size, 'FC_logvar')

  def __call__(self, x):
    for f in self.blocks:
      x = f(x)

    mean = self.fc_mean(x)
    logvar = self.fc_logvar(x)
    # The following is an arbitrarily chosen *deterministic* latent vector
    # computation. Another option is to let z = mean, but gradients from logvar
    # will be None and will need to be removed.
    z = mean + tf.exp(0.5 * logvar)
    return z, mean, logvar
