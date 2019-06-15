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

import functools
import numpy as np
import tensorflow as tf


class LayerInstanceNorm(object):

  def __init__(self, scope_suffix='instance_norm'):
    curr_scope = tf.get_variable_scope().name
    self._scope = curr_scope + '/' + scope_suffix

  def __call__(self, x):
    with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
      return tf.contrib.layers.instance_norm(
        x, epsilon=1e-05, center=True, scale=True)


def layer_norm(x, scope='layer_norm'):
  return tf.contrib.layers.layer_norm(x, center=True, scale=True)


def pixel_norm(x):
  """Pixel normalization.

  Args:
    x: 4D image tensor in B01C format.

  Returns:
    4D tensor with pixel normalized channels.
  """
  return x * tf.rsqrt(tf.reduce_mean(tf.square(x), [-1], keepdims=True) + 1e-8)


def global_avg_pooling(x):
  return tf.reduce_mean(x, axis=[1, 2], keepdims=True)


class FullyConnected(object):

  def __init__(self, n_out_units, scope_suffix='FC'):
    weight_init = tf.random_normal_initializer(mean=0., stddev=0.02)
    weight_regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)

    curr_scope = tf.get_variable_scope().name
    self._scope = curr_scope + '/' + scope_suffix
    self.fc_layer = functools.partial(
      tf.layers.dense, units=n_out_units, kernel_initializer=weight_init,
      kernel_regularizer=weight_regularizer, use_bias=True)

  def __call__(self, x):
    with tf.variable_scope(self._scope, reuse=tf.AUTO_REUSE):
      return self.fc_layer(x)


def init_he_scale(shape, slope=1.0):
  """He neural network random normal scaling for initialization.

  Args:
    shape: list of the dimensions of the tensor.
    slope: float, slope of the ReLu following the layer.

  Returns:
    a float, He's standard deviation.
  """
  fan_in = np.prod(shape[:-1])
  return np.sqrt(2. / ((1. + slope**2) * fan_in))


class LayerConv(object):
  """Convolution layer with support for equalized learning."""

  def __init__(self,
               name,
               w,
               n,
               stride,
               padding='SAME',
               use_scaling=False,
               relu_slope=1.):
    """Layer constructor.

    Args:
      name: string, layer name.
      w: int or 2-tuple, width of the convolution kernel.
      n: 2-tuple of ints, input and output channel depths.
      stride: int or 2-tuple, stride for the convolution kernel.
      padding: string, the padding method. {SAME, VALID, REFLECT}.
      use_scaling: bool, whether to use weight norm and scaling.
      relu_slope: float, the slope of the ReLu following the layer.
    """
    assert padding in ['SAME', 'VALID', 'REFLECT'], 'Error: unsupported padding'
    self._padding = padding
    with tf.variable_scope(name):
      if isinstance(stride, int):
        stride = [1, stride, stride, 1]
      else:
        assert len(stride) == 0, "stride is either an int or a 2-tuple"
        stride = [1, stride[0], stride[1], 1]
      if isinstance(w, int):
        w = [w, w]
      self.w = w
      shape = [w[0], w[1], n[0], n[1]]
      init_scale, pre_scale = init_he_scale(shape, relu_slope), 1.
      if use_scaling:
        init_scale, pre_scale = pre_scale, init_scale
      self._stride = stride
      self._pre_scale = pre_scale
      self._weight = tf.get_variable(
          'weight',
          shape=shape,
          initializer=tf.random_normal_initializer(stddev=init_scale))
      self._bias = tf.get_variable(
          'bias', shape=[n[1]], initializer=tf.zeros_initializer)

  def __call__(self, x):
    """Apply layer to tensor x."""
    if self._padding != 'REFLECT':
      padding = self._padding
    else:
      padding = 'VALID'
      pad_top = self.w[0] // 2
      pad_left = self.w[1] // 2
      if (self.w[0] - self._stride[1]) % 2 == 0:
        pad_bottom = pad_top
      else:
        pad_bottom = self.w[0] - self._stride[1] - pad_top
      if (self.w[1] - self._stride[2]) % 2 == 0:
        pad_right = pad_left
      else:
        pad_right = self.w[1] - self._stride[2] - pad_left
      x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right],
                     [0, 0]], mode='REFLECT')
    y = tf.nn.conv2d(x, self._weight, strides=self._stride, padding=padding)
    return self._pre_scale * y + self._bias


class LayerTransposedConv(object):
  """Convolution layer with support for equalized learning."""

  def __init__(self,
               name,
               w,
               n,
               stride,
               padding='SAME',
               use_scaling=False,
               relu_slope=1.):
    """Layer constructor.

    Args:
      name: string, layer name.
      w: int or 2-tuple, width of the convolution kernel.
      n: 2-tuple int, [n_in_channels, n_out_channels]
      stride: int or 2-tuple, stride for the convolution kernel.
      padding: string, the padding method {SAME, VALID, REFLECT}.
      use_scaling: bool, whether to use weight norm and scaling.
      relu_slope: float, the slope of the ReLu following the layer.
    """
    assert padding in ['SAME'], 'Error: unsupported padding for transposed conv'
    if isinstance(stride, int):
      stride = [1, stride, stride, 1]
    else:
      assert len(stride) == 2, "stride is either an int or a 2-tuple"
      stride = [1, stride[0], stride[1], 1]
    if isinstance(w, int):
      w = [w, w]
    self.padding = padding
    self.nc_in, self.nc_out = n
    self.stride = stride
    with tf.variable_scope(name):
      kernel_shape = [w[0], w[1], self.nc_out, self.nc_in]
      init_scale, pre_scale = init_he_scale(kernel_shape, relu_slope), 1.
      if use_scaling:
        init_scale, pre_scale = pre_scale, init_scale
      self._pre_scale = pre_scale
      self._weight = tf.get_variable(
          'weight',
          shape=kernel_shape,
          initializer=tf.random_normal_initializer(stddev=init_scale))
      self._bias = tf.get_variable(
          'bias', shape=[self.nc_out], initializer=tf.zeros_initializer)

  def __call__(self, x):
    """Apply layer to tensor x."""
    x_shape = x.get_shape().as_list()
    batch_size = tf.shape(x)[0]
    stride_x, stride_y = self.stride[1], self.stride[2]
    output_shape = tf.stack([
      batch_size, x_shape[1] * stride_x, x_shape[2] * stride_y, self.nc_out])
    y = tf.nn.conv2d_transpose(
      x, filter=self._weight, output_shape=output_shape, strides=self.stride,
      padding=self.padding)
    return self._pre_scale * y + self._bias


class ResBlock(object):
  def __init__(self,
               name,
               nc,
               norm_layer_constructor,
               activation,
               padding='SAME',
               use_scaling=False,
               relu_slope=1.):
    """Layer constructor."""
    self.name = name
    conv2d = functools.partial(
        LayerConv, w=3, n=[nc, nc], stride=1, padding=padding,
        use_scaling=use_scaling, relu_slope=relu_slope)
    self.blocks = []
    with tf.variable_scope(self.name):
      with tf.variable_scope('res0'):
        self.blocks.append(
          LayerPipe([
            conv2d('res0_conv'),
            norm_layer_constructor('res0_norm'),
            activation
          ])
        )
      with tf.variable_scope('res1'):
        self.blocks.append(
          LayerPipe([
            conv2d('res1_conv'),
            norm_layer_constructor('res1_norm')
          ])
        )

  def __call__(self, x_init):
    """Apply layer to tensor x."""
    x = x_init
    for f in self.blocks:
      x = f(x)
    return x + x_init


class BasicBlock(object):
  def __init__(self,
               name,
               n,
               activation=functools.partial(tf.nn.leaky_relu, alpha=0.2),
               padding='SAME',
               use_scaling=True,
               relu_slope=1.):
    """Layer constructor."""
    self.name = name
    conv2d = functools.partial(
        LayerConv, stride=1, padding=padding,
        use_scaling=use_scaling, relu_slope=relu_slope)
    avg_pool = functools.partial(downscale, n=2)
    nc_in, nc_out = n  # n is a 2-tuple
    with tf.variable_scope(self.name):
      self.path1_blocks = []
      with tf.variable_scope('bb_path1'):
        self.path1_blocks.append(
          LayerPipe([
            activation,
            conv2d('bb_conv0', w=3, n=[nc_in, nc_out]),
            activation,
            conv2d('bb_conv1', w=3, n=[nc_out, nc_out]),
            downscale
          ])
        )

      self.path2_blocks = []
      with tf.variable_scope('bb_path2'):
        self.path2_blocks.append(
          LayerPipe([
            downscale,
            conv2d('path2_conv', w=1, n=[nc_in, nc_out])
          ])
        )

  def __call__(self, x_init):
    """Apply layer to tensor x."""
    x1 = x_init
    x2 = x_init
    for f in self.path1_blocks:
      x1 = f(x1)
    for f in self.path2_blocks:
      x2 = f(x2)
    return x1 + x2


class LayerDense(object):
  """Dense layer with a non-linearity."""

  def __init__(self, name, n, use_scaling=False, relu_slope=1.):
    """Layer constructor.

    Args:
      name: string, layer name.
      n: 2-tuple of ints, input and output widths.
      use_scaling: bool, whether to use weight norm and scaling.
      relu_slope: float, the slope of the ReLu following the layer.
    """
    with tf.variable_scope(name):
      init_scale, pre_scale = init_he_scale(n, relu_slope), 1.
      if use_scaling:
        init_scale, pre_scale = pre_scale, init_scale
      self._pre_scale = pre_scale
      self._weight = tf.get_variable(
          'weight',
          shape=n,
          initializer=tf.random_normal_initializer(stddev=init_scale))
      self._bias = tf.get_variable(
          'bias', shape=[n[1]], initializer=tf.zeros_initializer)

  def __call__(self, x):
    """Apply layer to tensor x."""
    return self._pre_scale * tf.matmul(x, self._weight) + self._bias


class LayerPipe(object):
  """Pipe a sequence of functions."""

  def __init__(self, functions):
    """Layer constructor.

    Args:
      functions: list, functions to pipe.
    """
    self._functions = tuple(functions)

  def __call__(self, x, **kwargs):
    """Apply pipe to tensor x and return result."""
    del kwargs
    for f in self._functions:
      x = f(x)
    return x


def downscale(x, n=2):
  """Box downscaling.

  Args:
    x: 4D image tensor.
    n: integer scale (must be a power of 2).

  Returns:
    4D tensor of images down scaled by a factor n.
  """
  if n == 1:
    return x
  return tf.nn.avg_pool(x, [1, n, n, 1], [1, n, n, 1], 'VALID')


def upscale(x, n):
  """Box upscaling (also called nearest neighbors).

  Args:
    x: 4D image tensor in B01C format.
    n: integer scale (must be a power of 2).

  Returns:
    4D tensor of images up scaled by a factor n.
  """
  if n == 1:
    return x
  x_shape = tf.shape(x)
  height, width = x_shape[1], x_shape[2]
  return tf.image.resize_nearest_neighbor(x, [n * height, n * width])


def tile_and_concatenate(x, z, n_z):
  z = tf.reshape(z, shape=[-1, 1, 1, n_z])
  z = tf.tile(z, [1, tf.shape(x)[1], tf.shape(x)[2], 1])
  x = tf.concat([x, z], axis=-1)
  return x


def minibatch_mean_variance(x):
  """Computes the variance average.

  This is used by the discriminator as a form of batch discrimination.

  Args:
    x: nD tensor for which to compute variance average.

  Returns:
    a scalar, the mean variance of variable x.
  """
  mean = tf.reduce_mean(x, 0, keepdims=True)
  vals = tf.sqrt(tf.reduce_mean(tf.squared_difference(x, mean), 0) + 1e-8)
  vals = tf.reduce_mean(vals)
  return vals


def scalar_concat(x, scalar):
  """Concatenate a scalar to a 4D tensor as an extra channel.

  Args:
    x: 4D image tensor in B01C format.
    scalar: a scalar to concatenate to the tensor.

  Returns:
    a 4D tensor with one extra channel containing the value scalar at
     every position.
  """
  s = tf.shape(x)
  return tf.concat([x, tf.ones([s[0], s[1], s[2], 1]) * scalar], axis=3)
