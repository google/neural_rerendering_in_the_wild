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
import glob
import numpy as np
import os.path as osp
import random
import tensorflow as tf


def provide_data(dataset_name='', parent_dir='', batch_size=8, subset=None,
                 max_examples=None, crop_flag=False, crop_size=256, seeds=None,
                 use_appearance=True, shuffle=128):
  # Parsing function for each tfrecord example.
  record_parse_fn = functools.partial(
      _parser_rendered_dataset, crop_flag=crop_flag, crop_size=crop_size,
      use_alpha=opts.use_alpha, use_depth=opts.use_depth,
      use_semantics=opts.use_semantic, seeds=seeds,
      use_appearance=use_appearance)

  input_dict_var = multi_input_fn_record(
      record_parse_fn, parent_dir, dataset_name, batch_size,
      subset=subset, max_examples=max_examples, shuffle=shuffle)
  return input_dict_var


def _parser_rendered_dataset(
    serialized_example, crop_flag, crop_size, seeds, use_alpha, use_depth,
    use_semantics, use_appearance):
  """
  Parses a single tf.Example into a features dictionary with input tensors.
  """
  # Structure of features_dict need to match the dictionary structure that was
  # serialized to a tf.Example
  features_dict = {'height': tf.FixedLenFeature([], tf.int64),
                   'width': tf.FixedLenFeature([], tf.int64),
                   'rendered': tf.FixedLenFeature([], tf.string),
                   'depth': tf.FixedLenFeature([], tf.string),
                   'real': tf.FixedLenFeature([], tf.string),
                   'seg': tf.FixedLenFeature([], tf.string)}
  features = tf.parse_single_example(serialized_example, features=features_dict)
  height = tf.cast(features['height'], tf.int32)
  width = tf.cast(features['width'], tf.int32)

  # Parse the rendered image.
  rendered = tf.decode_raw(features['rendered'], tf.uint8)
  rendered = tf.cast(rendered, tf.float32) * (2.0 / 255) - 1.0
  rendered = tf.reshape(rendered, [height, width, 4])
  if not use_alpha:
    rendered = tf.slice(rendered, [0, 0, 0], [height, width, 3])
  conditional_input = rendered

  # Parse the depth image.
  if use_depth:
    depth = tf.decode_raw(features['depth'], tf.uint16)
    depth = tf.reshape(depth, [height, width, 1])
    depth = tf.cast(depth, tf.float32) * (2.0 / 255) - 1.0
    conditional_input = tf.concat([conditional_input, depth], axis=-1)

  # Parse the semantic map.
  if use_semantics:
    seg_img = tf.decode_raw(features['seg'], tf.uint8)
    seg_img = tf.reshape(seg_img, [height, width, 3])
    seg_img = tf.cast(seg_img, tf.float32) * (2.0 / 255) - 1
    conditional_input = tf.concat([conditional_input, seg_img], axis=-1)

  # Verify that the parsed input has the correct number of channels.
  assert conditional_input.shape[-1] == opts.deep_buffer_nc, ('num channels '
      'in the parsed input doesn\'t match num input channels specified in '
      'opts.deep_buffer_nc!')

  # Parse the ground truth image.
  real = tf.decode_raw(features['real'], tf.uint8)
  real = tf.cast(real, tf.float32) * (2.0 / 255) - 1.0
  real = tf.reshape(real, [height, width, 3])

  # Parse the appearance image (if any).
  appearance_input = []
  if use_appearance:
    # Concatenate the deep buffer to the real image.
    appearance_input = tf.concat([real, conditional_input], axis=-1)
    # Verify that the parsed input has the correct number of channels.
    assert appearance_input.shape[-1] == opts.appearance_nc, ('num channels '
        'in the parsed appearance input doesn\'t match num input channels '
        'specified in opts.appearance_nc!')

  # Crop conditional_input and real images, but keep the appearance input
  # uncropped (learn a one-to-many mapping from appearance to output)
  if crop_flag:
    assert crop_size is not None, 'crop_size is not provided!'
    if isinstance(crop_size, int):
      crop_size = [crop_size, crop_size]
    assert len(crop_size) == 2, 'crop_size is either an int or a 2-tuple!'

    # Central crop
    if seeds is not None and len(seeds) <= 1:
      conditional_input = tf.image.resize_image_with_crop_or_pad(
          conditional_input, crop_size[0], crop_size[1])
      real = tf.image.resize_image_with_crop_or_pad(real, crop_size[0],
                                                    crop_size[1])
    else:
      if not seeds:  # random crops
        seed = random.randint(0, (1 << 31) - 1)
      else:  # fixed crops
        seed_idx = random.randint(0, len(seeds) - 1)
        seed = seeds[seed_idx]
      conditional_input = tf.random_crop(
          conditional_input, crop_size + [opts.deep_buffer_nc], seed=seed)
      real = tf.random_crop(real, crop_size + [3], seed=seed)

  features = {'conditional_input': conditional_input,
              'expected_output': real,
              'peek_input': appearance_input}
  return features


def multi_input_fn_record(
    record_parse_fn, parent_dir, tfrecord_basename, batch_size, subset=None,
    max_examples=None, shuffle=128):
  """Creates a Dataset pipeline for tfrecord files.

  Returns:
    Dataset iterator.
  """
  subset_suffix = '*_%s.tfrecord' % subset if subset else '*.tfrecord'
  input_pattern = osp.join(parent_dir, tfrecord_basename + subset_suffix)
  filenames = sorted(glob.glob(input_pattern))
  assert len(filenames) > 0, ('Error! input pattern "%s" didn\'t match any '
                              'files' % input_pattern)
  dataset = tf.data.TFRecordDataset(filenames)
  if shuffle == 0:  # keep input deterministic
    # use one thread to get deterministic results
    dataset = dataset.map(record_parse_fn, num_parallel_calls=None)
  else:
    dataset = dataset.repeat()  # Repeat indefinitely.
    dataset = dataset.map(record_parse_fn,
                          num_parallel_calls=max(4, batch_size // 4))
    if opts.training_pipeline == 'drit':
      dataset1 = dataset.shuffle(shuffle)
      dataset2 = dataset.shuffle(shuffle)
      paired_dataset = tf.data.Dataset.zip((dataset1, dataset2))

      def _join_paired_dataset(features_a, features_b):
        features_a['conditional_input_2'] = features_b['conditional_input']
        features_a['expected_output_2'] = features_b['expected_output']
        return features_a

      joined_dataset = paired_dataset.map(_join_paired_dataset)
      dataset = joined_dataset
    else:
      dataset = dataset.shuffle(shuffle)
  if max_examples is not None:
    dataset = dataset.take(max_examples)
  dataset = dataset.batch(batch_size)
  if shuffle > 0:  # input is not deterministic
    dataset = dataset.prefetch(4)  # Prefetch a few batches.
  return dataset.make_one_shot_iterator().get_next()
