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

from PIL import Image
from absl import app
from absl import flags
from options import FLAGS as opts
import glob
import networks
import numpy as np
import os
import os.path as osp
import pickle
import style_loss
import tensorflow as tf
import utils


def _load_and_concatenate_image_channels(
    rgb_path=None, rendered_path=None, depth_path=None, seg_path=None,
    crop_size=512):
  if (rgb_path is None and rendered_path is None and depth_path is None and
      seg_path is None):
    raise ValueError('At least one of the inputs has to be not None')

  channels = ()
  if rgb_path is not None:
    rgb_img = np.array(Image.open(rgb_path)).astype(np.float32)
    rgb_img = utils.get_central_crop(rgb_img, crop_size, crop_size)
    channels = channels + (rgb_img,)
  if rendered_path is not None:
    rendered_img = np.array(Image.open(rendered_path)).astype(np.float32)
    rendered_img = utils.get_central_crop(rendered_img, crop_size, crop_size)
    if not opts.use_alpha:
      rendered_img = rendered_img[:,:, :3]  # drop the alpha channel
    channels = channels + (rendered_img,)
  if depth_path is not None:
    depth_img = np.array(Image.open(depth_path))
    depth_img = depth_img.astype(np.float32)
    depth_img = utils.get_central_crop(depth_img, crop_size, crop_size)
    channels = channels + (depth_img,)
  if seg_path is not None:
    seg_img = np.array(Image.open(seg_path)).astype(np.float32)
    channels = channels + (seg_img,)
  # Concatenate and normalize channels
  img = np.dstack(channels)
  img = img * (2.0 / 255) - 1.0
  return img


def read_single_appearance_input(rgb_img_path):
  base_path = rgb_img_path[:-14]  # remove the '_reference.png' suffix
  rendered_img_path = base_path + '_color.png'
  depth_img_path = base_path + '_depth.png'
  semantic_img_path = base_path + '_seg_rgb.png'
  network_input_img = _load_and_concatenate_image_channels(
      rgb_img_path, rendered_img_path, depth_img_path, semantic_img_path,
      crop_size=opts.train_resolution)
  return network_input_img


def get_triplet_input_fn(dataset_path, dist_file_path=None, k_max_nearest=5,
                         k_max_farthest=13):
  input_images_pattern = osp.join(dataset_path, '*_reference.png')
  filenames = sorted(glob.glob(input_images_pattern))
  print('DBG: obtained %d input filenames for triplet inputs' % len(filenames))
  print('DBG: Computing pairwise style distances:')
  if dist_file_path is not None and osp.exists(dist_file_path):
    print('*** Loading distance matrix from %s' % dist_file_path)
    with open(dist_file_path, 'rb') as f:
      dist_matrix = pickle.load(f)['dist_matrix']
      print('loaded a dist_matrix of shape: %s' % str(dist_matrix.shape))
  else:
    dist_matrix = style_loss.compute_pairwise_style_loss_v2(filenames)
    dist_dict = {'dist_matrix': dist_matrix}
    print('Saving distance matrix to %s' % dist_file_path)
    with open(dist_file_path, 'wb') as f:
      pickle.dump(dist_dict, f)

  # Sort neighbors for each anchor image
  num_imgs = len(dist_matrix)
  sorted_neighbors = [np.argsort(dist_matrix[ii, :]) for ii in range(num_imgs)]

  def triplet_input_fn(anchor_idx):
    # start from 1 to avoid getting the same image as its own neighbor
    positive_neighbor_idx = np.random.randint(1, k_max_nearest + 1)
    negative_neighbor_idx = num_imgs - 1 - np.random.randint(0, k_max_farthest)
    positive_img_idx = sorted_neighbors[anchor_idx][positive_neighbor_idx]
    negative_img_idx = sorted_neighbors[anchor_idx][negative_neighbor_idx]
    # Read anchor image
    anchor_rgb_path = osp.join(dataset_path, filenames[anchor_idx])
    anchor_input = read_single_appearance_input(anchor_rgb_path)
    # Read positive image
    positive_rgb_path = osp.join(dataset_path, filenames[positive_img_idx])
    positive_input = read_single_appearance_input(positive_rgb_path)
    # Read negative image
    negative_rgb_path = osp.join(dataset_path, filenames[negative_img_idx])
    negative_input = read_single_appearance_input(negative_rgb_path)
    # Return triplet
    return anchor_input, positive_input, negative_input

  return triplet_input_fn


def get_tf_triplet_dataset_iter(
    dataset_path, trainset_size, dist_file_path, batch_size=4,
    deterministic_flag=False, shuffle_buf_size=128, repeat_flag=True):
  # Create a dataset of anchor image indices.
  idx_dataset = tf.data.Dataset.range(trainset_size)
  # Create a mapper function from anchor idx to triplet images.
  triplet_mapper = lambda idx: tuple(tf.py_func(
      get_triplet_input_fn(dataset_path, dist_file_path), [idx],
      [tf.float32, tf.float32, tf.float32]))
  # Convert triplet to a dictionary for the estimator input format.
  triplet_to_dict_mapper = lambda anchor, pos, neg: {
      'anchor_img': anchor, 'positive_img': pos, 'negative_img': neg}
  if repeat_flag:
    idx_dataset = idx_dataset.repeat()  # Repeat indefinitely.
  if not deterministic_flag:
    idx_dataset = idx_dataset.shuffle(shuffle_buf_size)
    triplet_dataset = idx_dataset.map(
        triplet_mapper, num_parallel_calls=max(4, batch_size // 4))
    triplet_dataset = triplet_dataset.map(
        triplet_to_dict_mapper, num_parallel_calls=max(4, batch_size // 4))
  else:
    triplet_dataset = idx_dataset.map(triplet_mapper, num_parallel_calls=None)
    triplet_dataset = triplet_dataset.map(triplet_to_dict_mapper,
                                          num_parallel_calls=None)
  triplet_dataset = triplet_dataset.batch(batch_size)
  if not deterministic_flag:
    triplet_dataset = triplet_dataset.prefetch(4)  # Prefetch a few batches.
  return triplet_dataset.make_one_shot_iterator()


def build_model_fn(batch_size, lr_app_pretrain=0.0001, adam_beta1=0.0,
                   adam_beta2=0.99):
  def model_fn(features, labels, mode, params):
    del labels, params

    step = tf.train.get_global_step()
    app_func = networks.DRITAppearanceEncoderConcat(
      'appearance_net', opts.appearance_nc, opts.normalize_drit_Ez)

    if mode == tf.estimator.ModeKeys.TRAIN:
      op_increment_step = tf.assign_add(step, 1)
      with tf.name_scope('Appearance_Loss'):
        anchor_img = features['anchor_img']
        positive_img = features['positive_img']
        negative_img = features['negative_img']
        # Compute embeddings (each of shape [batch_sz, 1, 1, app_vector_sz])
        z_anchor, _, _ = app_func(anchor_img)
        z_pos, _, _ = app_func(positive_img)
        z_neg, _, _ = app_func(negative_img)
        # Squeeze into shape of [batch_sz x vec_sz]
        anchor_embedding = tf.squeeze(z_anchor, axis=[1, 2], name='z_anchor')
        positive_embedding = tf.squeeze(z_pos, axis=[1, 2])
        negative_embedding = tf.squeeze(z_neg, axis=[1, 2])
        # Compute triplet loss
        margin = 0.1
        anchor_positive_dist = tf.reduce_sum(
            tf.square(anchor_embedding - positive_embedding), axis=1)
        anchor_negative_dist = tf.reduce_sum(
            tf.square(anchor_embedding - negative_embedding), axis=1)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
        triplet_loss = tf.maximum(triplet_loss, 0.)
        triplet_loss = tf.reduce_sum(triplet_loss) / batch_size
        tf.summary.scalar('appearance_triplet_loss', triplet_loss)

        # Image summaries
        anchor_rgb = tf.slice(anchor_img, [0, 0, 0, 0], [-1, -1, -1, 3])
        positive_rgb = tf.slice(positive_img, [0, 0, 0, 0], [-1, -1, -1, 3])
        negative_rgb = tf.slice(negative_img, [0, 0, 0, 0], [-1, -1, -1, 3])
        tb_vis = tf.concat([anchor_rgb, positive_rgb, negative_rgb], axis=2)
        with tf.name_scope('triplet_vis'):
          tf.summary.image('anchor-pos-neg', tb_vis)

      optimizer = tf.train.AdamOptimizer(lr_app_pretrain, adam_beta1,
                                         adam_beta2)
      optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
      app_vars = utils.model_vars('appearance_net')[0]
      print('\n\n***************************************************')
      print('DBG: len(app_vars) = %d' % len(app_vars))
      for ii, v in enumerate(app_vars):
        print('%03d) %s' % (ii, str(v)))
      print('***************************************************\n\n')
      app_train_op = optimizer.minimize(triplet_loss, var_list=app_vars)
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=triplet_loss,
          train_op=tf.group(app_train_op, op_increment_step))
    elif mode == tf.estimator.ModeKeys.PREDICT:
      imgs = features['anchor_img']
      embeddings = tf.squeeze(app_func(imgs), axis=[1, 2])
      app_vars = utils.model_vars('appearance_net')[0]
      tf.train.init_from_checkpoint(osp.join(opts.train_dir),
                                    {'appearance_net/': 'appearance_net/'})
      return tf.estimator.EstimatorSpec(mode=mode, predictions=embeddings)
    else:
      raise ValueError('Unsupported mode for the appearance model: ' + mode)

  return model_fn


def compute_dist_matrix(imageset_dir, dist_file_path, recompute_dist=False):
  if not recompute_dist and osp.exists(dist_file_path):
   print('*** Loading distance matrix from %s' % dist_file_path)
   with open(dist_file_path, 'rb') as f:
     dist_matrix = pickle.load(f)['dist_matrix']
     print('loaded a dist_matrix of shape: %s' % str(dist_matrix.shape))
     return dist_matrix
  else:
    images_paths = sorted(glob.glob(osp.join(imageset_dir, '*_reference.png')))
    dist_matrix = style_loss.compute_pairwise_style_loss_v2(images_paths)
    dist_dict = {'dist_matrix': dist_matrix}
    print('Saving distance matrix to %s' % dist_file_path)
    with open(dist_file_path, 'wb') as f:
      pickle.dump(dist_dict, f)
    return dist_matrix


def train_appearance(train_dir, imageset_dir, dist_file_path):
  batch_size = 8
  lr_app_pretrain = 0.001

  trainset_size = len(glob.glob(osp.join(imageset_dir, '*_reference.png')))
  resume_step = utils.load_global_step_from_checkpoint_dir(train_dir)
  if resume_step != 0:
    tf.logging.warning('DBG: resuming apperance pretraining at %d!' %
                       resume_step)
  model_fn = build_model_fn(batch_size, lr_app_pretrain)
  config = tf.estimator.RunConfig(
      save_summary_steps=50,
      save_checkpoints_steps=500,
      keep_checkpoint_max=5,
      log_step_count_steps=100)
  est = tf.estimator.Estimator(
      tf.contrib.estimator.replicate_model_fn(model_fn), train_dir,
      config, params={})
  # Get input function
  input_train_fn = lambda: get_tf_triplet_dataset_iter(
      imageset_dir, trainset_size, dist_file_path,
      batch_size=batch_size).get_next()
  print('Starting pretraining steps...')
  est.train(input_train_fn, steps=None, hooks=None)  # train indefinitely


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  train_dir = opts.train_dir
  dataset_name = opts.dataset_name
  imageset_dir = opts.imageset_dir
  output_dir = opts.metadata_output_dir
  if not osp.exists(output_dir):
    os.makedirs(output_dir)
  dist_file_path = osp.join(output_dir, 'dist_%s.pckl' % dataset_name)
  compute_dist_matrix(imageset_dir, dist_file_path)
  train_appearance(train_dir, imageset_dir, dist_file_path)

if __name__ == '__main__':
  app.run(main)
