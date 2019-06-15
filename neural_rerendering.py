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
from options import FLAGS as opts
import data
import datetime
import functools
import glob
import losses
import networks
import numpy as np
import options
import os.path as osp
import random
import skimage.measure
import staged_model
import tensorflow as tf
import time
import utils


def build_model_fn(use_exponential_moving_average=True):
  """Builds and returns the model function for an estimator.

  Args:
    use_exponential_moving_average: bool. If true, the exponential moving
    average will be used.

  Returns:
    function, the model_fn function typically required by an estimator.
  """
  arch_type = opts.arch_type
  use_appearance = opts.use_appearance
  def model_fn(features, labels, mode, params):
    """An estimator build_fn."""
    del labels, params
    if mode == tf.estimator.ModeKeys.TRAIN:
      step = tf.train.get_global_step()

      x_in = features['conditional_input']
      x_gt = features['expected_output']  # ground truth output
      x_app = features['peek_input']

      if opts.training_pipeline == 'staged':
        ops = staged_model.create_computation_graph(x_in, x_gt, x_app=x_app,
                                                    arch_type=opts.arch_type)
        op_increment_step = tf.assign_add(step, 1)
        train_disc_op = ops['train_disc_op']
        train_renderer_op = ops['train_renderer_op']
        train_op = tf.group(train_disc_op, train_renderer_op, op_increment_step)

        utils.HookReport.log_tensor(ops['total_loss_d'], 'total_loss_d')
        utils.HookReport.log_tensor(ops['loss_d_real'], 'loss_d_real')
        utils.HookReport.log_tensor(ops['loss_d_fake'], 'loss_d_fake')
        utils.HookReport.log_tensor(ops['total_loss_g'], 'total_loss_g')
        utils.HookReport.log_tensor(ops['loss_g_gan'], 'loss_g_gan')
        utils.HookReport.log_tensor(ops['loss_g_recon'], 'loss_g_recon')
        utils.HookReport.log_tensor(step, 'global_step')

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=ops['total_loss_d'] + ops['total_loss_g'],
            train_op=train_op)
      else:
        raise NotImplementedError('%s training is not implemented.' %
                                  opts.training_pipeline)
    elif mode == tf.estimator.ModeKeys.EVAL:
      raise NotImplementedError('Eval is not implemented.')
    else:  # all below modes are for difference inference tasks.
      # Build network and initialize inference variables.
      g_func = networks.RenderingModel(arch_type, use_appearance)
      if use_appearance:
        app_func = g_func.get_appearance_encoder()
      if use_exponential_moving_average:
        ema = tf.train.ExponentialMovingAverage(decay=0.999)
        var_dict = ema.variables_to_restore()
        tf.train.init_from_checkpoint(osp.join(opts.train_dir), var_dict)

      if mode == tf.estimator.ModeKeys.PREDICT:
        x_in = features['conditional_input']
        if use_appearance:
          x_app = features['peek_input']
          x_app_embedding, _, _ = app_func(x_app)
        else:
          x_app_embedding = None
        y = g_func(x_in, x_app_embedding)
        tf.logging.info('DBG: shape of y during prediction %s.' % str(y.shape))
        return tf.estimator.EstimatorSpec(mode=mode, predictions=y)

      # 'eval_subset' mode is same as PREDICT but it concatenates the output to
      # the input render, semantic map and ground truth for easy comparison.
      elif mode == 'eval_subset':
        x_in = features['conditional_input']
        x_gt = features['expected_output']
        if use_appearance:
          x_app = features['peek_input']
          x_app_embedding, _, _ = app_func(x_app)
        else:
          x_app_embedding = None
        y = g_func(x_in, x_app_embedding)
        tf.logging.info('DBG: shape of y during prediction %s.' % str(y.shape))
        x_in_rgb = tf.slice(x_in, [0, 0, 0, 0], [-1, -1, -1, 3])
        if opts.use_semantic:
          x_in_semantic = tf.slice(x_in, [0, 0, 0, 4], [-1, -1, -1, 3])
          output_tuple = tf.concat([x_in_rgb, x_in_semantic, y, x_gt], axis=2)
        else:
          output_tuple = tf.concat([x_in_rgb, y, x_gt], axis=2)
        return tf.estimator.EstimatorSpec(mode=mode, predictions=output_tuple)

      # 'compute_appearance' mode computes and returns the latent z vector.
      elif mode == 'compute_appearance':
        assert use_appearance, 'use_appearance is set to False!'
        x_app_in = features['peek_input']
        # NOTE the following line is a temporary hack (which is
        # specially bad for inputs smaller than 512x512).
        x_app_in = tf.image.resize_image_with_crop_or_pad(x_app_in, 512, 512)
        app_embedding, _, _ = app_func(x_app_in)
        return tf.estimator.EstimatorSpec(mode=mode, predictions=app_embedding)

      # 'interpolate_appearance' mode expects an already computed latent z
      # vector as input passed a value to the dict key 'appearance_embedding'.
      elif mode == 'interpolate_appearance':
        assert use_appearance, 'use_appearance is set to False!'
        x_in = features['conditional_input']
        x_app_embedding = features['appearance_embedding']
        y = g_func(x_in, x_app_embedding)
        tf.logging.info('DBG: shape of y during prediction %s.' % str(y.shape))
        return tf.estimator.EstimatorSpec(mode=mode, predictions=y)
      else:
        raise ValueError('Unsupported mode: ' + mode)

  return model_fn


def make_sample_grid_and_save(est, dataset_name, dataset_parent_dir, grid_dims,
                              output_dir, cur_nimg):
  """Evaluate a fixed set of validation images and save output.

  Args:
    est: tf,estimator.Estimator, TF estimator to run the predictions.
    dataset_name: basename for the validation tfrecord from which to load
      validation images.
    dataset_parent_dir: path to a directory containing the validation tfrecord.
    grid_dims: 2-tuple int for the grid size (1 unit = 1 image).
    output_dir: string, where to save image samples.
    cur_nimg: int, current number of images seen by training.

  Returns:
    None.
  """
  num_examples = grid_dims[0] * grid_dims[1]
  def input_val_fn():
    dict_inp = data.provide_data(
        dataset_name=dataset_name, parent_dir=dataset_parent_dir, subset='val',
        batch_size=1, crop_flag=True, crop_size=opts.train_resolution,
        seeds=[0], max_examples=num_examples,
        use_appearance=opts.use_appearance, shuffle=0)
    x_in = dict_inp['conditional_input']
    x_gt = dict_inp['expected_output']  # ground truth output
    x_app = dict_inp['peek_input']
    return x_in, x_gt, x_app

  def est_input_val_fn():
    x_in, _, x_app = input_val_fn()
    features = {'conditional_input': x_in, 'peek_input': x_app}
    return features

  images = [x for x in est.predict(est_input_val_fn)]
  images = np.array(images, 'f')
  images = images.reshape(grid_dims + images.shape[1:])
  utils.save_images(utils.to_png(utils.images_to_grid(images)), output_dir,
                    cur_nimg)


def visualize_image_sequence(est, dataset_name, dataset_parent_dir,
                             input_sequence_name, app_base_path, output_dir):
  """Generates an image sequence as a video and stores it to disk."""
  batch_sz = opts.batch_size
  def input_seq_fn():
    dict_inp = data.provide_data(
        dataset_name=dataset_name, parent_dir=dataset_parent_dir,
        subset=input_sequence_name, batch_size=batch_sz, crop_flag=False,
        seeds=None, use_appearance=False, shuffle=0)
    x_in = dict_inp['conditional_input']
    return x_in

  # Compute appearance embedding only once and use it for all input frames.
  app_rgb_path = app_base_path + '_reference.png'
  app_rendered_path = app_base_path + '_color.png'
  app_depth_path = app_base_path + '_depth.png'
  app_sem_path = app_base_path + '_seg_rgb.png'
  x_app = _load_and_concatenate_image_channels(
      app_rgb_path, app_rendered_path, app_depth_path, app_sem_path)
  def seq_with_single_appearance_inp_fn():
    """input frames with a fixed latent appearance vector."""
    x_in_op = input_seq_fn()
    x_app_op = tf.convert_to_tensor(x_app)
    x_app_tiled_op = tf.tile(x_app_op, [tf.shape(x_in_op)[0], 1, 1, 1])
    return {'conditional_input': x_in_op,
            'peek_input': x_app_tiled_op}

  images = [x for x in est.predict(seq_with_single_appearance_inp_fn)]
  for i, gen_img in enumerate(images):
    output_file_path = osp.join(output_dir, 'out_%04d.png' % i)
    print('Saving frame #%d to %s' % (i, output_file_path))
    with tf.gfile.Open(output_file_path, 'wb') as f:
      f.write(utils.to_png(gen_img))


def train(dataset_name, dataset_parent_dir, load_pretrained_app_encoder,
          load_trained_fixed_app, save_samples_kimg=50):
  """Main training procedure.

  The trained model is saved in opts.train_dir, the function itself does not
   return anything.

  Args:
    save_samples_kimg: int, period (in KiB) to save sample images.

  Returns:
    None.
  """
  image_dir = osp.join(opts.train_dir, 'images')  # to save validation images.
  tf.gfile.MakeDirs(image_dir)
  config = tf.estimator.RunConfig(
      save_summary_steps=(1 << 10) // opts.batch_size,
      save_checkpoints_steps=(save_samples_kimg << 10) // opts.batch_size,
      keep_checkpoint_max=5,
      log_step_count_steps=1 << 30)
  model_dir = opts.train_dir
  if (opts.use_appearance and load_trained_fixed_app and
      not tf.train.latest_checkpoint(model_dir)):
    tf.logging.warning('***** Loading resume_step from %s!' %
                       opts.fixed_appearance_train_dir)
    resume_step = utils.load_global_step_from_checkpoint_dir(
        opts.fixed_appearance_train_dir)
  else:
    tf.logging.warning('***** Loading resume_step (if any) from %s!' %
                       model_dir)
    resume_step = utils.load_global_step_from_checkpoint_dir(model_dir)
  if resume_step != 0:
    tf.logging.warning('****** Resuming training at %d!' % resume_step)

  model_fn = build_model_fn()  # model function for TFEstimator.

  hooks = [utils.HookReport(1 << 12, opts.batch_size)]

  if opts.use_appearance and load_pretrained_app_encoder:
    tf.logging.warning('***** will warm-start from %s!' %
                       opts.appearance_pretrain_dir)
    ws = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=opts.appearance_pretrain_dir,
        vars_to_warm_start='appearance_net/.*')
  elif opts.use_appearance and load_trained_fixed_app:
    tf.logging.warning('****** finetuning will warm-start from %s!' %
                       opts.fixed_appearance_train_dir)
    ws = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=opts.fixed_appearance_train_dir,
        vars_to_warm_start='.*')
  else:
    ws = None
    tf.logging.warning('****** No warm-starting; using random initialization!')

  est = tf.estimator.Estimator(model_fn, model_dir, config, params={},
                               warm_start_from=ws)

  for next_kimg in range(opts.save_samples_kimg, opts.total_kimg + 1,
                         opts.save_samples_kimg):
    next_step = (next_kimg << 10) // opts.batch_size
    if opts.num_crops == -1:  # use random crops
      crop_seeds = None
    else:
      crop_seeds = list(100 * np.arange(opts.num_crops))
    input_train_fn = functools.partial(
        data.provide_data, dataset_name=dataset_name,
        parent_dir=dataset_parent_dir, subset='train',
        batch_size=opts.batch_size, crop_flag=True,
        crop_size=opts.train_resolution, seeds=crop_seeds,
        use_appearance=opts.use_appearance)
    est.train(input_train_fn, max_steps=next_step, hooks=hooks)
    tf.logging.info('DBG: kimg=%d, cur_step=%d' % (next_kimg, next_step))
    tf.logging.info('DBG: Saving a validation grid image %06d to %s' % (
        next_kimg, image_dir))
    make_sample_grid_and_save(est, dataset_name, dataset_parent_dir, (3, 3),
                              image_dir, next_kimg << 10)


def _build_inference_estimator(model_dir):
  model_fn = build_model_fn()
  est = tf.estimator.Estimator(model_fn, model_dir)
  return est


def evaluate_sequence(dataset_name, dataset_parent_dir, virtual_seq_name,
                      app_base_path):
  output_dir = osp.join(opts.train_dir, 'seq_output_%s' % virtual_seq_name)
  tf.gfile.MakeDirs(output_dir)
  est = _build_inference_estimator(opts.train_dir)
  visualize_image_sequence(est, dataset_name, dataset_parent_dir,
                           virtual_seq_name, app_base_path, output_dir)


def evaluate_image_set(dataset_name, dataset_parent_dir, subset_suffix,
                       output_dir=None, batch_size=6):
  if output_dir is None:
    output_dir = osp.join(opts.train_dir, 'validation_output_%s' % subset_suffix)
  tf.gfile.MakeDirs(output_dir)
  model_fn_old = build_model_fn()
  def model_fn_wrapper(features, labels, mode, params):
    del mode
    return model_fn_old(features, labels, 'eval_subset', params)
  model_dir = opts.train_dir
  est = tf.estimator.Estimator(model_fn_wrapper, model_dir)
  est_inp_fn = functools.partial(
      data.provide_data, dataset_name=dataset_name,
      parent_dir=dataset_parent_dir, subset=subset_suffix,
      batch_size=batch_size, use_appearance=opts.use_appearance, shuffle=0)

  print('Evaluating images for subset %s' % subset_suffix)
  images = [x for x in est.predict(est_inp_fn)]
  print('Evaluated %d images' % len(images))
  for i, img in enumerate(images):
    output_file_path = osp.join(output_dir, 'out_%04d.png' % i)
    print('Saving file #%d: %s' % (i, output_file_path))
    with tf.gfile.Open(output_file_path, 'wb') as f:
      f.write(utils.to_png(img))


def _load_and_concatenate_image_channels(rgb_path=None, rendered_path=None,
                                         depth_path=None, seg_path=None,
                                         size_multiple=64):
  """Prepares a single input for the network."""
  if (rgb_path is None and rendered_path is None and depth_path is None and
      seg_path is None):
    raise ValueError('At least one of the inputs has to be not None')

  channels = ()
  if rgb_path is not None:
    rgb_img = np.array(Image.open(rgb_path)).astype(np.float32)
    rgb_img = utils.crop_to_multiple(rgb_img, size_multiple)
    channels = channels + (rgb_img,)
  if rendered_path is not None:
    rendered_img = np.array(Image.open(rendered_path)).astype(np.float32)
    if not opts.use_alpha:
      rendered_img = rendered_img[:, :, :3]  # drop the alpha channel
    rendered_img = utils.crop_to_multiple(rendered_img, size_multiple)
    channels = channels + (rendered_img,)
  if depth_path is not None:
    depth_img = np.array(Image.open(depth_path))
    depth_img = depth_img.astype(np.float32)
    depth_img = utils.crop_to_multiple(depth_img[:, :, np.newaxis],
                                       size_multiple)
    channels = channels + (depth_img,)
    # depth_img = depth_img * (2.0 / 255) - 1.0
  if seg_path is not None:
    seg_img = np.array(Image.open(seg_path)).astype(np.float32)
    seg_img = utils.crop_to_multiple(seg_img, size_multiple)
    channels = channels + (seg_img,)
  # Concatenate and normalize channels
  img = np.dstack(channels)
  img = np.expand_dims(img, axis=0)
  img = img * (2.0 / 255) - 1.0
  return img


def infer_dir(model_dir, input_dir, output_dir):
  tf.gfile.MakeDirs(output_dir)
  est = _build_inference_estimator(opts.train_dir)

  def read_image(base_path, is_appearance=False):
    if is_appearance:
      ref_img_path = base_path + '_reference.png'
    else:
      ref_img_path = None
    rendered_img_path = base_path + '_color.png'
    depth_img_path = base_path + '_depth.png'
    seg_img_path = base_path + '_seg_rgb.png'
    img = _load_and_concatenate_image_channels(
        rgb_path=ref_img_path, rendered_path=rendered_img_path,
        depth_path=depth_img_path, seg_path=seg_img_path)
    return img

  def get_inference_input_fn(base_path, app_base_path):
    x_in = read_image(base_path, False)
    x_app_in = read_image(app_base_path, True)
    def infer_input_fn():
      return {'conditional_input': x_in, 'peek_input': x_app_in}
    return infer_input_fn

  file_paths = sorted(glob.glob(osp.join(input_dir, '*_depth.png')))
  base_paths = [x[:-10] for x in file_paths]  # remove the '_depth.png' suffix
  for inp_base_path in base_paths:
    est_inp_fn = get_inference_input_fn(inp_base_path, inp_base_path)
    img = next(est.predict(est_inp_fn))
    basename = osp.basename(inp_base_path)
    output_img_path = osp.join(output_dir, basename + '_out.png')
    print('Saving generated image to %s' % output_img_path)
    with tf.gfile.Open(output_img_path, 'wb') as f:
      f.write(utils.to_png(img))


def joint_interpolation(model_dir, app_input_dir, st_app_basename,
                        end_app_basename, camera_path_dir):
  """
  Interpolates both viewpoint and appearance between two input images.
  """
  # Create output direcotry
  output_dir = osp.join(model_dir, 'joint_interpolation_out')
  tf.gfile.MakeDirs(output_dir)

  # Build estimator
  model_fn_old = build_model_fn()
  def model_fn_wrapper(features, labels, mode, params):
    del mode
    return model_fn_old(features, labels, 'interpolate_appearance', params)
  def appearance_model_fn(features, labels, mode, params):
    del mode
    return model_fn_old(features, labels, 'compute_appearance', params)
  config = tf.estimator.RunConfig(
      save_summary_steps=1000, save_checkpoints_steps=50000,
      keep_checkpoint_max=50, log_step_count_steps=1 << 30)
  model_dir = model_dir
  est = tf.estimator.Estimator(model_fn_wrapper, model_dir, config, params={})
  est_app = tf.estimator.Estimator(appearance_model_fn, model_dir, config,
                                   params={})

  # Compute appearance embeddings for the two input appearance images.
  app_inputs = []
  for app_basename in [st_app_basename, end_app_basename]:
    app_rgb_path = osp.join(app_input_dir, app_basename + '_reference.png')
    app_rendered_path = osp.join(app_input_dir, app_basename + '_color.png')
    app_depth_path = osp.join(app_input_dir, app_basename + '_depth.png')
    app_seg_path = osp.join(app_input_dir, app_basename + '_seg_rgb.png')
    app_in = _load_and_concatenate_image_channels(
        rgb_path=app_rgb_path, rendered_path=app_rendered_path,
        depth_path=app_depth_path, seg_path=app_seg_path)
    # app_inputs.append(tf.convert_to_tensor(app_in))
    app_inputs.append(app_in)

  embedding1 = next(est_app.predict(
      lambda: {'peek_input': app_inputs[0]}))
  embedding1 = np.expand_dims(embedding1, axis=0)
  embedding2 = next(est_app.predict(
      lambda: {'peek_input': app_inputs[1]}))
  embedding2 = np.expand_dims(embedding2, axis=0)

  file_paths = sorted(glob.glob(osp.join(camera_path_dir, '*_depth.png')))
  base_paths = [x[:-10] for x in file_paths]  # remove the '_depth.png' suffix

  # Compute interpolated appearance embeddings
  num_interpolations = len(base_paths)
  interpolated_embeddings = []
  delta_vec = (embedding2 - embedding1) / (num_interpolations - 1)
  for delta_iter in range(num_interpolations):
    x_app_embedding = embedding1 + delta_iter * delta_vec
    interpolated_embeddings.append(x_app_embedding)

  # Generate and save interpolated images
  for frame_idx, embedding in enumerate(interpolated_embeddings):
    # Read in input frame
    frame_render_path = osp.join(base_paths[frame_idx] + '_color.png')
    frame_depth_path = osp.join(base_paths[frame_idx] + '_depth.png')
    frame_seg_path = osp.join(base_paths[frame_idx] + '_seg_rgb.png')
    x_in = _load_and_concatenate_image_channels(
        rgb_path=None, rendered_path=frame_render_path,
        depth_path=frame_depth_path, seg_path=frame_seg_path)

    img = next(est.predict(
        lambda: {'conditional_input': tf.convert_to_tensor(x_in),
                 'appearance_embedding': tf.convert_to_tensor(embedding)}))
    output_img_name = '%s_%s_%03d.png' % (st_app_basename, end_app_basename,
                                          frame_idx)
    output_img_path = osp.join(output_dir, output_img_name)
    print('Saving interpolated image to %s' % output_img_path)
    with tf.gfile.Open(output_img_path, 'wb') as f:
      f.write(utils.to_png(img))


def interpolate_appearance(model_dir, input_dir, target_img_basename,
                           appearance_img1_basename, appearance_img2_basename):
  # Create output direcotry
  output_dir = osp.join(model_dir, 'interpolate_appearance_out')
  tf.gfile.MakeDirs(output_dir)

  # Build estimator
  model_fn_old = build_model_fn()
  def model_fn_wrapper(features, labels, mode, params):
    del mode
    return model_fn_old(features, labels, 'interpolate_appearance', params)
  def appearance_model_fn(features, labels, mode, params):
    del mode
    return model_fn_old(features, labels, 'compute_appearance', params)
  config = tf.estimator.RunConfig(
      save_summary_steps=1000, save_checkpoints_steps=50000,
      keep_checkpoint_max=50, log_step_count_steps=1 << 30)
  model_dir = model_dir
  est = tf.estimator.Estimator(model_fn_wrapper, model_dir, config, params={})
  est_app = tf.estimator.Estimator(appearance_model_fn, model_dir, config,
                                   params={})

  # Compute appearance embeddings for the two input appearance images.
  app_inputs = []
  for app_basename in [appearance_img1_basename, appearance_img2_basename]:
    app_rgb_path = osp.join(input_dir, app_basename + '_reference.png')
    app_rendered_path = osp.join(input_dir, app_basename + '_color.png')
    app_depth_path = osp.join(input_dir, app_basename + '_depth.png')
    app_seg_path = osp.join(input_dir, app_basename + '_seg_rgb.png')
    app_in = _load_and_concatenate_image_channels(
        rgb_path=app_rgb_path, rendered_path=app_rendered_path,
        depth_path=app_depth_path, seg_path=app_seg_path)
    # app_inputs.append(tf.convert_to_tensor(app_in))
    app_inputs.append(app_in)

  embedding1 = next(est_app.predict(
      lambda: {'peek_input': app_inputs[0]}))
  embedding2 = next(est_app.predict(
      lambda: {'peek_input': app_inputs[1]}))
  embedding1 = np.expand_dims(embedding1, axis=0)
  embedding2 = np.expand_dims(embedding2, axis=0)

  # Compute interpolated appearance embeddings
  num_interpolations = 10
  interpolated_embeddings = []
  delta_vec = (embedding2 - embedding1) / num_interpolations
  for delta_iter in range(num_interpolations + 1):
    x_app_embedding = embedding1 + delta_iter * delta_vec
    interpolated_embeddings.append(x_app_embedding)

  # Read in the generator input for the target image to render
  rendered_img_path = osp.join(input_dir, target_img_basename + '_color.png')
  depth_img_path = osp.join(input_dir, target_img_basename + '_depth.png')
  seg_img_path = osp.join(input_dir, target_img_basename + '_seg_rgb.png')
  x_in = _load_and_concatenate_image_channels(
      rgb_path=None, rendered_path=rendered_img_path,
      depth_path=depth_img_path, seg_path=seg_img_path)

  # Generate and save interpolated images
  for interpolate_iter, embedding in enumerate(interpolated_embeddings):
    img = next(est.predict(
        lambda: {'conditional_input': tf.convert_to_tensor(x_in),
                 'appearance_embedding': tf.convert_to_tensor(embedding)}))
    output_img_name = 'interpolate_%s_%s_%s_%03d.png' % (
        target_img_basename, appearance_img1_basename, appearance_img2_basename,
        interpolate_iter)
    output_img_path = osp.join(output_dir, output_img_name)
    print('Saving interpolated image to %s' % output_img_path)
    with tf.gfile.Open(output_img_path, 'wb') as f:
      f.write(utils.to_png(img))


def main(argv):
  del argv
  configs_str = options.list_options()
  tf.gfile.MakeDirs(opts.train_dir)
  with tf.gfile.Open(osp.join(opts.train_dir, 'configs.txt'), 'wb') as f:
    f.write(configs_str)
  tf.logging.info('Local configs\n%s' % configs_str)

  if opts.run_mode == 'train':
    dataset_name = opts.dataset_name
    dataset_parent_dir = opts.dataset_parent_dir
    load_pretrained_app_encoder = opts.load_pretrained_app_encoder
    load_trained_fixed_app = opts.load_from_another_ckpt
    batch_size = opts.batch_size
    train(dataset_name, dataset_parent_dir, load_pretrained_app_encoder,
          load_trained_fixed_app)
  elif opts.run_mode == 'eval':  # generate a camera path output sequence from TFRecord inputs.
    dataset_name = opts.dataset_name
    dataset_parent_dir = opts.dataset_parent_dir
    virtual_seq_name = opts.virtual_seq_name
    inp_app_img_base_path = opts.inp_app_img_base_path
    evaluate_sequence(dataset_name, dataset_parent_dir, virtual_seq_name,
                      inp_app_img_base_path)
  elif opts.run_mode == 'eval_subset':  # generate output for validation set (encoded as TFRecords)
    dataset_name = opts.dataset_name
    dataset_parent_dir = opts.dataset_parent_dir
    virtual_seq_name = opts.virtual_seq_name
    evaluate_image_set(dataset_name, dataset_parent_dir, virtual_seq_name,
                       opts.output_validation_dir, opts.batch_size)
  elif opts.run_mode == 'eval_dir':  # evaluate output for a directory with input images
    input_dir = opts.inference_input_path
    output_dir = opts.inference_output_dir
    model_dir = opts.train_dir
    infer_dir(model_dir, input_dir, output_dir)
  elif opts.run_mode == 'interpolate_appearance':  # interpolate appearance only between two images.
    model_dir = opts.train_dir
    input_dir = opts.inference_input_path
    target_img_basename = opts.target_img_basename
    app_img1_basename = opts.appearance_img1_basename
    app_img2_basename = opts.appearance_img2_basename
    interpolate_appearance(model_dir, input_dir, target_img_basename,
                           app_img1_basename, app_img2_basename)
  elif opts.run_mode == 'joint_interpolation':  # interpolate viewpoint and appearance between two images
    model_dir = opts.train_dir
    app_input_dir = opts.inference_input_path
    st_app_basename = opts.appearance_img1_basename
    end_app_basename = opts.appearance_img2_basename
    frames_dir = opts.frames_dir
    joint_interpolation(model_dir, app_input_dir, st_app_basename,
                        end_app_basename, frames_dir)
  else:
    raise ValueError('Unsupported --run_mode %s' % opts.run_mode)


if __name__ == '__main__':
  app.run(main)
