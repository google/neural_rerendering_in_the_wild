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


from absl import flags
import numpy as np

FLAGS = flags.FLAGS

# ------------------------------------------------------------------------------
# Train flags
# ------------------------------------------------------------------------------

# Dataset, model directory and run mode
flags.DEFINE_string('train_dir', '/tmp/nerual_rendering',
                    'Directory for model training.')
flags.DEFINE_string('dataset_name', 'sanmarco9k', 'name ID for a dataset.')
flags.DEFINE_string(
    'dataset_parent_dir', '',
    'Directory containing generated tfrecord dataset.')
flags.DEFINE_string('run_mode', 'train', "{'train', 'eval', 'infer'}")
flags.DEFINE_string('imageset_dir', None, 'Directory containing trainset '
                    'images for appearance pretraining.')
flags.DEFINE_string('metadata_output_dir', None, 'Directory to save pickled '
                    'pairwise distance matrix for appearance pretraining.')
flags.DEFINE_integer('save_samples_kimg', 50, 'kimg cycle to save sample'
                     'validation ouptut during training.')

# Network inputs/outputs
flags.DEFINE_boolean('use_depth', True, 'Add depth image to the deep buffer.')
flags.DEFINE_boolean('use_alpha', False,
                     'Add alpha channel to the deep buffer.')
flags.DEFINE_boolean('use_semantic', True,
                     'Add semantic map to the deep buffer.')
flags.DEFINE_boolean('use_appearance', True,
                     'Capture appearance from an input real image.')
flags.DEFINE_integer('deep_buffer_nc', 7,
                     'Number of input channels in the deep buffer.')
flags.DEFINE_integer('appearance_nc', 10,
                     'Number of input channels to the appearance encoder.')
flags.DEFINE_integer('output_nc', 3,
                     'Number of channels for the generated image.')

# Staged training flags
flags.DEFINE_string(
    'vgg16_path', './vgg16_weights/vgg16.npy',
    'path to a *.npy file with vgg16 pretrained weights')
flags.DEFINE_boolean('load_pretrained_app_encoder', False,
                     'Warmstart appearance encoder with pretrained weights.')
flags.DEFINE_string('appearance_pretrain_dir', '',
                    'Model dir for the pretrained appearance encoder.')
flags.DEFINE_boolean('train_app_encoder', False, 'Whether to make the weights '
                     'for the appearance encoder trainable or not.')
flags.DEFINE_boolean(
    'load_from_another_ckpt', False, 'Load weights from another trained model, '
                     'e.g load model trained with a fixed appearance encoder.')
flags.DEFINE_string('fixed_appearance_train_dir', '',
                    'Model dir for training G with a fixed appearance net.')

# -----------------------------------------------------------------------------

# More hparams
flags.DEFINE_integer('train_resolution', 256,
                     'Crop train images to this resolution.')
flags.DEFINE_float('d_lr', 0.001, 'Learning rate for the discriminator.')
flags.DEFINE_float('g_lr', 0.001, 'Learning rate for the generator.')
flags.DEFINE_float('ez_lr', 0.0001, 'Learning rate for appearance encoder.')
flags.DEFINE_integer('batch_size', 8, 'Batch size for training.')
flags.DEFINE_boolean('use_scaling', True, "use He's scaling.")
flags.DEFINE_integer('num_crops', 30, 'num crops from train images'
                     '(use -1 for random crops).')
flags.DEFINE_integer('app_vector_size', 8, 'Size of latent appearance vector.')
flags.DEFINE_integer('total_kimg', 20000,
                     'Max number (in kilo) of training images for training.')
flags.DEFINE_float('adam_beta1', 0.0, 'beta1 for adam optimizer.')
flags.DEFINE_float('adam_beta2', 0.99, 'beta2 for adam optimizer.')

# Loss weights
flags.DEFINE_float('w_loss_vgg', 0.3, 'VGG loss weight.')
flags.DEFINE_float('w_loss_feat', 10., 'Feature loss weight (from pix2pixHD).')
flags.DEFINE_float('w_loss_l1', 50., 'L1 loss weight.')
flags.DEFINE_float('w_loss_z_recon', 10., 'Z reconstruction loss weight.')
flags.DEFINE_float('w_loss_gan', 1., 'Adversarial loss weight.')
flags.DEFINE_float('w_loss_z_gan', 1., 'Z adversarial loss weight.')
flags.DEFINE_float('w_loss_kl', 0.01, 'KL divergence weight.')
flags.DEFINE_float('w_loss_l2_reg', 0.01, 'Weight for L2 regression on Z.')

# -----------------------------------------------------------------------------

# Architecture and training setup
flags.DEFINE_string('arch_type', 'pggan',
                    'Architecture type: {pggan, pix2pixhd}.')
flags.DEFINE_string('training_pipeline', 'staged',
                    'Training type type: {staged, bicycle_gan, drit}.')
flags.DEFINE_integer('g_nf', 64,
                     'num filters in the first/last layers of U-net.')
flags.DEFINE_boolean('concatenate_skip_layers', True,
                     'Use concatenation for skip connections.')

## if arch_type == 'pggan':
flags.DEFINE_integer('pggan_n_blocks', 5,
                     'Num blocks for the pggan architecture.')
## if arch_type == 'pix2pixhd':
flags.DEFINE_integer('p2p_n_downsamples', 3,
                     'Num downsamples for the pix2pixHD architecture.')
flags.DEFINE_integer('p2p_n_resblocks', 4, 'Num residual blocks at the '
                     'end/start of the pix2pixHD encoder/decoder.')
## if use_drit_pipeline:
flags.DEFINE_boolean('use_concat', True, '"concat" mode from DRIT.')
flags.DEFINE_boolean('normalize_drit_Ez', True, 'Add pixelnorm layers to the '
                     'appearance encoder.')
flags.DEFINE_boolean('concat_z_in_all_layers', True, 'Inject z at each '
                     'upsampling layer in the decoder (only for DRIT baseline)')
flags.DEFINE_string('inject_z', 'to_bottleneck', 'Method for injecting z; '
                     'one of {to_encoder, to_bottleneck}.')
flags.DEFINE_boolean('use_vgg_loss', True, 'vgg v L1 reconstruction loss.')

# ------------------------------------------------------------------------------
# Inference flags
# ------------------------------------------------------------------------------

flags.DEFINE_string('inference_input_path', '',
                    'Parent directory for input images at inference time.')
flags.DEFINE_string('inference_output_dir', '', 'Output path for inference')
flags.DEFINE_string('target_img_basename', '',
                    'basename of target image to render for interpolation')
flags.DEFINE_string('virtual_seq_name', 'full_camera_path',
                    'name for the virtual camera path suffix for the TFRecord.')
flags.DEFINE_string('inp_app_img_base_path', '',
                    'base path for the input appearance image for camera paths')

flags.DEFINE_string('appearance_img1_basename', '',
                    'basename of the first appearance image for interpolation')
flags.DEFINE_string('appearance_img2_basename', '',
                    'basename of the first appearance image for interpolation')
flags.DEFINE_list('input_basenames', [], 'input basenames for inference')
flags.DEFINE_list('input_app_basenames', [], 'input appearance basenames for '
                  'inference')
flags.DEFINE_string('frames_dir', '',
                    'Folder with input frames to a camera path')
flags.DEFINE_string('output_validation_dir', '',
                    'dataset_name for storing results in a structured folder')
flags.DEFINE_string('input_rendered', '',
                    'input rendered image name for inference')
flags.DEFINE_string('input_depth', '', 'input depth image name for inference')
flags.DEFINE_string('input_seg', '',
                    'input segmentation mask image name for inference')
flags.DEFINE_string('input_app_rgb', '',
                    'input appearance rgb image name for inference')
flags.DEFINE_string('input_app_rendered', '',
                    'input appearance rendered image name for inference')
flags.DEFINE_string('input_app_depth', '',
                    'input appearance depth image name for inference')
flags.DEFINE_string('input_app_seg', '',
                    'input appearance segmentation mask image name for'
                    'inference')
flags.DEFINE_string('output_img_name', '',
                    '[OPTIONAL] output image name for inference')

# -----------------------------------------------------------------------------
# Some validation and assertions
# -----------------------------------------------------------------------------

def validate_options():
  if FLAGS.use_drit_training:
    assert FLAGS.use_appearance, 'DRIT pipeline requires --use_appearance'
  assert not (
    FLAGS.load_pretrained_appearance_encoder and FLAGS.load_from_another_ckpt), (
      'You cannot load weights for the appearance encoder from two different '
      'checkpoints!')
  if not FLAGS.use_appearance:
    print('**Warning: setting --app_vector_size to 0 since '
          '--use_appearance=False!')
    FLAGS.set_default('app_vector_size', 0)
  
# -----------------------------------------------------------------------------
# Print all options
# -----------------------------------------------------------------------------

def list_options():
  configs = ('# Run flags/options from options.py:\n'
             '# ----------------------------------\n')
  configs += ('## Train flags:\n'
              '## ------------\n')
  configs += 'train_dir = %s\n' % FLAGS.train_dir
  configs += 'dataset_name = %s\n' % FLAGS.dataset_name
  configs += 'dataset_parent_dir = %s\n' % FLAGS.dataset_parent_dir
  configs += 'run_mode = %s\n' % FLAGS.run_mode
  configs += 'save_samples_kimg = %d\n' % FLAGS.save_samples_kimg
  configs += '\n# --------------------------------------------------------\n\n'

  configs += ('## Network inputs and outputs:\n'
              '## ---------------------------\n')
  configs += 'use_depth = %s\n' % str(FLAGS.use_depth)
  configs += 'use_alpha = %s\n' % str(FLAGS.use_alpha)
  configs += 'use_semantic = %s\n' % str(FLAGS.use_semantic)
  configs += 'use_appearance = %s\n' % str(FLAGS.use_appearance)
  configs += 'deep_buffer_nc = %d\n' % FLAGS.deep_buffer_nc
  configs += 'appearance_nc = %d\n' % FLAGS.appearance_nc
  configs += 'output_nc = %d\n' % FLAGS.output_nc
  configs += 'train_resolution = %d\n' % FLAGS.train_resolution
  configs += '\n# --------------------------------------------------------\n\n'

  configs += ('## Staged training flags:\n'
              '## ----------------------\n')
  configs += 'load_pretrained_app_encoder = %s\n' % str(
                                            FLAGS.load_pretrained_app_encoder)
  configs += 'appearance_pretrain_dir = %s\n' % FLAGS.appearance_pretrain_dir
  configs += 'train_app_encoder = %s\n' % str(FLAGS.train_app_encoder)
  configs += 'load_from_another_ckpt = %s\n' % str(FLAGS.load_from_another_ckpt)
  configs += 'fixed_appearance_train_dir = %s\n' % str(
                                            FLAGS.fixed_appearance_train_dir)
  configs += '\n# --------------------------------------------------------\n\n'

  configs += ('## More hyper-parameters:\n'
              '## ----------------------\n')
  configs += 'd_lr = %f\n' % FLAGS.d_lr
  configs += 'g_lr = %f\n' % FLAGS.g_lr
  configs += 'ez_lr = %f\n' % FLAGS.ez_lr
  configs += 'batch_size = %d\n' % FLAGS.batch_size
  configs += 'use_scaling = %s\n' % str(FLAGS.use_scaling)
  configs += 'num_crops = %d\n' % FLAGS.num_crops
  configs += 'app_vector_size = %d\n' % FLAGS.app_vector_size
  configs += 'total_kimg = %d\n' % FLAGS.total_kimg
  configs += 'adam_beta1 = %f\n' % FLAGS.adam_beta1
  configs += 'adam_beta2 = %f\n' % FLAGS.adam_beta2
  configs += '\n# --------------------------------------------------------\n\n'

  configs += ('## Loss weights:\n'
              '## -------------\n')
  configs += 'w_loss_vgg = %f\n' % FLAGS.w_loss_vgg
  configs += 'w_loss_feat = %f\n' % FLAGS.w_loss_feat
  configs += 'w_loss_l1 = %f\n' % FLAGS.w_loss_l1
  configs += 'w_loss_z_recon = %f\n' % FLAGS.w_loss_z_recon
  configs += 'w_loss_gan = %f\n' % FLAGS.w_loss_gan
  configs += 'w_loss_z_gan = %f\n' % FLAGS.w_loss_z_gan
  configs += 'w_loss_kl = %f\n' % FLAGS.w_loss_kl
  configs += 'w_loss_l2_reg = %f\n' % FLAGS.w_loss_l2_reg
  configs += '\n# --------------------------------------------------------\n\n'

  configs += ('## Architecture and training setup:\n'
              '## --------------------------------\n')
  configs += 'arch_type = %s\n' % FLAGS.arch_type
  configs += 'training_pipeline = %s\n' % FLAGS.training_pipeline
  configs += 'g_nf = %d\n' % FLAGS.g_nf
  configs += 'concatenate_skip_layers = %s\n' % str(
                                                FLAGS.concatenate_skip_layers)
  configs += 'p2p_n_downsamples = %d\n' % FLAGS.p2p_n_downsamples
  configs += 'p2p_n_resblocks = %d\n' % FLAGS.p2p_n_resblocks
  configs += 'use_concat = %s\n' % str(FLAGS.use_concat)
  configs += 'normalize_drit_Ez = %s\n' % str(FLAGS.normalize_drit_Ez)
  configs += 'inject_z = %s\n' % FLAGS.inject_z
  configs += 'concat_z_in_all_layers = %s\n' % str(FLAGS.concat_z_in_all_layers)
  configs += 'use_vgg_loss = %s\n' % str(FLAGS.use_vgg_loss)
  configs += '\n# --------------------------------------------------------\n\n'

  return configs
