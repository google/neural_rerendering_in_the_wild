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
import functools
import glob
import numpy as np
import os
import os.path as osp
import skimage.measure
import tensorflow as tf
import utils

FLAGS = flags.FLAGS
flags.DEFINE_string('val_set_out_dir', None,
                    'Output directory with concatenated fake and real images.')
flags.DEFINE_string('experiment_title', 'experiment',
                    'Name for the experiment to evaluate')


def _extract_real_and_fake_from_concatenated_output(val_set_out_dir):
      out_dir = osp.join(val_set_out_dir, 'fake')
      gt_dir = osp.join(val_set_out_dir, 'real')
      if not osp.exists(out_dir):
        os.makedirs(out_dir)
      if not osp.exists(gt_dir):
        os.makedirs(gt_dir)
      imgs_pattern = osp.join(val_set_out_dir, '*.png')
      imgs_paths = sorted(glob.glob(imgs_pattern))
      print('Separating %d images in %s.' % (len(imgs_paths), val_set_out_dir))
      for img_path in imgs_paths:
        basename = osp.basename(img_path)[:-4]  # remove the '.png' suffix
        img = np.array(Image.open(img_path))
        img_res = 512
        fake = img[:, -2*img_res:-img_res, :]
        real = img[:, -img_res:, :]
        fake_path = osp.join(out_dir, '%s_fake.png' % basename)
        real_path = osp.join(gt_dir, '%s_real.png' % basename)
        Image.fromarray(fake).save(fake_path)
        Image.fromarray(real).save(real_path)


def compute_l1_loss_metric(image_set1_paths, image_set2_paths):
  assert len(image_set1_paths) == len(image_set2_paths)
  assert len(image_set1_paths) > 0
  print('Evaluating L1 loss for %d pairs' % len(image_set1_paths))

  total_loss = 0.
  for ii, (img1_path, img2_path) in enumerate(zip(image_set1_paths,
                                                  image_set2_paths)):
    img1_in_ar = np.array(Image.open(img1_path), dtype=np.float32)
    img1_in_ar = utils.crop_to_multiple(img1_in_ar)

    img2_in_ar = np.array(Image.open(img2_path), dtype=np.float32)
    img2_in_ar = utils.crop_to_multiple(img2_in_ar)

    loss_l1 = np.mean(np.abs(img1_in_ar - img2_in_ar))
    total_loss += loss_l1

  return total_loss / len(image_set1_paths)


def compute_psnr_loss_metric(image_set1_paths, image_set2_paths):
  assert len(image_set1_paths) == len(image_set2_paths)
  assert len(image_set1_paths) > 0
  print('Evaluating PSNR loss for %d pairs' % len(image_set1_paths))

  total_loss = 0.
  for ii, (img1_path, img2_path) in enumerate(zip(image_set1_paths,
                                                  image_set2_paths)):
    img1_in_ar = np.array(Image.open(img1_path))
    img1_in_ar = utils.crop_to_multiple(img1_in_ar)

    img2_in_ar = np.array(Image.open(img2_path))
    img2_in_ar = utils.crop_to_multiple(img2_in_ar)

    loss_psnr = skimage.measure.compare_psnr(img1_in_ar, img2_in_ar)
    total_loss += loss_psnr

  return total_loss / len(image_set1_paths)


def evaluate_experiment(val_set_out_dir, title='experiment',
                        metrics=['psnr', 'l1']):

  out_dir = osp.join(val_set_out_dir, 'fake')
  gt_dir = osp.join(val_set_out_dir, 'real')
  _extract_real_and_fake_from_concatenated_output(val_set_out_dir)
  input_pattern1 = osp.join(gt_dir, '*.png')
  input_pattern2 = osp.join(out_dir, '*.png')
  set1 = sorted(glob.glob(input_pattern1))
  set2 = sorted(glob.glob(input_pattern2))
  for metric in metrics:
    if metric == 'l1':
      mean_loss = compute_l1_loss_metric(set1, set2)
    elif metric == 'psnr':
      mean_loss = compute_psnr_loss_metric(set1, set2)
    print('*** mean %s loss for %s = %f' % (metric, title, mean_loss))


def main(argv):
  evaluate_experiment(FLAGS.val_set_out_dir, title=FLAGS.experiment_title,
                      metrics=['psnr', 'l1'])


if __name__ == '__main__':
  app.run(main)
