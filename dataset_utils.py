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
import cv2
import data
import functools
import glob
import numpy as np
import os
import os.path as osp
import shutil
import six
import tensorflow as tf
import segment_dataset as segment_utils
import utils

FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', None, 'Directory to save exported tfrecords.')
flags.DEFINE_string('xception_frozen_graph_path', None,
                    'Path to the deeplab xception model frozen graph')


class AlignedRenderedDataset(object):
  def __init__(self, rendered_filepattern, use_semantic_map=True):
    """
    Args:
      rendered_filepattern: string, path filepattern to 3D rendered images (
        assumes filenames are '/path/to/dataset/%d_color.png')
      use_semantic_map: bool, include semantic maps. in the TFRecord
    """
    self.filenames = sorted(glob.glob(rendered_filepattern))
    assert len(self.filenames) > 0, ('input %s didn\'t match any files!' %
                                     rendered_filepattern)
    self.iter_idx = 0
    self.use_semantic_map = use_semantic_map

  def __iter__(self):
    return self

  def __next__(self):
    return self.next()

  def next(self):
    if self.iter_idx < len(self.filenames):
      rendered_img_name = self.filenames[self.iter_idx]
      basename = rendered_img_name[:-9]  # remove the 'color.png' suffix
      ref_img_name = basename + 'reference.png'
      depth_img_name = basename + 'depth.png'
      # Read the 3D rendered image
      img_rendered = cv2.imread(rendered_img_name, cv2.IMREAD_UNCHANGED)
      # Change BGR (default cv2 format) to RGB
      img_rendered = img_rendered[:, :, [2,1,0,3]]  # it has a 4th alpha channel
      # Read the depth image
      img_depth = cv2.imread(depth_img_name, cv2.IMREAD_UNCHANGED)
      # Workaround as some depth images are read with a different data type!
      img_depth = img_depth.astype(np.uint16)
      # Read reference image if exists, otherwise replace with a zero image.
      if osp.exists(ref_img_name):
        img_ref = cv2.imread(ref_img_name)
        img_ref = img_ref[:, :, ::-1]  # Change BGR to RGB format.
      else:  # use a dummy 3-channel zero image as a placeholder
        print('Warning: no reference image found! Using a dummy placeholder!')
        img_height, img_width = img_depth.shape
        img_ref = np.zeros((img_height, img_width, 3), dtype=np.uint8)

      if self.use_semantic_map:
        semantic_seg_img_name = basename + 'seg_rgb.png'
        img_seg = cv2.imread(semantic_seg_img_name)
        img_seg = img_seg[:, :, ::-1]  # Change from BGR to RGB
        if img_seg.shape[0] == 512 and img_seg.shape[1] == 512:
          img_ref = utils.get_central_crop(img_ref)
          img_rendered = utils.get_central_crop(img_rendered)
          img_depth = utils.get_central_crop(img_depth)

      img_shape = img_depth.shape
      assert img_seg.shape == (img_shape + (3,)), 'error in seg image %s %s' % (
        basename, str(img_seg.shape))
      assert img_ref.shape == (img_shape + (3,)), 'error in ref image %s %s' % (
        basename, str(img_ref.shape))
      assert img_rendered.shape == (img_shape + (4,)), ('error in rendered '
        'image %s %s' % (basename, str(img_rendered.shape)))
      assert len(img_depth.shape) == 2, 'error in depth image %s %s' % (
        basename, str(img_depth.shape))

      raw_example = dict()
      raw_example['height'] = img_ref.shape[0]
      raw_example['width'] = img_ref.shape[1]
      raw_example['rendered'] = img_rendered.tostring()
      raw_example['depth'] = img_depth.tostring()
      raw_example['real'] = img_ref.tostring()
      if self.use_semantic_map:
        raw_example['seg'] = img_seg.tostring()
      self.iter_idx += 1
      return raw_example
    else:
      raise StopIteration()


def filter_out_sparse_renders(dataset_dir, splits, ratio_threshold=0.15):
  print('Filtering %s' % dataset_dir)
  if splits is None:
    imgs_dirs = [dataset_dir]
  else:
    imgs_dirs = [osp.join(dataset_dir, split) for split in splits]
  
  filtered_images = []
  total_images = 0
  sum_density = 0
  for cur_dir in imgs_dirs:
    filtered_dir = osp.join(cur_dir, 'sparse_renders')
    if not osp.exists(filtered_dir):
      os.makedirs(filtered_dir)
    imgs_file_pattern = osp.join(cur_dir, '*_color.png')
    images_path = sorted(glob.glob(imgs_file_pattern))
    print('Processing %d files' % len(images_path))
    total_images += len(images_path)
    for ii, img_path in enumerate(images_path):
      img = np.array(Image.open(img_path))
      aggregate = np.squeeze(np.sum(img, axis=2))
      height, width = aggregate.shape
      mask = aggregate > 0
      density = np.sum(mask) * 1. / (height * width)
      sum_density += density
      if density <= ratio_threshold:
        parent, basename = osp.split(img_path)
        basename = basename[:-10]  # remove the '_color.png' suffix
        srcs = sorted(glob.glob(osp.join(parent, basename + '_*')))
        dest = unicode(filtered_dir + '/.')
        for src in srcs:
          shutil.move(src, dest)
        filtered_images.append(basename)
        print('filtered fie %d: %s with a desnity of %.3f' % (ii, basename,
                                                              density))
    print('Filtered %d/%d images' % (len(filtered_images), total_images))
    print('Mean desnity = %.4f' % (sum_density / total_images))


def _to_example(dictionary):
  """Helper: build tf.Example from (string -> int/float/str list) dictionary."""
  features = {}
  for (k, v) in six.iteritems(dictionary):
    if isinstance(v, six.integer_types):
      features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))
    elif isinstance(v, float):
      features[k] = tf.train.Feature(float_list=tf.train.FloatList(value=[v]))
    elif isinstance(v, six.string_types):
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
    elif isinstance(v, bytes):
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
    else:
      raise ValueError("Value for %s is not a recognized type; v: %s type: %s" %
                       (k, str(v[0]), str(type(v[0]))))

  return tf.train.Example(features=tf.train.Features(feature=features))


def _generate_tfrecord_dataset(generator,
                              output_name,
                              output_dir):
  """Convert a dataset into TFRecord format."""
  output_filename = os.path.join(output_dir, output_name)
  output_file = os.path.join(output_dir, output_filename)
  tf.logging.info("Writing TFRecords to file %s", output_file)
  writer = tf.python_io.TFRecordWriter(output_file)

  counter = 0
  for case in generator:
    if counter % 100 == 0:
      print('Generating case %d for %s.' % (counter, output_name))
    counter += 1
    example = _to_example(case)
    writer.write(example.SerializeToString())

  writer.close()
  return output_file


def export_aligned_dataset_to_tfrecord(
    dataset_dir, output_dir, output_basename, splits,
    xception_frozen_graph_path):

  # Step 1: filter out sparse renders
  filter_out_sparse_renders(dataset_dir, splits, 0.15)

  # Step 2: generate semantic segmentation masks
  segment_utils.segment_and_color_dataset(
      dataset_dir, xception_frozen_graph_path, splits)

  # Step 3: export dataset to TFRecord
  if splits is None:
    input_filepattern = osp.join(dataset_dir, '*_color.png')
    dataset_iter = AlignedRenderedDataset(input_filepattern)
    output_name = output_basename + '.tfrecord'
    _generate_tfrecord_dataset(dataset_iter, output_name, output_dir)
  else:
    for split in splits:
      input_filepattern = osp.join(dataset_dir, split, '*_color.png')
      dataset_iter = AlignedRenderedDataset(input_filepattern)
      output_name = '%s_%s.tfrecord' % (output_basename, split)
      _generate_tfrecord_dataset(dataset_iter, output_name, output_dir)


def main(argv):
  # Read input flags
  dataset_name = opts.dataset_name
  dataset_parent_dir = opts.dataset_parent_dir
  output_dir = FLAGS.output_dir
  xception_frozen_graph_path = FLAGS.xception_frozen_graph_path
  splits = ['train', 'val']
  # Run the preprocessing pipeline
  export_aligned_dataset_to_tfrecord(
    dataset_parent_dir, output_dir, dataset_name, splits,
    xception_frozen_graph_path)


if __name__ == '__main__':
  app.run(main)
