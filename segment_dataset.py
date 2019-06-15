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

"""Generate semantic segmentations
This module uses Xception model trained on ADE20K dataset to generate semantic
segmentation mask to any set of images.
"""

from absl import app
from absl import flags
from PIL import Image
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import shutil
import tensorflow as tf
import utils


def get_semantic_color_coding():
  """
  assigns the 30 (actually 29) semantic colors from cityscapes semantic mapping
  to selected classes from the ADE20K150 semantic classes.
  """
  # Below are the 30 cityscape colors (one is duplicate. so total is 29 not 30)
  colors = [
    (111, 74,  0),
    ( 81,  0, 81),
    (128, 64,128),
    (244, 35,232),
    (250,170,160),
    (230,150,140),
    ( 70, 70, 70),
    (102,102,156),
    (190,153,153),
    (180,165,180),
    (150,100,100),
    (150,120, 90),
    (153,153,153),
    # (153,153,153),
    (250,170, 30),
    (220,220,  0),
    (107,142, 35),
    (152,251,152),
    ( 70,130,180),
    (220, 20, 60),
    (255,  0,  0),
    (  0,  0,142),
    (  0,  0, 70),
    (  0, 60,100),
    (  0,  0, 90),
    (  0,  0,110),
    (  0, 80,100),
    (  0,  0,230),
    (119, 11, 32),
    (  0,  0,142)]
  k_num_ade20k_classes = 150
  # initially all 150 classes are mapped to a single color (last color idx: -1)
  # Some classes are to be assigned independent colors
  # semantic classes are 1-based (1 thru 150)
  semantic_to_color_idx = -1 * np.ones(k_num_ade20k_classes + 1, dtype=int)
  semantic_to_color_idx [1] = 0    # wall
  semantic_to_color_idx [2] = 1    # building;edifice
  semantic_to_color_idx [3] = 2    # sky
  semantic_to_color_idx [105] = 3  # fountain
  semantic_to_color_idx [27] = 4   # sea
  semantic_to_color_idx [60] = 5   # stairway;staircase 
  semantic_to_color_idx [5] = 6    # tree
  semantic_to_color_idx [12] = 7   # sidewalk;pavement 
  semantic_to_color_idx [4]  = 7   # floor;flooring
  semantic_to_color_idx [7]  = 7   # road;route
  semantic_to_color_idx [13] = 8   # people
  semantic_to_color_idx [18] = 9   # plant;flora;plant;life
  semantic_to_color_idx [17] = 10  # mountain;mount
  semantic_to_color_idx [20] = 11  # chair
  semantic_to_color_idx [6] = 12   # ceiling
  semantic_to_color_idx [22] = 13  # water
  semantic_to_color_idx [35] = 14  # rock;stone
  semantic_to_color_idx [14] = 15  # earth;ground
  semantic_to_color_idx [10] = 16  # grass
  semantic_to_color_idx [70] = 17  # bench
  semantic_to_color_idx [54] = 18  # stairs;steps
  semantic_to_color_idx [101] = 19 # poster
  semantic_to_color_idx [77] = 20  # boat
  semantic_to_color_idx [85] = 21  # tower
  semantic_to_color_idx [23] = 22  # painting;picture
  semantic_to_color_idx [88] = 23  # streetlight;stree;lamp
  semantic_to_color_idx [43] = 24  # column;pillar
  semantic_to_color_idx [9] = 25   # window;windowpane
  semantic_to_color_idx [15] = 26  # door;
  semantic_to_color_idx [133] = 27 # sculpture

  semantic_to_rgb = np.array(
    [colors[col_idx][:] for col_idx in semantic_to_color_idx])
  return semantic_to_rgb


def _apply_colors(seg_images_path, save_dir, idx_to_color):
  for i, img_path in enumerate(seg_images_path):
    print('processing img #%05d / %05d: %s' % (i, len(seg_images_path),
                                               osp.split(img_path)[1]))
    seg = np.array(Image.open(img_path))
    seg_rgb = np.zeros(seg.shape + (3,), dtype=np.uint8)
    for col_idx in range(len(idx_to_color)):
      if idx_to_color[col_idx][0] != -1:
        mask = seg == col_idx
        seg_rgb[mask, :] = idx_to_color[col_idx][:]

    parent_dir, filename = osp.split(img_path)
    basename, ext = osp.splitext(filename)
    out_filename = basename + "_rgb.png"
    out_filepath = osp.join(save_dir, out_filename)
    # Save rescaled segmentation image
    Image.fromarray(seg_rgb).save(out_filepath)


# The frozen xception model only segments 512x512 images. But it would be better
# to segment the full image instead!
def segment_images(images_path, xception_frozen_graph_path, save_dir,
                   crop_height=512, crop_width=512):
  if not osp.exists(xception_frozen_graph_path):
    raise OSError('Xception frozen graph not found at %s' %
                            xception_frozen_graph_path)
  with tf.gfile.GFile(xception_frozen_graph_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  with tf.Graph().as_default() as graph:
    new_input = tf.placeholder(tf.uint8, [1, crop_height, crop_width, 3],
                               name="new_input")
    tf.import_graph_def(
      graph_def,
      input_map={"ImageTensor:0": new_input},
      return_elements=None,
      name="sem_seg",
      op_dict=None,
      producer_op_list=None
    )

  corrupted_dir = osp.join(save_dir, 'corrupted')
  if not osp.exists(corrupted_dir):
    os.makedirs(corrupted_dir)
  with tf.Session(graph=graph) as sess:
    for i, img_path in enumerate(images_path):
      print('Segmenting image %05d / %05d: %s' % (i + 1, len(images_path),
                                                  img_path))
      img = np.array(Image.open(img_path))
      if len(img.shape) == 2 or img.shape[2] != 3:
        print('Warning! corrupted image %s' % img_path)
        img_base_path = img_path[:-14]  # remove the '_reference.png' suffix
        srcs = sorted(glob.glob(img_base_path + '_*'))
        dest = unicode(corrupted_dir + '/.')
        for src in srcs:
          shutil.move(src, dest)
        continue
      img = utils.get_central_crop(img, crop_height=crop_height,
                             crop_width=crop_width)
      img = np.expand_dims(img, 0)  # convert to NHWC format
      seg = sess.run("sem_seg/SemanticPredictions:0", feed_dict={
          new_input: img})
      assert np.max(seg[:]) <= 255, 'segmentation image is not of type uint8!'
      seg = np.squeeze(np.uint8(seg))  # convert to uint8 and squeeze to WxH.
      parent_dir, filename = osp.split(img_path)
      basename, ext = osp.splitext(filename)
      basename = basename[:-10]  # remove the '_reference' suffix
      seg_filename = basename + "_seg.png"
      seg_filepath = osp.join(save_dir, seg_filename)
      # Save segmentation image
      Image.fromarray(seg).save(seg_filepath)

def segment_and_color_dataset(dataset_dir, xception_frozen_graph_path,
                              splits=None, resegment_images=True):
  if splits is None:
    imgs_dirs = [dataset_dir]
  else:
    imgs_dirs = [osp.join(dataset_dir, split) for split in splits]
  
  for cur_dir in imgs_dirs:
    imgs_file_pattern = osp.join(cur_dir, '*_reference.png')
    images_path = sorted(glob.glob(imgs_file_pattern))
    if resegment_images:
      segment_images(images_path, xception_frozen_graph_path, cur_dir,
                     crop_height=512, crop_width=512)

  idx_to_col = get_semantic_color_coding()

  for cur_dir in imgs_dirs:
    save_dir = cur_dir
    seg_file_pattern = osp.join(cur_dir, '*_seg.png')
    seg_imgs_paths = sorted(glob.glob(seg_file_pattern))
    _apply_colors(seg_imgs_paths, save_dir, idx_to_col)
