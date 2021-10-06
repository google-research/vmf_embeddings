# coding=utf-8
# Copyright 2021 The vMF Embeddings Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions regarding the dataset, augmentation, and sampling."""

import numpy as np
import torch


class EpisodicBatchSampler(torch.utils.data.sampler.Sampler):
  """Samples indices corresponding to k instances from n classes."""

  def __init__(self, labels, num_classes_per_episode, num_samples_per_class,
               num_episodes):
    """Initializes an EpisodicBatchSampler object.

    Args:
      labels: ndarray containing the labels for each input instance.
      num_classes_per_episode: Number of classes to sample in an episode (n).
      num_samples_per_class: Number of input samples drawn per class (k).
      num_episodes: Number of episodes in an "epoch".
    """
    super(EpisodicBatchSampler, self).__init__(None)
    self.labels = labels
    self.unique_labels = np.unique(labels)
    self.total_classes = len(self.unique_labels)
    self.num_episodes = num_episodes
    self.num_classes_per_episode = num_classes_per_episode
    self.num_samples_per_class = num_samples_per_class

  def __len__(self):
    return self.num_episodes

  def __iter__(self):
    for _ in range(self.num_episodes):
      selected_idxs = []
      classes = np.random.choice(
          self.unique_labels,
          size=min(self.num_classes_per_episode, self.total_classes),
          replace=False,
      )
      for cl in classes:
        all_cl_idxs = np.where(self.labels == cl)[0]
        idxs = np.random.choice(
            all_cl_idxs,
            size=min(len(all_cl_idxs), self.num_samples_per_class),
            replace=False,
        )
        selected_idxs.append(idxs)
      yield np.concatenate(selected_idxs, axis=0)


class RandomOcclusion:
  """Creates a random patch of occluding pixels in an image with value 0."""

  def __init__(self, img_size, occ_size=8):
    """Initializes a RandomOcclusion object.

    Args:
      img_size: Size of the height/width of the image (assumes a square image)
        in pixels.
      occ_size: Size of the occlusion (assumes a square occlusion) in pixels.
    """
    self.img_size = img_size
    self.occ_size = occ_size
    self.upper_bound = self.img_size - self.occ_size + 1

  def __call__(self, x):
    x1 = np.random.randint(0, self.upper_bound)
    y1 = np.random.randint(0, self.upper_bound)
    x2 = x1 + self.occ_size
    y2 = y1 + self.occ_size
    x[:, y1:y2, x1:x2] = 0.0
    return x
