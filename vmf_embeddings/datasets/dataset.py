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

"""Abstract dataset class for training embedding networks."""

import abc


class Dataset(abc.ABC):
  """Abstract dataset class for training embedding networks."""

  def __init__(self, dataset_path):
    """Initializes an abstract Dataset class.

    Args:
      dataset_path: Path to the directory containing the data.
    """
    self.split = None
    self.images = None
    self.labels = None
    self.original_labels = None
    self.dataset_path = dataset_path
    self.seed = 1234

  def switch_split(self, split):
    """Switch the split of the dataset between train, validation, and test."""
    self.split = split
    if split not in ["train", "valid", "test"]:
      raise ValueError("Unknown dataset split {}".format(split))

    image_attr = "{}_x".format(split)
    label_attr = "{}_y".format(split)
    assert hasattr(self, image_attr) and hasattr(self, label_attr)

    self.images = getattr(self, image_attr)
    self.labels = getattr(self, label_attr)

    # Original labels may not be zero-indexed
    org_label_attr = "{}_y_original".format(split)
    if hasattr(self, org_label_attr):
      self.original_labels = getattr(self, org_label_attr)

  def get_test_splits(self):
    return ["test"]

  def __len__(self):
    assert self.labels is not None
    return len(self.labels)

  @abc.abstractmethod
  def __getitem__(self, idx):
    raise NotImplementedError("Need to implement this in subclasses of Dataset")
