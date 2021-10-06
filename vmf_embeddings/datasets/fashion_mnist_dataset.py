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

"""Dataset class for FashionMNIST."""

import copy

import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.datasets as datasets

from vmf_embeddings.datasets import dataset


class FashionMNISTDataset(dataset.Dataset):
  """Dataset class for FashionMNIST."""

  def __init__(self, dataset_path):
    super(FashionMNISTDataset, self).__init__(dataset_path)
    self.val_proportion = 0.15
    self.normalize_val = 255.0

    train = datasets.FashionMNIST(
        root=dataset_path, train=True, download=True, transform=None)
    self.train_x = np.expand_dims(
        copy.deepcopy(train.train_data.numpy()).astype(np.float32) /
        self.normalize_val, 1)
    self.train_y = copy.deepcopy(train.train_labels.numpy())

    # Create a validation split containing self.val_proportion of the data
    (
        self.train_x,
        self.valid_x,
        self.train_y,
        self.valid_y,
    ) = train_test_split(
        self.train_x,
        self.train_y,
        test_size=self.val_proportion,
        random_state=self.seed,
        stratify=self.train_y,
    )

    test = datasets.FashionMNIST(
        root=dataset_path, train=False, download=True, transform=None)
    self.test_x = np.expand_dims(
        copy.deepcopy(test.test_data.numpy()).astype(np.float32) /
        self.normalize_val, 1)
    self.test_y = copy.deepcopy(test.test_labels.numpy())

    self.switch_split("train")

  def __getitem__(self, idx):
    return {
        "ids": self.labels[idx],
        "examples": self.images[idx],
    }
