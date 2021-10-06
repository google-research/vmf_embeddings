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

"""Dataset class for CIFAR100."""

import copy

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torchvision.datasets as datasets

from vmf_embeddings.datasets import dataset
from vmf_embeddings.datasets import utils


class Cifar100Dataset(dataset.Dataset):
  """Dataset class for CIFAR100."""

  def __init__(self, dataset_path):
    super(Cifar100Dataset, self).__init__(dataset_path)
    self.val_proportion = 0.15
    self.mean = (0.5071, 0.4865, 0.4409)
    self.std = (0.2673, 0.2564, 0.2762)

    train = datasets.CIFAR100(
        root=dataset_path, train=True, download=True, transform=None)
    self.train_x = copy.deepcopy(train.data)
    self.train_y = copy.deepcopy(np.array(train.targets))

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

    test = datasets.CIFAR100(
        root=dataset_path, train=False, download=True, transform=None)
    self.test_x = copy.deepcopy(test.data)
    self.test_y = copy.deepcopy(np.array(test.targets))

    self.switch_split("train")

    self.train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(self.mean, self.std),
        utils.RandomOcclusion(img_size=32, occ_size=8),
    ])
    self.test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(self.mean, self.std),
    ])

  def __getitem__(self, idx):
    img = self.images[idx]
    img = Image.fromarray(img)

    if self.split == "train":
      img = self.train_transforms(img)
    else:
      img = self.test_transforms(img)

    return {
        "ids": self.labels[idx],
        "examples": img,
    }
