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

"""Dataset class for CUB200-2011."""

import os

import numpy as np
from PIL import Image
from torchvision import transforms

from vmf_embeddings.datasets import dataset


class CUB200Dataset(dataset.Dataset):
  """Dataset class for CUB200-2011."""

  def __init__(self, dataset_path):
    super(CUB200Dataset, self).__init__(dataset_path)
    self.crop_size = 224
    self.scale = (0.16, 1.0)
    self.resize = 256
    self.ratio = (3.0 / 4.0, 4.0 / 3.0)
    self.color_jitter = (0.25, 0.25, 0.25, 0.0)
    self.mean = (0.485, 0.456, 0.406)
    self.std = (0.229, 0.224, 0.225)

    self.train_x, self.train_y, self.train_y_original = self._read_file(
        os.path.join(self.dataset_path, "train.txt"))
    self.valid_x, self.valid_y, self.valid_y_original = self._read_file(
        os.path.join(self.dataset_path, "val.txt"))
    self.test_x, self.test_y, self.test_y_original = self._read_file(
        os.path.join(self.dataset_path, "test.txt"))

    self.switch_split("train")

    self.train_transforms = transforms.Compose([
        transforms.Resize(self.resize),
        transforms.ColorJitter(*self.color_jitter),
        transforms.RandomResizedCrop(
            size=self.crop_size, scale=self.scale, ratio=self.ratio),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(self.mean, self.std),
    ])
    self.test_transforms = transforms.Compose([
        transforms.Resize(self.resize),
        transforms.CenterCrop(size=self.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(self.mean, self.std),
    ])

  def _read_file(self, filename):
    """Reads the file containing image paths and labels for each instance."""
    with open(filename, "r") as f:
      lines = f.read().splitlines()
    x = [os.path.join(self.dataset_path, line.split(",")[0]) for line in lines]
    y = [int(line.split(",")[1]) for line in lines]
    return x, np.unique(y, return_inverse=True)[1], y

  def __getitem__(self, idx):
    image_path = self.images[idx]
    label = self.labels[idx]
    original_label = self.original_labels[idx]

    image = Image.open(image_path).convert("RGB")

    if self.split == "train":
      image = self.train_transforms(image)
    else:
      image = self.test_transforms(image)

    return {
        "ids": label,
        "examples": image,
        "original_ids": original_label,
    }
