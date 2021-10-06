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

"""Downloads and processes the Cars196 dataset and saves to a directory.

Code adapted from:
  https://github.com/jeromerony/dml_cross_entropy/blob/master/prepare_data.py.

Only argument is the path to download and save the Cars196 dataset.
"""

import os
import sys
import tarfile

import numpy as np
from scipy.io import loadmat
from torchvision.datasets.utils import download_url


def download_extract_cars(cars_dir, cars_url, cars_annotations_url):
  """Downloads the Cars196 dataset and extracts the tar."""
  download_url(cars_annotations_url, root=cars_dir)
  download_url(cars_url, root=cars_dir)
  filename = os.path.join(cars_dir, os.path.basename(cars_url))
  with tarfile.open(filename, "r:gz") as tar:
    tar.extractall(path=cars_dir)
  return os.path.join(cars_dir, os.path.basename(cars_annotations_url))


def generate_cars_train_val_test(cars_dir, annotation_file):
  """Processes the data and generates a train/validation/test split."""
  n_classes_in_train = 98
  n_val_classes = 15
  train = []
  val = []
  test = []

  # Choose n_val_classes of the training classes to be in the validation set
  val_classes = np.random.choice(
      np.arange(n_classes_in_train), size=n_val_classes, replace=False)

  annotations = loadmat(annotation_file)
  label_dict = {
      anno[0][0]: anno[5][0][0] - 1 for anno in annotations["annotations"][0]
  }

  for image_path, label in label_dict.items():
    file_line = ",".join((image_path, str(label)))
    if label in val_classes:
      val.append(file_line)
    elif label < n_classes_in_train:
      train.append(file_line)
    else:
      test.append(file_line)

  with open(os.path.join(cars_dir, "train.txt"), "w") as f:
    f.write("\n".join(train))
  with open(os.path.join(cars_dir, "val.txt"), "w") as f:
    f.write("\n".join(val))
  with open(os.path.join(cars_dir, "test.txt"), "w") as f:
    f.write("\n".join(test))


def main():
  np.random.seed(1234)
  cars_dir = sys.argv[1]
  cars_url = "http://imagenet.stanford.edu/internal/car196/car_ims.tgz"
  cars_annotations_url = "http://imagenet.stanford.edu/internal/car196/cars_annos.mat"

  annotation_file = download_extract_cars(cars_dir, cars_url,
                                          cars_annotations_url)
  generate_cars_train_val_test(cars_dir, annotation_file)


if __name__ == "__main__":
  main()
