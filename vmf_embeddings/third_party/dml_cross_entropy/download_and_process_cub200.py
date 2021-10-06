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

"""Downloads and processes the CUB200-2011 dataset and saves to a directory.

Code adapted from:
  https://github.com/jeromerony/dml_cross_entropy/blob/master/prepare_data.py.

Only argument is the path to download and save the CUB200-2011 dataset.
"""

import os
import sys
import tarfile

import gdown
import numpy as np


def download_extract_cub(cub_dir, cub_url):
  """Downloads the CUB200-2011 dataset and extracts the tar."""
  fn = "CUB200-2011.tgz"
  gdown.download(cub_url, os.path.join(cub_dir, fn))
  filename = os.path.join(cub_dir, fn)
  with tarfile.open(filename, "r:gz") as tar:
    tar.extractall(path=cub_dir)


def generate_cub_train_val_test(cub_dir, image_file, train_file, val_file,
                                test_file):
  """Processes the data and generates a train/validation/test split."""
  n_classes_in_train = 100
  n_val_classes = 15
  image_file = os.path.join(cub_dir, image_file)
  train_file = os.path.join(cub_dir, train_file)
  val_file = os.path.join(cub_dir, val_file)
  test_file = os.path.join(cub_dir, test_file)

  train = []
  val = []
  test = []

  # Choose n_val_classes of the training classes to be in the validation set
  val_classes = np.random.choice(
      np.arange(n_classes_in_train), size=n_val_classes, replace=False)

  with open(image_file) as f_images:
    lines_images = f_images.read().splitlines()

  for line in lines_images:
    image_path = line.split()[1]
    label = int(image_path.split(".")[0]) - 1
    file_line = ",".join((os.path.join("images", image_path), str(label)))
    if label in val_classes:
      val.append(file_line)
    elif label < n_classes_in_train:
      train.append(file_line)
    else:
      test.append(file_line)

  with open(train_file, "w") as f:
    f.write("\n".join(train))
  with open(val_file, "w") as f:
    f.write("\n".join(val))
  with open(test_file, "w") as f:
    f.write("\n".join(test))


def main():
  np.random.seed(1234)
  cub_dir = sys.argv[1]
  cub_url = "https://drive.google.com/uc?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45"
  image_file = "images.txt"
  train_file = "train.txt"
  val_file = "val.txt"
  test_file = "test.txt"
  tar_root = "CUB_200_2011"

  download_extract_cub(cub_dir, cub_url)
  generate_cub_train_val_test(
      os.path.join(cub_dir, tar_root), image_file, train_file, val_file,
      test_file)


if __name__ == "__main__":
  main()
