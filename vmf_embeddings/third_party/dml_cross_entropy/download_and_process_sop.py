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

"""Downloads and processes the SOP dataset and saves to a directory.

Code adapted from:
  https://github.com/jeromerony/dml_cross_entropy/blob/master/prepare_data.py.

Only argument is the path to download and save the SOP dataset.
"""

import os
import sys
import zipfile

import numpy as np
from torchvision.datasets.utils import download_url


def download_extract_sop(sop_dir, sop_url):
  """Downloads the SOP dataset and extracts the tar."""
  download_url(sop_url, root=sop_dir)
  filename = os.path.join(sop_dir, os.path.basename(sop_url))
  with zipfile.ZipFile(filename) as zipf:
    zipf.extractall(path=sop_dir)


def generate_sop_train_val_test(sop_dir):
  """Processes the data and generates a train/validation/test split."""
  n_train_classes = 11318
  n_val_classes = 1698
  original_train_file = os.path.join(sop_dir, "Stanford_Online_Products",
                                     "Ebay_train.txt")
  original_test_file = os.path.join(sop_dir, "Stanford_Online_Products",
                                    "Ebay_test.txt")

  with open(original_train_file) as f_images:
    train_lines = f_images.read().splitlines()[1:]
  with open(original_test_file) as f_images:
    test_lines = f_images.read().splitlines()[1:]

  # Choose n_val_classes of the training classes to be in the validation set
  val_classes = np.random.choice(
      np.arange(n_train_classes), size=n_val_classes, replace=False)

  train = []
  val = []
  for l in train_lines:
    _, class_idx, _, path = l.split()
    class_idx = int(class_idx)
    p = ",".join((path, str(class_idx - 1)))
    if class_idx - 1 in val_classes:
      val.append(p)
    else:
      train.append(p)

  test = [
      ",".join((l.split()[-1], str(int(l.split()[1]) - 1))) for l in test_lines
  ]

  with open(os.path.join(sop_dir, "train.txt"), "w") as f:
    f.write("\n".join(train))
  with open(os.path.join(sop_dir, "val.txt"), "w") as f:
    f.write("\n".join(val))
  with open(os.path.join(sop_dir, "test.txt"), "w") as f:
    f.write("\n".join(test))


def main():
  np.random.seed(1234)
  sop_dir = sys.argv[1]
  sop_url = "ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip"

  download_extract_sop(sop_dir, sop_url)
  generate_sop_train_val_test(sop_dir)


if __name__ == "__main__":
  main()
