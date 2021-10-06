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

"""Compute performance metrics given deterministic/point embeddings.

First argument is the path to the directory containing some number of .npz
embedding files.
The code will recurse to find them all.

Second argument is the normalization method to use
(one of "l2", "hyperbolic", "none").
"""

import os
import random
import sys

import faiss
import numpy as np
from scipy.special import softmax
from sklearn.neighbors import KNeighborsClassifier

from vmf_embeddings import eval_utils
from vmf_embeddings import utils
from vmf_embeddings.third_party.hyperbolic_image_embeddings import hypnn


def softmax_accuracy(logits, ids):
  """Computes accuracy given class logits and ground-truth ids."""
  correct = np.argmax(logits, axis=1) == ids
  acc = np.sum(correct) / float(len(ids))
  return acc, correct, softmax(logits, axis=1)


def recall_at_1(embs, ids, index=None, dist_matrix=None):
  """Computes recall@1 given embeddings and ground-truth ids.

  Args:
    embs: An ndarray of embeddings.
    ids: An ndarray of ground-truth class ids for each embedding.
    index: A FAISS index to use for similarity search.
    dist_matrix: A pairwise distance matrix to use if index is None.

  Returns:
    recall@1 metric value.
  """
  assert index is not None or dist_matrix is not None

  if index is not None:
    _, idxs = index.search(embs, 2)
    preds = ids[idxs[:, 1]]
  else:
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1, metric="precomputed")
    knn.fit(dist_matrix, ids)
    idxs = knn.kneighbors(return_distance=False)
    preds = np.squeeze(ids[idxs], axis=1)

  correct = ids == preds
  acc = np.sum(correct) / float(len(ids))
  return acc


def map_at_r(embs, ids, index=None, dist_matrix=None):
  """Computes mAP@R (average precision) given embeddings and ground-truth ids.

  mAP@r code adapted from
    https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/utils/accuracy_calculator.py

  Args:
    embs: An ndarray of embeddings.
    ids: An ndarray of ground-truth class ids for each embedding.
    index: A FAISS index to use for similarity search.
    dist_matrix: A pairwise distance matrix to use if index is None.

  Returns:
    mAP@R metric value.
  """
  assert index is not None or dist_matrix is not None

  _, counts = np.unique(ids, return_counts=True)
  r_mask = np.zeros((ids.shape[0], np.max(counts) - 1), dtype=np.bool)
  for i, count in enumerate(counts):
    r_mask[np.where(ids == i), :count - 1] = True

  if index is not None:
    try:
      # If search uses too much memory on GPU, switch to CPU
      _, all_idxs = index.search(embs, int(np.max(counts)))
    except:
      index = faiss.index_gpu_to_cpu(index)
      _, all_idxs = index.search(embs, int(np.max(counts)))
    all_idxs = all_idxs[:, 1:]
  else:
    knn = KNeighborsClassifier(
        n_neighbors=np.max(counts) - 1,
        n_jobs=-1,
        metric="precomputed",
    )
    knn.fit(dist_matrix, ids)
    all_idxs = knn.kneighbors(return_distance=False)

  ids_matrix = ids[all_idxs]
  correct = (ids_matrix == ids[:, np.newaxis]) * r_mask
  cumulative_correct = np.cumsum(correct, axis=1)
  k_idx = np.tile(np.arange(1, r_mask.shape[1] + 1), (r_mask.shape[0], 1))
  precision_at_ks = (cumulative_correct * correct) / k_idx
  summed_precision_per_row = np.sum(precision_at_ks * r_mask, axis=1)
  max_possible_matches_per_row = np.sum(r_mask, axis=1)
  aps = summed_precision_per_row / max_possible_matches_per_row

  return np.mean(aps)


def main():
  path = sys.argv[1]
  normalize = sys.argv[2]
  random.seed(1234)
  np.random.seed(1234)
  n_bins = 15
  assert normalize in [
      "l2",
      "hyperbolic",
      "none",
  ]
  norm_method = utils.get_norm_method_by_name(normalize)

  split_by_dataset = {
      "mnist": "softmax",
      "fashionmnist": "softmax",
      "cifar10": "softmax",
      "cifar100": "softmax",
      "cars196": "retrieval",
      "stanfordonlineproducts": "retrieval",
      "synthetic": "softmax",
      "cub200": "retrieval"
  }

  if "fashion_mnist" in path:
    dataset = "fashionmnist"
  elif "mnist" in path:
    dataset = "mnist"
  elif "cifar100" in path:
    dataset = "cifar100"
  elif "cifar10" in path:
    dataset = "cifar10"
  elif "cars196" in path:
    dataset = "cars196"
  elif "synthetic" in path:
    dataset = "synthetic"
  elif "cub200" in path:
    dataset = "cub200"
  else:
    dataset = "stanfordonlineproducts"

  results = {}
  split = split_by_dataset[dataset]
  # Softmax computes different metrics compared to retrieval.
  if split == "softmax":
    results[split] = {"acc": [], "ece": []}
  else:
    results[split] = {
        "map@r": [],
        "r@1": [],
    }

  for root, _, files in os.walk(path):
    for f in files:
      if not f.endswith(".npz"):
        continue

      print("\nPath {}".format(root))

      # Extract .npz files for the test dataset
      data_files = [
          os.path.join(root, f)
          for f in os.listdir(root)
          if f == dataset + "_test.npz"
      ]

      kwargs = {}
      # Find the c parameter if the method is hyperbolic
      if normalize == "hyperbolic":
        with open(os.path.join(root, "main.train.log"), "r") as f:
          for line in f:
            line = line.strip()
            if line.startswith("c:"):
              kwargs["c"] = float(line.split(":")[1])
              print("Found c = {}".format(kwargs["c"]))
              break

      for df in data_files:
        print("Split: {}".format(df.split("/")[-1]))
        data = np.load(df)
        ids = data["ids"]

        if split_by_dataset[dataset] == "softmax":
          logits = data["logits"]
          softmax_acc, softmax_correct, softmax_probs = softmax_accuracy(
              logits, ids)
          softmax_max_prob = np.max(softmax_probs, axis=1)
          ece = eval_utils.calc_ece_em_quant(
              softmax_max_prob,
              softmax_correct,
              n_bins,
              lambda_exp=2,
          )
          results["softmax"]["acc"].append(softmax_acc)
          results["softmax"]["ece"].append(ece)
        else:
          embs = data["embeddings"]
          embs, norms = norm_method(
              embs, use_torch=False, return_norms=True, **kwargs)
          norms = norms.squeeze(1)

          dist_matrix = None
          gpu_index = None
          # For hyperbolic, cannot use FAISS
          if normalize == "hyperbolic":
            batch_size = 512
            dist_matrix = hypnn.pairwise_dist_matrix(
                embs, kwargs["c"], batch_size=batch_size)
          else:
            res = faiss.StandardGpuResources()
            if normalize == "none":
              index = faiss.IndexFlatL2(embs.shape[1])
            else:
              index = faiss.IndexFlatIP(embs.shape[1])
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index.add(embs)

          r1_acc = recall_at_1(embs, ids, gpu_index, dist_matrix)
          results[split]["r@1"].append(r1_acc)

          map_at_r_val = map_at_r(embs, ids, gpu_index, dist_matrix)
          results[split]["map@r"].append(map_at_r_val)

      break

  for k, v in results.items():
    print("\n=== {} ===".format(k))
    if k != "softmax":
      print("Mean Recall@1: acc = {:.4f} +/- {:.4f}".format(
          100.0 * np.mean(v["r@1"]),
          100.0 * np.std(v["r@1"]) / np.sqrt(len(v["r@1"])),
      ))
      print("Mean mAP@R: val = {:.4f} +/- {:.4f}".format(
          np.mean(v["map@r"]),
          np.std(v["map@r"]) / np.sqrt(len(v["map@r"])),
      ))
    else:
      print(
          ("Mean {}: acc = {:.4f} +/- {:.4f}, ece = {:.4f} +/- {:.4f}").format(
              k,
              100.0 * np.mean(v["acc"]),
              100.0 * np.std(v["acc"]) / np.sqrt(len(v["acc"])),
              np.mean(v["ece"]),
              np.std(v["ece"]) / np.sqrt(len(v["ece"])),
          ))


if __name__ == "__main__":
  main()
