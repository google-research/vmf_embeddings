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

"""Compute performance metrics given vMF embeddings.

First argument is the path to the directory containing some number of .npz
embedding files.
The code will recurse to find them all.
"""

import os
import random
import sys

import faiss
import numpy as np
from scipy.special import softmax
from scipy.stats import mode
import torch

from vmf_embeddings import eval_utils
from vmf_embeddings import utils
from vmf_embeddings.third_party.s_vae_pytorch import distributions


def softmax_accuracy(logits, ids):
  """Computes accuracy of class logits and ids marginalizing over samples."""
  softmax_probs = np.mean(softmax(logits, axis=2), axis=1)
  correct = np.argmax(softmax_probs, axis=1) == ids
  acc = np.sum(correct) / float(len(ids))
  return (acc, softmax_probs)


def recall_at_1(embs, embs_norms, ids, n_samples=10):
  """Computes recall@1 for embeddings and ground-truth ids maringalizing samples.

  Args:
    embs: An ndarray of embeddings.
    embs_norms: ndarray of norms of the embeddings.
    ids: An ndarray of ground-truth class ids for each embedding.
    n_samples: Number of samples for marginalization.

  Returns:
    recall@1 metric value.
  """
  with torch.no_grad():
    z_dist = distributions.VonMisesFisher(
        torch.from_numpy(embs), torch.from_numpy(embs_norms))
    z_samples = z_dist.sample(torch.Size([n_samples])).permute(1, 0, 2).numpy()

  res = faiss.StandardGpuResources()
  corrects = []
  for i in range(n_samples):
    z = z_samples[:, i, :]
    index = faiss.GpuIndexFlatIP(res, z.shape[1])
    index.add(z)
    _, idxs = index.search(z, 2)
    preds = ids[idxs[:, 1]]
    correct = ids == preds
    corrects.append(correct)

  corrects = np.array(corrects)
  correct_mode = mode(corrects, axis=0)[0]
  return np.mean(correct_mode)


def map_at_r(embs, embs_norms, ids, n_samples=10):
  """Computes mAP@R for embeddings and ground-truth ids maringalizing samples.

  mAP@r code adapted from
    https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/utils/accuracy_calculator.py

  Args:
    embs: An ndarray of embeddings.
    embs_norms: ndarray of norms of the embeddings.
    ids: An ndarray of ground-truth class ids for each embedding.
    n_samples: Number of samples for marginalization.

  Returns:
    mAP@R metric value.
  """
  with torch.no_grad():
    z_dist = distributions.VonMisesFisher(
        torch.from_numpy(embs), torch.from_numpy(embs_norms))
    z_samples = z_dist.sample(torch.Size([n_samples])).permute(1, 0, 2).numpy()

  _, counts = np.unique(ids, return_counts=True)
  r_mask = np.zeros((embs.shape[0], np.max(counts) - 1), dtype=np.bool)
  for i, count in enumerate(counts):
    r_mask[np.where(ids == i), :count - 1] = True

  res = faiss.StandardGpuResources()
  maps = []
  for i in range(n_samples):
    z = z_samples[:, i, :]
    index = faiss.GpuIndexFlatIP(res, z.shape[1])
    index.add(z)
    try:
      # If search uses too much memory on GPU, switch to CPU
      _, all_idxs = index.search(z, int(np.max(counts)))
    except:
      index = faiss.index_gpu_to_cpu(index)
      _, all_idxs = index.search(z, int(np.max(counts)))
    all_idxs = all_idxs[:, 1:]

    ids_matrix = ids[all_idxs]
    correct = (ids_matrix == ids[:, np.newaxis]) * r_mask
    cumulative_correct = np.cumsum(correct, axis=1)
    k_idx = np.tile(np.arange(1, r_mask.shape[1] + 1), (r_mask.shape[0], 1))
    precision_at_ks = (cumulative_correct * correct) / k_idx
    summed_precision_per_row = np.sum(precision_at_ks * r_mask, axis=1)
    max_possible_matches_per_row = np.sum(r_mask, axis=1)
    aps = summed_precision_per_row / max_possible_matches_per_row
    maps.append(np.mean(aps))

  return np.mean(maps)


def main():
  path = sys.argv[1]
  n_samples = 10
  n_bins = 15
  torch.manual_seed(1234)
  random.seed(1234)
  np.random.seed(1234)

  norm_method = utils.get_norm_method_by_name("l2")
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

      data_files = [
          os.path.join(root, f)
          for f in os.listdir(root)
          if f == dataset + "_test.npz"
      ]

      for df in data_files:
        print("Split: {}".format(df.split("/")[-1]))
        data = np.load(df)
        ids = data["ids"]

        if split_by_dataset[dataset] == "softmax":
          logits = data["logits"]
          softmax_acc, softmax_probs = softmax_accuracy(logits, ids)
          ece = eval_utils.calc_ece_em_quant(
              np.max(softmax_probs, axis=1),
              np.argmax(softmax_probs, axis=1) == ids,
              n_bins,
              lambda_exp=2,
          )
          results["softmax"]["acc"].append(softmax_acc)
          results["softmax"]["ece"].append(ece)
        else:
          embs = data["embeddings"]
          embs, embs_norms = norm_method(
              embs, use_torch=False, return_norms=True)

          r1_acc = recall_at_1(embs, embs_norms, ids, n_samples=n_samples)
          results[split]["r@1"].append(r1_acc)

          map_at_r_val = map_at_r(embs, embs_norms, ids, n_samples=n_samples)
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
