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

"""Functions for performing fixed-set and open-set validation of a model."""

import faiss
import numpy as np
from scipy.stats import mode
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn.functional as F

from vmf_embeddings import utils
from vmf_embeddings.third_party.hyperbolic_image_embeddings import hypnn
from vmf_embeddings.third_party.s_vae_pytorch import distributions


def fixed_set_val_loop(arch, ds, method, cfg):
  """Performs fixed-set validation by computing accuracy.

  Args:
    arch: An archs.arch.Arch object (e.g., ResNet50 or N-digit MNIST arch).
    ds: A datasets object (e.g., MNIST, CIFAR10).
    method: A methods.methods softmax cross-entropy object (e.g., ArcFace).
    cfg: The configuration dictionary from Hydra.

  Returns:
    Validation accuracy.
  """
  method_name = method.__class__.__name__
  arch.train(False)
  ds.switch_split("valid")
  loader = utils.get_data_loader("sequential", ds, cfg.num_workers,
                                 {"batch_size": 128})
  n_correct = 0
  with torch.no_grad():
    for batch in loader:
      examples = batch["examples"]
      ids = batch["ids"]
      if torch.cuda.is_available():
        examples = examples.cuda()
        ids = ids.cuda()
      # vMF has samples, so computing validation accuracy differs
      if method_name == "VMFSoftmax":
        preds, _ = method(
            examples,
            ids,
            get_predictions=True,
            n_samples=method.n_samples,
        )
        preds = torch.mean(F.softmax(preds, dim=2), dim=1)
        preds = torch.argmax(preds, dim=1)
        n_correct += torch.sum((preds == ids).int()).item()
      # Compute n_correct for deterministic methods
      else:
        preds, _ = method(examples, ids, get_predictions=True)
        n_correct += torch.sum((torch.argmax(preds, dim=1) == ids).int()).item()
  return n_correct / float(len(ds))


def open_set_val_loop(arch, ds, method, cfg):
  """Performs open-set validation by computing recall@1.

  Args:
    arch: An archs.arch.Arch object (e.g., ResNet50 or N-digit MNIST arch).
    ds: A datasets object (e.g., MNIST, CIFAR10).
    method: A methods.methods softmax cross-entropy object (e.g., ArcFace).
    cfg: The configuration dictionary from Hydra.

  Returns:
    Validation recall@1.
  """
  method_name = method.__class__.__name__
  arch.train(False)
  ds.switch_split("valid")
  loader = utils.get_data_loader("sequential", ds, cfg.num_workers,
                                 {"batch_size": 128})

  # Extract embeddings and ids
  embs = []
  ids = []
  with torch.no_grad():
    for batch in loader:
      ids.append(batch["ids"].detach().cpu().numpy())
      examples = batch["examples"]
      if torch.cuda.is_available():
        examples = examples.cuda()
      embs.append(method.get_embeddings(examples).detach().cpu().numpy())
  embs = np.concatenate(embs, axis=0)
  ids = np.concatenate(ids, axis=0)

  # For l2-normalized methods
  if method_name in ["VMFSoftmax", "ArcFace", "NormalizedSoftmaxCE"]:
    norm_method = utils.get_norm_method_by_name("l2")
    embs, norms = norm_method(embs, use_torch=False, return_norms=True)
  # For hyperbolic softmax
  elif method_name == "HyperbolicSoftmaxCE":
    norm_method = utils.get_norm_method_by_name("hyperbolic")
    embs = norm_method(embs, use_torch=False, return_norms=False, c=method.c)

  # For vMF, need to marginalize over samples
  if method_name == "VMFSoftmax":
    with torch.no_grad():
      z = torch.from_numpy(embs)
      z_norms = torch.from_numpy(norms)
      if torch.cuda.is_available():
        z = z.cuda()
        z_norms = z_norms.cuda()
      z_dist = distributions.VonMisesFisher(z, z_norms)
      z_samples = (
          z_dist.sample(torch.Size([method.n_samples
                                   ])).permute(1, 0, 2).detach().cpu().numpy())

    norms = norms.squeeze(1)
    corrects = []
    for i in range(method.n_samples):
      z = z_samples[:, i, :]
      index = faiss.IndexFlatIP(z.shape[1])
      # pylint: disable=no-value-for-parameter
      index.add(z)
      # pylint: disable=no-value-for-parameter
      _, idxs = index.search(z, 2)
      preds = ids[idxs[:, 1]]
      correct = ids == preds
      corrects.append(correct)
    corrects = np.array(corrects)
    valid_acc = np.mean(mode(corrects, axis=0)[0])

  # For hyperbolic, need to compute Poincare distance matrix
  elif method_name == "HyperbolicSoftmaxCE":
    # Since hyperbolic distance is non-trivial to compute, we use numpy instead
    # of faiss
    dist_matrix = hypnn.pairwise_dist_matrix(embs, method.c, batch_size=256)
    # NOTE: Need to use kNN with precomputed distances since we're using
    # hyperbolic distance
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1, metric="precomputed")
    knn.fit(dist_matrix, ids)
    idxs = knn.kneighbors(return_distance=False)
    preds = np.squeeze(ids[idxs], axis=1)
    correct = ids == preds
    valid_acc = np.mean(correct)

  # For all other methods, just compute pairwise distances
  else:
    index = faiss.IndexFlatL2(embs.shape[1])
    # pylint: disable=no-value-for-parameter
    index.add(embs)
    # pylint: disable=no-value-for-parameter
    _, idxs = index.search(embs, 2)
    preds = ids[idxs[:, 1]]
    correct = ids == preds
    valid_acc = np.mean(correct)

  return valid_acc
