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

"""Utility functions for evaluation."""

import numpy as np


def calc_ece_em_quant(fx, y, n_bins, lambda_exp=2):
  """Computes expected calibration error with equal-mass bins.

  Args:
    fx: Model's softmax predictions in range [0, 1].
    y: Binary variable for if prediction is correct class or not (either 0 or
      1).
    n_bins: Number of bins in histogram to do equal mass binning, sort the items
      and partition them by sorted index.
    lambda_exp: Exponent used for computing L_p norm across bins.

  Returns:
    Expected calibration error.
  """
  sort_ix = np.argsort(fx)
  n_examples = fx.shape[0]
  bins = np.zeros((n_examples), dtype=int)
  bins[sort_ix] = np.minimum(
      n_bins - 1, np.floor(
          (np.arange(n_examples) / n_examples) * n_bins)).astype(int)
  return _calc_quant_postbin(n_bins, bins, fx, y, lambda_exp)


def _calc_quant_postbin(n_bins, bins, fx, y, lambda_exp=2):
  """Utility function in computing expected calibration error.

  Args:
    n_bins: Number of bins in histogram to do equal mass binning, sort the items
      and partition them by sorted index.
    bins: ndarray containing the bin each instance belongs to.
    fx: Model's softmax predictions in range [0, 1].
    y: Binary variable for if prediction is correct class or not (either 0 or
      1).
    lambda_exp: Exponent used for computing L_p norm across bins.

  Returns:
    Expected calibration error.
  """
  ece = 0.0
  for i in range(n_bins):
    cur = bins == i
    if any(cur):
      fxm = np.mean(fx[cur])
      ym = np.mean(y[cur])
      n = np.sum(cur)
      ece += n * pow(np.abs(ym - fxm), lambda_exp)
  return pow(ece / fx.shape[0], 1.0 / lambda_exp)
