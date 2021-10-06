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

"""Functions approximating modified Bessel functions of first-kind for vMF."""

import torch


def ratio_of_bessel_approx(kappa, d):
  """Approximates the ratio of modified Bessel functions of the first kind.

  Args:
    kappa: The value of the concentration parameter for a vMF distribution.
    d: Dimensionality of the embedding space.

  Returns:
    The approximation to the ratio of modified Bessel functions of first kind.
  """
  # NOTE: This is an approximation from https://arxiv.org/pdf/1606.02008.pdf
  kappa_squared = kappa**2

  d_m_half = (d / 2.0) - 0.5
  sqrt_d_m_half = torch.sqrt(d_m_half**2 + kappa_squared)

  d_p_half = (d / 2.0) + 0.5
  sqrt_d_p_half = torch.sqrt(d_p_half**2 + kappa_squared)

  return 0.5 * ((kappa / (d_m_half + sqrt_d_p_half)) +
                (kappa / (d_m_half + sqrt_d_m_half)))


def log_vmf_normalizer_approx(k_squared, d):
  """Approximates log C_d(kappa) from the vMF probability density function.

  Args:
    k_squared: The value of the concentration parameter for a vMF distribution
      squared.
    d: Dimensionality of the embedding space.

  Returns:
    The approximation to log C_d(kappa).
  """
  d_m_half = (d / 2.0) - 0.5
  sqrt_d_m_half = torch.sqrt(d_m_half**2 + k_squared)

  d_p_half = (d / 2.0) + 0.5
  sqrt_d_p_half = torch.sqrt(d_p_half**2 + k_squared)

  return 0.5 * (
      d_m_half * torch.log(d_m_half + sqrt_d_m_half) - sqrt_d_m_half +
      d_m_half * torch.log(d_m_half + sqrt_d_p_half) - sqrt_d_p_half)
