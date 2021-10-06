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

"""Hyperbolic neural network operations.

Code adapted from:
    https://github.com/leymir/hyperbolic-image-embeddings/blob/master/hyptorch/nn.py
    https://github.com/leymir/hyperbolic-image-embeddings/blob/master/hyptorch/pmath.py
"""

import numpy as np
import torch
import torch.nn as nn


def artanh(x):
  """Computes arctanh."""
  x = np.clip(x, a_min=-1 + 1e-5, a_max=1 - 1e-5)
  return 0.5 * (np.log(1 + x) - np.log(1 - x))


def tanh(x, clamp=15):
  """Computes tanh."""
  return x.clamp(-clamp, clamp).tanh()


class Arsinh(torch.autograd.Function):
  """torch.autograd.Function implementation of arcsinh."""

  @staticmethod
  def forward(ctx, x):
    ctx.save_for_backward(x)
    return (x + torch.sqrt_(1 + x.pow(2))).clamp_min_(1e-5).log_()

  @staticmethod
  def backward(ctx, grad_output):
    (inp,) = ctx.saved_tensors
    return grad_output / (1 + inp**2)**0.5


def arsinh(x):
  """Computes arcsinh."""
  return Arsinh.apply(x)


def pairwise_dist_matrix(embs, c, batch_size=100):
  """Computes pairwise distance matrix using hyperbolic distance in batches."""
  full_dists = []
  y2 = np.sum(np.power(embs, 2), axis=1, keepdims=True)  # B x 1
  # Compute pairwise distances in batches of rows instead of full matrix.
  # Doing this can help with out-of-memory errors on large matrices.
  # For full pairwise matrix computation, set batch_size to be number of rows in
  # matrix.
  for i in range(0, len(embs), batch_size):
    query = embs[i:i + batch_size]
    xy = np.einsum("ij,kj->ik", -query, embs)
    x2 = y2[i:i + batch_size]
    num = 1 + 2 * c * xy + c * np.transpose(y2, (1, 0))  # B x B
    num = np.expand_dims(num, axis=2) * np.expand_dims(-query, axis=1)
    num = num + np.expand_dims(1 - c * x2, axis=2) * embs  # B x C x D
    denom_part1 = 1 + 2 * c * xy  # B x C
    denom_part2 = c**2 * x2 * np.transpose(y2, (1, 0))
    denom = denom_part1 + denom_part2
    res = num / (np.expand_dims(denom, axis=2) + 1e-5)
    sqr_c = np.sqrt(c)
    dist_c = artanh(sqr_c * np.linalg.norm(res, ord=2, axis=2))
    full_dists.append(dist_c * 2.0 / sqr_c)
  full_dists = np.concatenate(full_dists, axis=0)
  assert (full_dists.shape[0] == full_dists.shape[1] and
          full_dists.shape[0] == embs.shape[0])
  return full_dists


def project(x, c):
  """Projects x into the Poincare ball with curvature, c."""
  c = torch.as_tensor(c).type_as(x)
  norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
  maxnorm = (1 - 1e-3) / (c**0.5)
  cond = norm > maxnorm
  projected = x / norm * maxnorm
  return torch.where(cond, projected, x)


def lambda_x(x, c, keepdim=False):
  """Computes the conformal factor of x with curvature c."""
  c = torch.as_tensor(c).type_as(x)
  return 2 / (1 - c * x.pow(2).sum(-1, keepdim=keepdim))


def mobius_add(x, y, c):
  """Performs Mobius addition operator between x and y with curvature c."""
  c = torch.as_tensor(c).type_as(x)
  x2 = x.pow(2).sum(dim=-1, keepdim=True)
  y2 = y.pow(2).sum(dim=-1, keepdim=True)
  xy = (x * y).sum(dim=-1, keepdim=True)
  num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
  denom = 1 + 2 * c * xy + c**2 * x2 * y2
  return num / (denom + 1e-5)


def mobius_addition_batch(x, y, c):
  """Performs a vectorized Mobius addition operator between x and y."""
  xy = tensor_dot(x, y)  # B x C
  x2 = x.pow(2).sum(-1, keepdim=True)  # B x 1
  y2 = y.pow(2).sum(-1, keepdim=True)  # C x 1
  num = 1 + 2 * c * xy + c * y2.permute(1, 0)  # B x C
  num = num.unsqueeze(2) * x.unsqueeze(1)
  num = num + (1 - c * x2).unsqueeze(2) * y  # B x C x D
  denom_part1 = 1 + 2 * c * xy  # B x C
  denom_part2 = c**2 * x2 * y2.permute(1, 0)
  denom = denom_part1 + denom_part2
  res = num / (denom.unsqueeze(2) + 1e-5)
  return res


def expmap0(u, c):
  """Applies the exponential map on u at 0 with curvature c."""
  c = torch.as_tensor(c).type_as(u)
  sqrt_c = c**0.5
  u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
  gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
  return gamma_1


def expmap(x, u, c):
  """Applies the exponential map on u at tangent of x with curvature c."""
  c = torch.as_tensor(c).type_as(x)
  sqrt_c = c**0.5
  u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
  second_term = (
      tanh(sqrt_c / 2 * lambda_x(x, c, keepdim=True) * u_norm) * u /
      (sqrt_c * u_norm))
  gamma_1 = mobius_add(x, second_term, c)
  return gamma_1


def tensor_dot(x, y):
  """Performs a tensor dot product."""
  res = torch.einsum("ij,kj->ik", (x, y))
  return res


def hyperbolic_softmax(x, a, p, c):
  """Computes the hyperbolic softmax function."""
  lambda_pkc = 2 / (1 - c * p.pow(2).sum(dim=1).clamp_max((1.0 / c) - 1e-4))
  k = lambda_pkc * torch.norm(a, dim=1).clamp_min(1e-7) / torch.sqrt(c)
  mob_add = mobius_addition_batch(-p, x, c)
  num = 2 * torch.sqrt(c) * torch.sum(mob_add * a.unsqueeze(1), dim=-1)
  denom = torch.norm(
      a, dim=1,
      keepdim=True).clamp_min(1e-7) * (1 - c * mob_add.pow(2).sum(dim=2))
  logit = k.unsqueeze(1) * arsinh(num / denom)
  return logit.permute(1, 0)


class RiemannianGradient(torch.autograd.Function):
  """Computes Riemannian gradients for backward pass."""
  c = 1

  @staticmethod
  def forward(ctx, x):
    ctx.save_for_backward(x)
    return x

  @staticmethod
  def backward(ctx, grad_output):
    (x,) = ctx.saved_tensors
    scale = (1 -
             RiemannianGradient.c * x.pow(2).sum(-1, keepdim=True)).pow(2) / 4
    return grad_output * scale


class HyperbolicMLR(nn.Module):
  """Performs softmax classification in Hyperbolic space."""

  def __init__(self, ball_dim, n_classes, c):
    """Initializes a HyperbolicMLR object.

    Args:
      ball_dim: Dimensionality of the embedding space.
      n_classes: Number of classes for training the network.
      c: Curvature of the Poincare ball.
    """
    super(HyperbolicMLR, self).__init__()
    self.a_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
    self.p_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
    self.c = c
    self.n_classes = n_classes
    self.ball_dim = ball_dim
    self.reset_parameters()

  def forward(self, x):
    c = torch.as_tensor(self.c).type_as(x)
    p_vals_poincare = expmap0(self.p_vals, c)
    conformal_factor = 1 - c * p_vals_poincare.pow(2).sum(dim=1, keepdim=True)
    a_vals_poincare = self.a_vals * conformal_factor
    logits = hyperbolic_softmax(x, a_vals_poincare, p_vals_poincare, c)
    return logits

  def reset_parameters(self):
    nn.init.kaiming_uniform_(self.a_vals, a=np.sqrt(5))
    nn.init.kaiming_uniform_(self.p_vals, a=np.sqrt(5))


class ToPoincare(nn.Module):
  """Maps points in d-dim Euclidean space to d-dim Poincare ball."""

  def __init__(self, c, riemannian=True):
    """Initializes a ToPoincare object.

    Args:
      c: Curvature of the Poincare ball.
      riemannian: Boolean for if using Riemannian gradients.
    """
    super(ToPoincare, self).__init__()
    self.c = c
    self.riemannian = RiemannianGradient
    self.riemannian.c = c

    if riemannian:
      self.grad_fix = lambda x: self.riemannian.apply(x)
    else:
      self.grad_fix = lambda x: x

  def forward(self, x):
    return self.grad_fix(project(expmap0(x, c=self.c), self.c))
