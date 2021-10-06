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

"""Classes for various softmax cross-entropy methods."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vmf_embeddings import utils
from vmf_embeddings.methods import bessel_approx
from vmf_embeddings.third_party.hyperbolic_image_embeddings import hypnn
from vmf_embeddings.third_party.s_vae_pytorch import distributions


class BaseMethod:
  """Base class for a softmax cross-entropy method."""

  def __init__(self, arch):
    """Initializes a BaseMethod object.

    Args:
      arch: An archs.arch.Arch object (e.g., ResNet50 or N-digit MNIST arch).
    """
    self.arch = arch
    assert (hasattr(self.arch, "forward") and
            hasattr(self.arch, "forward_embedding") and
            hasattr(self.arch, "encoder"))

  def get_embeddings(self, inputs):
    return self.arch.forward_embedding(inputs)

  def get_class_weights(self):
    return self.arch.encoder[-1].weight


class UnnormalizedSoftmaxCE(BaseMethod):
  """Standard unnormalized softmax cross-entropy."""

  def __init__(self, arch):
    super().__init__(arch)
    self.loss = nn.CrossEntropyLoss()

  def __call__(self, inputs, targets, get_predictions=False):
    embs = self.get_embeddings(inputs)
    output_weights = self.get_class_weights()

    preds = torch.mm(embs, output_weights.transpose(0, 1))
    if get_predictions:
      return preds, embs

    return self.loss(preds, targets)


class NormalizedSoftmaxCE(BaseMethod):
  """Cosine softmax (or normalized) cross-entropy."""

  def __init__(self, arch):
    super().__init__(arch)
    self.norm_method = utils.get_norm_method_by_name("l2")
    self.loss = nn.CrossEntropyLoss()
    assert hasattr(self.arch, "temp")

  def __call__(self, inputs, targets, get_predictions=False):
    embs = self.get_embeddings(inputs)
    output_weights = self.norm_method(self.get_class_weights())

    preds = torch.mm(self.norm_method(embs), output_weights.transpose(0, 1))
    preds = preds / torch.exp(self.arch.temp)

    if get_predictions:
      return preds, embs

    return self.loss(preds, targets)


class HyperbolicSoftmaxCE(BaseMethod):
  """Hyperbolic softmax cross-entropy https://arxiv.org/pdf/1904.02239.pdf."""

  def __init__(self, c, arch):
    """Initializes a HyperbolicSoftmaxCE object.

    Args:
      c: Curvature of the Poincare ball.
      arch: An archs.arch.Arch object (e.g., ResNet50 or N-digit MNIST arch).
    """
    super().__init__(arch)
    assert hasattr(self.arch, "embedding_dim") and hasattr(
        self.arch, "n_classes")
    self.c = c
    # This is used to project onto Poincare ball
    self.arch.tp = hypnn.ToPoincare(c)
    self.arch.mlr = hypnn.HyperbolicMLR(
        ball_dim=arch.embedding_dim, n_classes=arch.n_classes, c=c)
    self.loss = F.nll_loss

  def __call__(self, inputs, targets, get_predictions=False):
    embs = self.get_embeddings(inputs)
    preds = self.arch.mlr(self.arch.tp(embs))

    if get_predictions:
      return preds, embs

    return self.loss(F.log_softmax(preds, dim=1), targets)


class VMFBase(BaseMethod):
  """Base class for vMF softmax cross-entropy method."""

  def __init__(self, n_samples, arch):
    """Initializes a VMFBase object.

    Args:
      n_samples: Number of samples to draw during training and testing.
      arch: An archs.arch.Arch object (e.g., ResNet50 or N-digit MNIST arch).
    """
    super().__init__(arch)
    self.dist = distributions.VonMisesFisher
    self.n_samples = n_samples
    self.l2_norm_method = utils.get_norm_method_by_name("l2")
    assert hasattr(self.arch, "z_kappa_mult") and hasattr(self.arch, "temp")

  def get_embeddings(self, inputs):
    return self.arch.z_kappa_mult * self.arch.forward_embedding(inputs)

  def get_predictions(self, inputs, n_samples):
    """Computes class predictions for set of inputs using n_samples samples."""
    z = self.get_embeddings(inputs)
    mu_z, kappa_z = self.l2_norm_method(z, return_norms=True)
    z_dist = self.dist(mu_z, kappa_z)

    weights = self.get_class_weights()
    mu_w, kappa_w = self.l2_norm_method(weights, return_norms=True)
    w_dist = self.dist(mu_w, kappa_w)

    batch_size = z.size(0)

    # Sample from z and w's and marginalize over samples
    z_samples = z_dist.sample(torch.Size([n_samples])).permute(1, 0, 2)
    w_samples = w_dist.sample(torch.Size([batch_size, n_samples]))

    mat = torch.bmm(
        z_samples.reshape(z_samples.size(0) * z_samples.size(1),
                          -1).unsqueeze(1),
        w_samples.view(batch_size * n_samples, -1,
                       w_samples.size(-1)).permute(0, 2, 1),
    ).view(z_samples.size(0), n_samples, w_samples.size(2))

    return mat / torch.exp(self.arch.temp), z


class VMFSoftmax(VMFBase):
  """von Mises-Fisher softmax cross-entropy."""

  def __init__(self, n_samples, arch):
    """Initializes a VMFSoftmax object.

    Args:
      n_samples: Number of samples to draw during training and testing.
      arch: An archs.arch.Arch object (e.g., ResNet50 or N-digit MNIST arch).
    """
    super().__init__(n_samples, arch)
    self.vmf_projection = utils.get_norm_method_by_name("vmf")
    assert hasattr(self.arch, "embedding_dim")
    self.dim = self.arch.embedding_dim

  def __call__(self, inputs, targets, n_samples=None, get_predictions=False):
    # If n_samples is None, sample self.n_samples times
    if n_samples is None:
      n_samples = self.n_samples

    if get_predictions:
      return self.get_predictions(inputs, n_samples)

    z = self.get_embeddings(inputs)
    weights = self.get_class_weights()

    beta = 1.0 / torch.exp(self.arch.temp)

    exp_z, kappa_z = self.vmf_projection(z, return_norms=True)
    exp_c = self.vmf_projection(weights[targets], return_norms=False)
    second_term = beta * (exp_z * exp_c).sum(dim=1)

    batch_size = z.size(0)
    dim = z.size(1)

    mu_z = z / kappa_z
    z_dist = self.dist(mu_z, kappa_z)
    z_samples = z_dist.rsample(torch.Size([n_samples])).permute(1, 0, 2)
    z_s = beta * z_samples

    kappa_w_sq = (weights**2).sum(dim=1, keepdim=False)
    log_c_d_kappa_w = bessel_approx.log_vmf_normalizer_approx(kappa_w_sq, dim)

    # Since log_vmf_normalizer_approx operates on squared-kappa
    # we can simplify the argument in the denominator of the first term
    # of the objective
    log_c_d_denom = bessel_approx.log_vmf_normalizer_approx(
        kappa_w_sq.unsqueeze(0).unsqueeze(0) + beta**2 + 2.0 * torch.mm(
            z_s.reshape(-1, dim),
            torch.transpose(weights, 1, 0),
        ).view(batch_size, n_samples, -1),
        dim,
    )

    first_term = torch.log(
        torch.exp(log_c_d_kappa_w.unsqueeze(0).unsqueeze(0) -
                  log_c_d_denom).sum(dim=2)).mean(dim=1)

    return (first_term - second_term).mean()
