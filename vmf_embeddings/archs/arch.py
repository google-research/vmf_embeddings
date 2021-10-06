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

"""Base class for a network architecture."""

import torch
import torch.nn as nn


class Arch(nn.Module):
  """Base class for a network architecture."""

  def __init__(self, embedding_dim, n_classes, use_vmf, learn_temp, init_temp,
               kappa_confidence):
    """Initializes an Arch object.

    Args:
      embedding_dim: Dimnensionality of the embedding space.
      n_classes: Number of classes to use during training.
      use_vmf: Boolean for if using the vMF loss.
      learn_temp: Boolean for if the temperature is being learned.
      init_temp: Initial value to use for the temperature. Note that the
        temperature is exponentiated.
      kappa_confidence: Specified how certain the class and instance vMF
        distributions should be. Takes a value between [0, 1], where 0
        represents a uniform distribution and 1 represents a delta distribution.
    """
    super(Arch, self).__init__()
    self.encoder = nn.Sequential(nn.Identity())
    self.embedding_dim = embedding_dim
    self.n_classes = n_classes
    self.use_vmf = use_vmf
    self.learn_temp = learn_temp
    self.init_temp = init_temp
    assert self.n_classes > 0

    # NOTE: This value only applies for the vMF softmax method. Kappa confidence
    # specifies how certain the class and instance vMF distributions should be
    # (approximately) upon initialization of the network. This is a value
    # between 0 (uniform distribution) and 1 (delta distribution)
    self.kappa_confidence = kappa_confidence

    # NOTE: This value only applies for the vMF softmax method.
    # z-kappa-mult is a fixed constant that ensures the kappa of the instance
    # distributions has self.kappa_confidence for it's confidence. See
    # ../utils.py for the method that sets the value.
    self.z_kappa_mult = 1.0

    if self.learn_temp:
      self.temp = nn.Parameter(
          torch.zeros(
              1,
              1,
              dtype=torch.float32,
              device="cuda" if torch.cuda.is_available() else "cpu",
              requires_grad=True) + self.init_temp)

  def forward(self, x):
    return self.encoder(x)

  def forward_embedding(self, x):
    """Forwards up until the embedding (i.e., the layer preceding logits.)"""
    for i in range(len(self.encoder) - 1):
      x = self.encoder[i](x)
    return x
