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

"""Architecture for MNIST and FashionMNIST.

Based on the architecture used in Hedged Instance Embedding.
https://arxiv.org/abs/1810.00319
"""

import torch
import torch.nn as nn

from vmf_embeddings.archs import arch
from vmf_embeddings.archs import utils


class NDigitMNISTArch(arch.Arch):
  """Architecture for MNIST and FashionMNIST."""

  def __init__(self, embedding_dim, n_classes, use_vmf, learn_temp, init_temp,
               kappa_confidence, n_digits):
    """Initializes an NDigitMNISTArch object.

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
      n_digits: Number of MNIST digits in an image. For MNIST and FashionMNIST,
        the value should always be 1.
    """
    super(NDigitMNISTArch,
          self).__init__(embedding_dim, n_classes, use_vmf, learn_temp,
                         init_temp, kappa_confidence)
    self.n_digits = n_digits
    self.encoder = self._create_encoder()

  def _create_encoder(self):
    """Creates a network encoder."""

    def _init_weights(layer):
      """Initializes the weights of a layer based on type."""
      if isinstance(layer, (nn.Conv2d, nn.Linear)):
        torch.nn.init.xavier_uniform_(layer.weight)
        try:
          # Some layers may not have biases, so catch the exception and pass.
          layer.bias.data.fill_(0.0)
        except AttributeError:
          pass

    kernel_size = 5
    pad = 2
    input_channels = 1
    first_conv_channels = 6
    second_conv_channels = 16
    max_pool_kernel = 2
    linear_size = 120
    n_pixels = 7

    encoder = nn.Sequential(
        nn.Conv2d(
            input_channels, first_conv_channels, kernel_size, padding=pad),
        nn.BatchNorm2d(first_conv_channels),
        nn.ReLU(),
        nn.MaxPool2d(max_pool_kernel),
        nn.Conv2d(
            first_conv_channels, second_conv_channels, kernel_size,
            padding=pad),
        nn.BatchNorm2d(second_conv_channels),
        nn.ReLU(),
        nn.MaxPool2d(max_pool_kernel),
        utils.Flatten(),
        nn.Linear(n_pixels * n_pixels * self.n_digits * second_conv_channels,
                  linear_size),
        nn.BatchNorm1d(linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, self.embedding_dim),
        nn.Linear(self.embedding_dim, self.n_classes, bias=False),
    )

    encoder.apply(_init_weights)

    # This is the empirical approximation for initialization the vMF
    # distributions for each class in the final layer.
    if self.use_vmf:
      utils.vmf_class_weight_init(encoder[-1].weight, self.kappa_confidence,
                                  self.embedding_dim)

    return encoder
