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

"""Utility functions for network architecture and parameter initialization."""

import numpy as np
import torch.nn as nn


class Flatten(nn.Module):
  """Operation that flattens an input except for the batch dimension."""

  def forward(self, x):
    return x.view(x.size(0), -1)


def vmf_class_weight_init(weights, kappa_confidence, embedding_dim):
  """Initializes class weight vectors as vMF random variables."""
  # This is the empirical approximation for initialization the vMF distributions
  # for each class in the final layer (Equation 19 in the paper).
  nn.init.normal_(
      weights,
      mean=0.0,
      std=(kappa_confidence / (1.0 - kappa_confidence**2)) *
      ((embedding_dim - 1.0) / np.sqrt(embedding_dim)),
  )
