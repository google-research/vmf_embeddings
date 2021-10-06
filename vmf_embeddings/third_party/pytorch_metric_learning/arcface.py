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

"""ArcFace softmax cross-entropy."""

import torch
import torch.nn as nn

from vmf_embeddings import utils
from vmf_embeddings.methods import methods


class ArcFace(methods.BaseMethod):
  """ArcFace softmax cross-entropy.

  Code adapted from:
  https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/arcface_loss.py
  """

  def __init__(self, m, arch):
    super().__init__(arch)
    self.norm_method = utils.get_norm_method_by_name("l2")
    self.m = m
    self.eps = 1e-7
    self.loss = nn.CrossEntropyLoss()
    assert hasattr(self.arch, "temp")

  def __call__(self, inputs, targets, get_predictions=False, set_m_zero=False):
    m = self.m
    if set_m_zero:
      m = 0.0

    beta = 1.0 / torch.exp(self.arch.temp)

    embs = self.get_embeddings(inputs)
    output_weights = self.norm_method(self.get_class_weights())
    cos_theta = torch.mm(self.norm_method(embs), output_weights.transpose(0, 1))

    if get_predictions:
      return beta * cos_theta, embs

    mask = torch.zeros(cos_theta.size(), device=inputs.device)
    mask[torch.arange(cos_theta.size(0)), targets] = 1

    # Get target logits and add angular margin
    target_cos = cos_theta[mask == 1]
    # Need to ensure cos_theta isn't +/- 1.0 or acos will return +/- infinity
    target_cos_with_margin = torch.cos(
        torch.acos(torch.clamp(target_cos, -1 + self.eps, 1 - self.eps)) + m)

    diff = (target_cos_with_margin - target_cos).unsqueeze(1)
    preds = cos_theta + (mask * diff)
    preds = beta * preds

    return self.loss(preds, targets)
