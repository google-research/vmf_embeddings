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

"""Class for instantiating a ResNet in PyTorch.

Code adapted from:
    https://github.com/jeromerony/dml_cross_entropy/blob/master/models/base_model.py
    https://github.com/jeromerony/dml_cross_entropy/blob/master/models/architectures/resnet.py
"""

import logging

import torch.nn as nn
from torchvision.models import resnet
from torchvision.models.utils import load_state_dict_from_url

from vmf_embeddings.archs import arch
from vmf_embeddings.archs import utils

log = logging.getLogger("main")


class ResNet(arch.Arch):
  """Class for defining a ResNet architecture."""

  def __init__(
      self,
      n_classes,
      embedding_dim,
      set_bn_eval,
      first_conv_3x3,
      use_vmf,
      learn_temp,
      init_temp,
      kappa_confidence,
      block,
      layers,
      groups=1,
      width_per_group=64,
      replace_stride_with_dilation=None,
  ):
    """Initializes a ResNet architecture object. See arguments in arch.py."""
    super(ResNet, self).__init__(embedding_dim, n_classes, use_vmf, learn_temp,
                                 init_temp, kappa_confidence)
    self.backbone_features = 512 * block.expansion
    self._norm_layer = nn.BatchNorm2d

    # Fixes batch-norm to eval mode during training
    self.set_bn_eval = set_bn_eval

    # Make first convolution use a 3x3 kernel for CIFAR datasets
    self.first_conv_3x3 = first_conv_3x3

    # Linear layer that remaps from the backbone output of ResNet
    # to the embedding dimensionality
    self.remap = nn.Linear(self.backbone_features, self.embedding_dim)
    nn.init.zeros_(self.remap.bias)

    self.classifier = nn.Linear(self.embedding_dim, self.n_classes, bias=False)

    if self.use_vmf:
      # This is the empirical approximation for initialization the vMF
      # distributions for each class in the final layer.
      utils.vmf_class_weight_init(self.classifier.weight, self.kappa_confidence,
                                  self.embedding_dim)

    self.inplanes = 64
    self.dilation = 1
    if replace_stride_with_dilation is None:
      # Each element in the tuple indicates if we should replace
      # the 2x2 stride with a dilated convolution instead
      replace_stride_with_dilation = [False, False, False]
    if len(replace_stride_with_dilation) != 3:
      raise ValueError(
          "replace_stride_with_dilation should be None "
          "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
    self.groups = groups
    self.base_width = width_per_group

    if self.first_conv_3x3:
      self.conv1 = nn.Conv2d(
          3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
    else:
      self.conv1 = nn.Conv2d(
          3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

    self.bn1 = self._norm_layer(self.inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(
        block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
    self.layer3 = self._make_layer(
        block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
    self.layer4 = self._make_layer(
        block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    # Zero-init
    for m in self.modules():
      if isinstance(m, resnet.Bottleneck):
        nn.init.constant_(m.bn3.weight, 0)
      elif isinstance(m, resnet.BasicBlock):
        nn.init.constant_(m.bn2.weight, 0)

  def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation
    if dilate:
      self.dilation *= stride
      stride = 1
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
          norm_layer(planes * block.expansion),
      )

    layers = []
    layers.append(
        block(
            self.inplanes,
            planes,
            stride,
            downsample,
            self.groups,
            self.base_width,
            previous_dilation,
            norm_layer,
        ))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(
          block(
              self.inplanes,
              planes,
              groups=self.groups,
              base_width=self.base_width,
              dilation=self.dilation,
              norm_layer=norm_layer,
          ))

    return nn.Sequential(*layers)

  def create_encoder(self):
    self.encoder = nn.Sequential(
        self.conv1,
        self.bn1,
        self.relu,
        self.maxpool,
        self.layer1,
        self.layer2,
        self.layer3,
        self.layer4,
        self.avgpool,
        utils.Flatten(),
        self.remap,
        self.classifier,
    )

  def train(self, mode=True):
    """Sets the module in training mode.

    This has any effect only on certain modules. See documentations of
    particular modules for details of their behaviors in training/evaluation
    mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`, etc.

    Args:
      mode: whether to set training mode ("True") or evaluation mode ("False").

    Returns:
      self
    """
    self.training = mode
    for module in self.children():
      module.train(mode)

    if self.set_bn_eval:
      for module in self.modules():
        if isinstance(module, nn.BatchNorm2d):
          module.eval()
    return self


def _resnet(
    arch_name,
    block,
    layers,
    pretrained,
    progress,
    n_classes,
    embedding_dim,
    set_bn_eval,
    first_conv_3x3,
    use_vmf,
    learn_temp,
    init_temp,
    kappa_confidence,
):
  """Instantiates a ResNet model."""
  model = ResNet(
      n_classes,
      embedding_dim,
      set_bn_eval,
      first_conv_3x3,
      use_vmf,
      learn_temp,
      init_temp,
      kappa_confidence,
      block,
      layers,
  )
  if pretrained:
    log.info("Loading ResNet50 from Pytorch pretrained")
    state_dict = load_state_dict_from_url(
        resnet.model_urls[arch_name], progress=progress)
    model.load_state_dict(state_dict, strict=False)
  model.create_encoder()
  return model


def resnet50(
    n_classes,
    embedding_dim,
    set_bn_eval,
    pretrained,
    first_conv_3x3,
    use_vmf,
    learn_temp,
    init_temp,
    kappa_confidence,
    progress=False,
):
  """ResNet-50 model from "Deep Residual Learning for Image Recognition"."""
  return _resnet(
      "resnet50",
      resnet.Bottleneck,
      [3, 4, 6, 3],
      pretrained,
      progress,
      n_classes,
      embedding_dim,
      set_bn_eval,
      first_conv_3x3,
      use_vmf,
      learn_temp,
      init_temp,
      kappa_confidence,
  )
