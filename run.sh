#!/bin/bash

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

set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install -r requirements.txt

# Standard Softmax MNIST Test
python -m vmf_embeddings.main \
  name=mnist_standard_softmax \
  mode.learning_rate=0.1 \
  mode.nesterov=False \
  mode.momentum=0.9 \
  mode.epochs=1 \
  arch=n_digit_mnist \
  arch.params.embedding_dim=3 \
  arch.params.n_classes=10 \
  arch.params.use_vmf=False \
  arch.params.learn_temp=False \
  sampler=episodic \
  sampler.n_episodes=250 \
  sampler.n_classes=10 \
  sampler.n_samples=13 \
  method=unnormalized_sce \
  dataset=mnist

python -m vmf_embeddings.main arch=n_digit_mnist arch.params.embedding_dim=3 \
arch.params.n_classes=10 arch.params.use_vmf=False arch.params.learn_temp=False \
method=unnormalized_sce dataset=mnist name=mnist_standard_softmax mode=get_embeddings

python -m vmf_embeddings.third_party.pytorch_metric_learning.compute_deterministic_performance_metrics \
vmf_embeddings/runs/mnist/unnormalized_sce/3dim/mnist_standard_softmax none

# Hyperbolic Softmax MNIST Test
python -m vmf_embeddings.main \
  name=mnist_hyperbolic_softmax \
  mode.learning_rate=0.05 \
  mode.nesterov=True \
  mode.momentum=0.99 \
  mode.epochs=1 \
  arch=n_digit_mnist \
  arch.params.embedding_dim=3 \
  arch.params.n_classes=10 \
  arch.params.use_vmf=False \
  arch.params.learn_temp=False \
  sampler=episodic \
  sampler.n_episodes=250 \
  sampler.n_classes=10 \
  sampler.n_samples=13 \
  method=hyperbolic_sce \
  dataset=mnist

python -m vmf_embeddings.main arch=n_digit_mnist arch.params.embedding_dim=3 \
arch.params.n_classes=10 arch.params.use_vmf=False arch.params.learn_temp=False \
method=hyperbolic_sce dataset=mnist name=mnist_hyperbolic_softmax mode=get_embeddings

python -m vmf_embeddings.third_party.pytorch_metric_learning.compute_deterministic_performance_metrics \
vmf_embeddings/runs/mnist/hyperbolic_sce/3dim/mnist_hyperbolic_softmax hyperbolic

# Cosine Softmax MNIST Test
python -m vmf_embeddings.main \
  name=mnist_cosine_softmax \
  mode.learning_rate=0.5 \
  mode.nesterov=True \
  mode.momentum=0.9 \
  mode.temp_learning_rate=0.001 \
  mode.epochs=1 \
  arch=n_digit_mnist \
  arch.params.embedding_dim=3 \
  arch.params.n_classes=10 \
  arch.params.use_vmf=False \
  arch.params.learn_temp=True \
  sampler=episodic \
  sampler.n_episodes=250 \
  sampler.n_classes=10 \
  sampler.n_samples=13 \
  method=normalized_sce \
  dataset=mnist

python -m vmf_embeddings.main arch=n_digit_mnist arch.params.embedding_dim=3 \
arch.params.n_classes=10 arch.params.use_vmf=False arch.params.learn_temp=True \
method=normalized_sce dataset=mnist name=mnist_cosine_softmax mode=get_embeddings

python -m vmf_embeddings.third_party.pytorch_metric_learning.compute_deterministic_performance_metrics \
vmf_embeddings/runs/mnist/normalized_sce/3dim/mnist_cosine_softmax l2

# ArcFace MNIST Test
python -m vmf_embeddings.main \
  name=mnist_arcface_softmax \
  mode.learning_rate=0.05 \
  mode.nesterov=False \
  mode.momentum=0.9 \
  mode.temp_learning_rate=0.001 \
  mode.epochs=1 \
  arch=n_digit_mnist \
  arch.params.embedding_dim=3 \
  arch.params.n_classes=10 \
  arch.params.use_vmf=False \
  arch.params.learn_temp=True \
  sampler=episodic \
  sampler.n_episodes=250 \
  sampler.n_classes=10 \
  sampler.n_samples=13 \
  method=arcface \
  dataset=mnist

python -m vmf_embeddings.main arch=n_digit_mnist arch.params.embedding_dim=3 \
arch.params.n_classes=10 arch.params.use_vmf=False arch.params.learn_temp=True \
method=arcface dataset=mnist name=mnist_arcface_softmax mode=get_embeddings

python -m vmf_embeddings.third_party.pytorch_metric_learning.compute_deterministic_performance_metrics \
vmf_embeddings/runs/mnist/arcface/3dim/mnist_arcface_softmax l2

# vMF MNIST Test
python -m vmf_embeddings.main \
  name=mnist_vmf_softmax \
  mode.learning_rate=1.0 \
  mode.nesterov=True \
  mode.momentum=0.99 \
  mode.temp_learning_rate=0.001 \
  mode.epochs=1 \
  arch=n_digit_mnist \
  arch.params.embedding_dim=3 \
  arch.params.n_classes=10 \
  arch.params.use_vmf=True \
  arch.params.learn_temp=True \
  sampler=episodic \
  sampler.n_episodes=250 \
  sampler.n_classes=10 \
  sampler.n_samples=13 \
  method=vmf_softmax \
  dataset=mnist

python -m vmf_embeddings.main arch=n_digit_mnist arch.params.embedding_dim=3 \
arch.params.n_classes=10 arch.params.use_vmf=True arch.params.learn_temp=True \
method=vmf_softmax dataset=mnist name=mnist_vmf_softmax mode=get_embeddings

python -m vmf_embeddings.third_party.pytorch_metric_learning.compute_vmf_performance_metrics \
vmf_embeddings/runs/mnist/vmf_softmax/3dim/mnist_vmf_softmax

rm -rf vmf_embeddings/runs
