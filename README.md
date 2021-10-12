# von Mises-Fisher Embeddings

**Disclaimer: This is not an officially supported Google product.**

This repository contains the code for "von Mises-Fisher Loss: An Exploration of
Embedding Geometries for Supervised Learning" (ICCV 2021)
[[arXiv link]](https://arxiv.org/abs/2103.15718).

Created by:

*   Tyler R. Scott (tysc7237@colorado.edu), Research Intern + Student Researcher
    @ Google (Summer 2020 - Spring 2021)
*   Andrew C. Gallagher
*   Michael C. Mozer

## Usage

All commands should be run at the top-level `vmf_embeddings` directory.

## Installation

Requires Python 3.7+

`pip install -r requirements.txt`

## Test Examples

To run a suite of small MNIST & CIFAR10 example scripts to ensure the code is
running properly:

`bash run.sh`

## Datasets

### MNIST, FashionMNIST, CIFAR10, CIFAR100

These datasets rely on `torchvision`, and thus do not require any preprocessing.
They will be automatically downloaded to the directory indicated by
`dataset_path`, which must be specified in the configuration of a run in
advance. If the dataset is already downloaded, it will not be downloaded to the
same path again.

### Cars196

1.  Run this script to download and process Cars196: `python -m
    vmf_embeddings.third_party.dml_cross_entropy.download_and_process_cars196 <path_to_save_processed_data>`
2.  Update the `dataset_path` in `vmf_embeddings/conf/dataset/cars196.yaml` with
    `<path_to_save_processed_data>`

### CUB200-2011

1.  Run this script to download and process CUB200: `python -m
    vmf_embeddings.third_party.dml_cross_entropy.download_and_process_cub <path_to_save_processed_data>`
2.  Update the `dataset_path` in `vmf_embeddings/conf/dataset/cub200.yaml` with
    `<path_to_save_processed_data>/CUB_200_2011`

### Stanford Online Products (SOP)

1.  Run this script to download and process SOP: `python -m
    vmf_embeddings.third_party.dml_cross_entropy.download_and_process_sop <path_to_save_processed_data>`
2.  Update the `dataset_path` in `vmf_embeddings/conf/dataset/sop.yaml` with
    `<path_to_save_processed_data>`

## Train Models

Below are example commands to train networks using the vMF Softmax Cross-Entropy
loss on the various datasets. Note that configurations may change depending on
the method used. See the `vmf_embeddings/conf` directory for specific
configuration settings. All configurations are created using Hydra
(https://hydra.cc/). Also, it may be useful to see the default configuration
choices, if they aren't specified explicitly.

Models and logs will be saved to directories in `vmf_embeddings/runs` based on
the dataset, method, dimensionality, and parameters. If running a hyperparameter
sweep, models will be saved in the `sweeps` directory.

Hyperparameters for all of the method-dataset pairs are included in the appendix
of the linked paper above.

### MNIST

`python -m vmf_embeddings.main name=mnist_vmf_softmax mode.learning_rate=1.0 mode.nesterov=True
mode.momentum=0.99 mode.temp_learning_rate=0.001 arch=n_digit_mnist
arch.params.embedding_dim=3 arch.params.n_classes=10 arch.params.use_vmf=True
arch.params.learn_temp=True sampler=episodic sampler.n_episodes=250
sampler.n_classes=10 sampler.n_samples=13 method=vmf_softmax dataset=mnist`

### FashionMNIST

`python -m vmf_embeddings.main name=fashion_mnist_vmf_softmax mode.learning_rate=0.05
mode.nesterov=False mode.momentum=0.99 mode.temp_learning_rate=0.001
arch=n_digit_mnist arch.params.embedding_dim=3 arch.params.n_classes=10
arch.params.use_vmf=True arch.params.learn_temp=True sampler=episodic
sampler.n_episodes=250 sampler.n_classes=10 sampler.n_samples=13
method=vmf_softmax dataset=fashion_mnist`

### CIFAR10

`python -m vmf_embeddings.main name=cifar10_vmf_softmax mode.learning_rate=0.5
mode.nesterov=False mode.momentum=0.9 mode.temp_learning_rate=0.001
mode.weight_decay=0.0001 arch=resnet50 arch.params.embedding_dim=128
arch.params.n_classes=10 arch.params.use_vmf=True arch.params.learn_temp=True
sampler=episodic sampler.n_episodes=250 sampler.n_classes=10
sampler.n_samples=26 method=vmf_softmax dataset=cifar10`

### CIFAR100

`python -m vmf_embeddings.main name=cifar100_vmf_softmax mode.learning_rate=0.1
mode.nesterov=True mode.momentum=0.99 mode.temp_learning_rate=0.01
mode.weight_decay=0.0001 arch=resnet50 arch.params.embedding_dim=128
arch.params.n_classes=100 arch.params.use_vmf=True arch.params.learn_temp=True
sampler=episodic sampler.n_episodes=250 sampler.n_classes=32 sampler.n_samples=8
method=vmf_softmax dataset=cifar100`

### Cars196

`python -m vmf_embeddings.main name=cars196_vmf_softmax mode.learning_rate=0.0001
mode.nesterov=False mode.momentum=0.9 mode.temp_learning_rate=0.01
mode.weight_decay=0.0001 arch=resnet50 arch.params.embedding_dim=512
arch.params.n_classes=83 arch.params.set_bn_eval=True
arch.params.pretrained=True arch.params.first_conv_3x3=False
arch.params.use_vmf=True arch.params.learn_temp=True
arch.params.kappa_confidence=0.7 sampler=episodic sampler.n_episodes=100
sampler.n_classes=32 sampler.n_samples=4 method=vmf_softmax dataset=cars196`

### CUB200-2011

`python -m vmf_embeddings.main name=cub200_vmf_softmax mode.learning_rate=0.001
mode.nesterov=False mode.momentum=0.9 mode.temp_learning_rate=0.01
mode.weight_decay=0.0001 arch=resnet50 arch.params.embedding_dim=512
arch.params.n_classes=85 arch.params.set_bn_eval=True
arch.params.pretrained=True arch.params.first_conv_3x3=False
arch.params.use_vmf=True arch.params.learn_temp=True
arch.params.kappa_confidence=0.7 arch.params.init_temp=-2.773 sampler=episodic
sampler.n_episodes=100 sampler.n_classes=32 sampler.n_samples=4
method=vmf_softmax dataset=cub200`

### Stanford Online Products

`python -m vmf_embeddings.main name=sop_vmf_softmax mode.learning_rate=0.0001
mode.nesterov=True mode.momentum=0.9 mode.temp_learning_rate=0.01
mode.weight_decay=0.0005 arch=resnet50 arch.params.embedding_dim=512
arch.params.n_classes=9620 arch.params.set_bn_eval=True
arch.params.pretrained=True arch.params.first_conv_3x3=False
arch.params.use_vmf=True arch.params.learn_temp=True
arch.params.kappa_confidence=0.7 arch.params.init_temp=-2.773 sampler=episodic
sampler.n_episodes=500 sampler.n_classes=32 sampler.n_samples=2
method=vmf_softmax dataset=sop`

## Evaluate Models

To evaluate a model, you must first extract embeddings, as well as other
necessary data such as the temperature or classification weights, and then run
the evaluation script. For vMF Softmax, the evaluation script is
`vmf_embeddings/third_party/pytorch_metric_learning/compute_vmf_performance_metrics.py`. For all other
methods, the script is
`vmf_embeddings/third_party/pytorch_metric_learning/compute_deterministic_performance_metrics.py`.

Below are commands for extracting embeddings and evaluating the vMF Softmax
method. Other methods have very similar commands. See the evaluation scripts for
comments about command-line arguments. Note that the `name` specified in the
command for extracting embeddings must match the `name` specified in the command
for training the model.

### MNIST

1. Extract embeddings:

`python -m vmf_embeddings.main arch=n_digit_mnist arch.params.embedding_dim=3
arch.params.n_classes=10 arch.params.use_vmf=True arch.params.learn_temp=True
method=vmf_softmax dataset=mnist name=mnist_vmf_softmax mode=get_embeddings`

2. Run evaluation script:

`python -m vmf_embeddings.third_party.pytorch_metric_learning.compute_vmf_performance_metrics
vmf_embeddings/runs/mnist/vmf_softmax/3dim/mnist_vmf_softmax`

### FashionMNIST

1. Extract embeddings:

`python -m vmf_embeddings.main arch=n_digit_mnist arch.params.embedding_dim=3
arch.params.n_classes=10 arch.params.use_vmf=True arch.params.learn_temp=True
method=vmf_softmax dataset=fashion_mnist name=fashion_mnist_vmf_softmax
mode=get_embeddings`

2. Run evaluation script:

`python -m vmf_embeddings.third_party.pytorch_metric_learning.compute_vmf_performance_metrics
vmf_embeddings/runs/fashion_mnist/vmf_softmax/3dim/fashion_mnist_vmf_softmax`

### CIFAR10

1. Extract embeddings:

`python -m vmf_embeddings.main arch=resnet50 arch.params.embedding_dim=128
arch.params.n_classes=10 arch.params.use_vmf=True arch.params.learn_temp=True
method=vmf_softmax dataset=cifar10 name=cifar10_vmf_softmax mode=get_embeddings`

2. Run evaluation script:

`python -m vmf_embeddings.third_party.pytorch_metric_learning.compute_vmf_performance_metrics
vmf_embeddings/runs/cifar10/vmf_softmax/128dim/cifar10_vmf_softmax`

### CIFAR100

1. Extract embeddings:

`python -m vmf_embeddings.main arch=resnet50 arch.params.embedding_dim=128
arch.params.n_classes=100 arch.params.use_vmf=True arch.params.learn_temp=True
method=vmf_softmax dataset=cifar100 name=cifar100_vmf_softmax
mode=get_embeddings`

2. Run evaluation script:

`python -m vmf_embeddings.third_party.pytorch_metric_learning.compute_vmf_performance_metrics
vmf_embeddings/runs/cifar100/vmf_softmax/128dim/cifar100_vmf_softmax`

### Cars196

1. Extract embeddings:

`python -m vmf_embeddings.main arch=resnet50 arch.params.embedding_dim=512
arch.params.n_classes=83 arch.params.set_bn_eval=True
arch.params.pretrained=True arch.params.first_conv_3x3=False
arch.params.use_vmf=True arch.params.learn_temp=True
arch.params.kappa_confidence=0.7 method=vmf_softmax dataset=cars196
name=cars196_vmf_softmax mode=get_embeddings`

2. Run evaluation script:

`python -m vmf_embeddings.third_party.pytorch_metric_learning.compute_vmf_performance_metrics
vmf_embeddings/runs/cars196/vmf_softmax/512dim/cars196_vmf_softmax`

### CUB200-2011

1. Extract embeddings:

`python -m vmf_embeddings.main arch=resnet50 arch.params.embedding_dim=512
arch.params.n_classes=85 arch.params.set_bn_eval=True
arch.params.pretrained=True arch.params.first_conv_3x3=False
arch.params.use_vmf=True arch.params.learn_temp=True
arch.params.kappa_confidence=0.7 arch.params.init_temp=-2.773 method=vmf_softmax
dataset=cars196 name=cars196_vmf_softmax mode=get_embeddings`

2. Run evaluation script:

`python -m vmf_embeddings.third_party.pytorch_metric_learning.compute_vmf_performance_metrics
vmf_embeddings/runs/cub200/vmf_softmax/512dim/cub200_vmf_softmax`

### Stanford Online Products

1. Extract embeddings:

`python -m vmf_embeddings.main arch=resnet50 arch.params.embedding_dim=512
arch.params.n_classes=9620 arch.params.set_bn_eval=True
arch.params.pretrained=True arch.params.first_conv_3x3=False
arch.params.use_vmf=True arch.params.learn_temp=True
arch.params.kappa_confidence=0.7 arch.params.init_temp=-2.773 method=vmf_softmax
dataset=sop name=sop_vmf_softmax mode=get_embeddings`

2. Run evaluation script:

`python -m vmf_embeddings.third_party.pytorch_metric_learning.compute_vmf_performance_metrics
vmf_embeddings/runs/sop/vmf_softmax/512dim/sop_vmf_softmax`
