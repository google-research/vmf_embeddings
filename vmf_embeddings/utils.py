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

"""General utility functions."""

import logging
import os
import random
import subprocess

import numpy as np
import torch

from vmf_embeddings.datasets import utils as ds_utils
from vmf_embeddings.methods import bessel_approx
from vmf_embeddings.third_party.hyperbolic_image_embeddings import hypnn

log = logging.getLogger("main")


def set_seeds(random_seed):
  """Sets random seeds."""
  random.seed(random_seed)
  np.random.seed(random_seed)
  torch.manual_seed(random_seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  os.environ["PYTHONHASHSEED"] = str(random_seed)


def get_gpu_by_usage():
  """Gets the gpu idx with lowest usage."""
  result = subprocess.check_output([
      "nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"
  ]).decode("utf-8")
  gpu_memory = [int(x) for x in result.strip().split("\n")]
  return np.argmin(gpu_memory)


def get_data_loader(sampler_name, dataset, num_workers, params):
  """Gets a PyTorch DataLoader."""
  loader = None
  if sampler_name == "random":
    # Random sampler is identical to sequential except it shuffles the data.
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=params["batch_size"],
        num_workers=num_workers,
    )
  elif sampler_name == "sequential":
    # Sequential sampler runs through data in the order provided by the dataset.
    loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=params["batch_size"],
        num_workers=num_workers,
    )
  elif sampler_name == "episodic":
    # Episodic samplers first sample n classes and then k instances per class.
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=ds_utils.EpisodicBatchSampler(
            dataset.labels,
            params["n_classes"],
            params["n_samples"],
            params["n_episodes"],
        ),
        num_workers=num_workers,
    )
  else:
    raise ValueError("Unknown sampler name: {}".format(sampler_name))
  return loader


def get_parameter_list(arch, temp_learning_rate=0.0001):
  """Gets list of network parameters with hyperparams set for optimization."""

  # Certain parameters we don't want to use momentum (e.g., temperature) and
  # some parameters we don't want weight decay.
  # This method iterates through the parameters in the network and filters them
  # based on what optimization constraints we apply to them.

  assert len(list(arch.named_parameters())) == len(list(arch.parameters()))
  temp_name = "temp"
  weight_params = list(d[1] for d in filter(
      lambda kv: kv[0] != temp_name and kv[0] != "module." + temp_name and
      (kv[1].dim() > 1 and not kv[0].endswith("bias") and kv[0] !=
       "classifier.weight" and kv[0] != "module.classifier.weight" and not (
           hasattr(arch, "mlr") and kv[0] in ["mlr.a_vals", "mlr.p_vals"])),
      arch.named_parameters(),
  ))
  bias_params = list(d[1] for d in filter(
      lambda kv: kv[0] != temp_name and kv[0] != "module." + temp_name and
      (kv[1].dim() == 1 or kv[0].endswith("bias") or kv[
          0] == "classifier.weight" or kv[0] == "module.classifier.weight" or
       (hasattr(arch, "mlr") and kv[0] in ["mlr.a_vals", "mlr.p_vals"])),
      arch.named_parameters(),
  ))
  temp_param = list(d[1] for d in filter(
      lambda kv: kv[0] == temp_name or kv[0] == "module." + temp_name,
      arch.named_parameters(),
  ))

  p = [
      {
          "params": weight_params
      },
      {
          "params": bias_params,
          "weight_decay": 0.0
      },
  ]

  if hasattr(arch, "temp"):
    p.append({
        "params": temp_param,
        "lr": temp_learning_rate,
        "momentum": 0.0,
        "nesterov": False,
        "weight_decay": 0.0,
    })

  assert len(temp_param) + len(bias_params) + len(weight_params) == len(
      list(arch.named_parameters()))

  return p


def load_z_kappa_mult_vmf(arch, cwd):
  """Loads z_kappa_mult for VMFSoftmax when performing evaluation."""
  z_kappa_mult = 1.0
  with open(os.path.join(cwd, "main.train.log"), "r") as f:
    for line in f:
      if ("Set z-kappa-mult:" not in line) and ("Set z-mult:" not in line):
        continue
      z_kappa_mult = float(line.strip().split()[-1])
  arch.z_kappa_mult = torch.tensor(
      [z_kappa_mult],
      dtype=torch.float32,
      device="cuda" if torch.cuda.is_available() else "cpu",
      requires_grad=False,
  )
  log.info("Load z-kappa-mult: %.4f", arch.z_kappa_mult.item())


def set_z_kappa_mult_vmf(arch, method, ds, num_workers=0):
  """Sets z_kappa_mult for VMFSoftmax for training."""

  # Upon initialization of the network for vMF softmax, set z_kappa_mult so the
  # confidence of the instance distributions is approximately in alignment
  # with the class distributions.
  # This requires passing the training set through the network to compute the
  # expected kappa value which we then scale by z-kappa-mult appropriately.

  arch.train(True)
  ds.switch_split("train")
  loader = get_data_loader("sequential", ds, num_workers, {"batch_size": 256})
  expected_z = []
  with torch.no_grad():
    for batch in loader:
      examples = batch["examples"]
      if torch.cuda.is_available():
        examples = examples.cuda()
      expected_z.append(
          torch.abs(method.get_embeddings(examples)).mean(
              dim=1).detach().cpu().numpy())
  expected_z = np.mean(np.concatenate(expected_z, axis=0))

  # This is an empirical approximation for setting z_kappa_mult that is
  # equivariant with the dimensionality of the latent space
  arch.z_kappa_mult = torch.tensor(
      [(1.0 / expected_z) * (arch.kappa_confidence /
                             (1.0 - arch.kappa_confidence**2)) *
       ((arch.embedding_dim - 1.0) / np.sqrt(arch.embedding_dim))],
      dtype=torch.float32,
      device="cuda" if torch.cuda.is_available() else "cpu",
      requires_grad=False,
  )
  log.info("Set z-kappa-mult: %.4f", arch.z_kappa_mult.item())


def save_model(epoch, best_epoch, arch, optim, scheduler, previous_epoch):
  """Saves a PyTorch model including optimizer and scheduler."""
  torch.save(
      {
          "epoch":
              epoch,
          "best_epoch":
              best_epoch,
          "arch_state_dict":
              arch.state_dict(),
          "optimizer_state_dict":
              optim.state_dict(),
          "scheduler_state_dict":
              scheduler.state_dict(),
          "z_kappa_mult":
              arch.z_kappa_mult.item() if isinstance(
                  arch.z_kappa_mult, torch.Tensor) else arch.z_kappa_mult,
      },
      os.path.join(os.getcwd(), "model_{}.pt".format(epoch)),
  )
  # Remove previously saved checkpoints.
  if previous_epoch >= 0:
    os.remove(os.path.join(os.getcwd(), "model_{}.pt".format(previous_epoch)))


def load_model(loc=None):
  """Loads a PyTorch model."""
  model_f = None
  for f in os.listdir(os.getcwd()):
    if f.startswith("model") and f.endswith(".pt"):
      model_f = f
  if model_f is None:
    raise ValueError("Could not find .pt saved model file")
  log.info("Loading %s", model_f)
  if loc is not None:
    return torch.load(os.path.join(os.getcwd(), model_f), map_location=loc)
  return torch.load(os.path.join(os.getcwd(), model_f))


def get_norms_and_dim(x, use_torch):
  """Computes L2-norm and returns the dimensionality of the embedding space."""
  if use_torch:
    norms = torch.norm(x, p=2, dim=1, keepdim=True)
    dim = x.size(1)
  else:
    norms = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    dim = x.shape[1]
  return norms, dim


def get_norm_method_by_name(name):
  """Returns a norm function based on provided name."""
  if name == "l2":
    method = l2_norm
  elif name == "vmf":
    method = vmf_projection
  elif name == "hyperbolic":
    method = hyperbolic_projection
  elif name == "none":
    method = no_norm
  else:
    raise ValueError("Unknown normalization method: {}".format(name))

  return method


def no_norm(x, use_torch=True, return_norms=False):
  """Identity function when computing norms for consistency."""
  if not return_norms:
    return x
  return x, get_norms_and_dim(x, use_torch)[0]


def l2_norm(x, use_torch=True, return_norms=False):
  """Computes L2 norm."""
  norms = get_norms_and_dim(x, use_torch)[0]
  x = x / norms

  if not return_norms:
    return x
  return x, norms


def vmf_projection(x, use_torch=True, return_norms=False):
  """Projects x via expectation of a vMF distribution."""
  norms, dim = get_norms_and_dim(x, use_torch)
  # L2-normalize the input (i.e., get mu of the vMF distribution)
  x = x / norms

  # NOTE: This is an approximation from https://arxiv.org/pdf/1606.02008.pdf
  # Scale mu by the ratio of modified bessel functions using our approximations
  # Here, the L2-norm represents kappa
  if use_torch:
    x = x * bessel_approx.ratio_of_bessel_approx(norms, dim)
  else:
    x = x * bessel_approx.ratio_of_bessel_approx(torch.from_numpy(norms),
                                                 dim).numpy()

  if not return_norms:
    return x
  return x, norms


def hyperbolic_projection(x, use_torch=True, return_norms=False, c=1.0):
  """Projects the input onto the Poincare ball."""
  with torch.no_grad():
    tp = hypnn.ToPoincare(c, riemannian=False)
    if use_torch:
      x = tp(x)
    else:
      x = tp(torch.from_numpy(x)).numpy()

  if not return_norms:
    return x
  return x, get_norms_and_dim(x, use_torch)[0]
