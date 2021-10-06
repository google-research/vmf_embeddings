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

"""Main driver for training embedding networks."""

import logging
import os

import hydra
import numpy as np
import torch

from vmf_embeddings import evaluate
from vmf_embeddings import utils

log = logging.getLogger(__name__)


def train(cfg):
  """Trains the backbone embedding network."""
  utils.set_seeds(cfg.random_seed)
  if torch.cuda.is_available():
    gpu_idx = utils.get_gpu_by_usage()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
  open_set_datasets = ["cars196", "stanfordonlineproducts", "cub200"]

  ds = hydra.utils.instantiate(cfg.dataset)
  # Remove the "Dataset" suffix from the class name
  ds_name = ds.__class__.__name__.lower()[:-7]
  arch = hydra.utils.instantiate(cfg.arch)
  method = hydra.utils.instantiate(cfg.method, arch=arch)
  method_name = method.__class__.__name__
  if torch.cuda.is_available():
    arch = arch.cuda()

  if method_name == "VMFSoftmax":
    assert arch.use_vmf, "use_vmf must be true if using VMFSoftmax."
    # If using the vMF Softmax, need to run data through net
    # to estimate the value of z_kappa_mult
    utils.set_z_kappa_mult_vmf(arch, method, ds, cfg.num_workers)
  else:
    assert not arch.use_vmf, ("use_vmf should be False if not using the vMF "
                              "method.")

  p = utils.get_parameter_list(arch, cfg.mode.temp_learning_rate)

  optim = torch.optim.SGD(
      p,
      lr=cfg.mode.learning_rate,
      momentum=cfg.mode.momentum,
      weight_decay=cfg.mode.weight_decay,
      nesterov=(cfg.mode.nesterov and 0.0 < cfg.mode.momentum < 1.0),
  )

  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optim, mode="max", factor=0.5, patience=15)

  best_epoch = {
      "epoch": -1,
      "train_loss": float("inf"),
      "valid_acc": 0.0,
  }
  for e in range(cfg.mode.epochs):
    # Training loop
    arch.train(True)
    ds.switch_split("train")
    loader = utils.get_data_loader(cfg.sampler.name, ds, cfg.num_workers,
                                   cfg.sampler)
    losses = []
    for batch in loader:
      optim.zero_grad()

      examples = batch["examples"]
      ids = batch["ids"]
      if torch.cuda.is_available():
        examples = examples.cuda()
        ids = ids.cuda()

      # Catch OOM issues with rejection sampling of vMF
      try:

        # NOTE: This is a hack for stable training of ArcFace.
        # For ArcFace, train with m = 0 for first 20 epochs for MNISTs
        # and train with m = 0 for first 60 epochs for CIFAR10
        # and train with m = 0 for first 100 epochs for CIFAR100
        # unless using pretrained ImageNet weights (Cars196, SOP, CUB200)
        if method_name == "ArcFace" and ((e < 20 and "mnist" in ds_name) or
                                         (e < 100 and "cifar100" in ds_name) or
                                         (e < 60 and "cifar10" in ds_name)):
          # Set margin to zero
          ls = method(examples, ids, set_m_zero=True)
        else:
          ls = method(examples, ids)
        ls.backward()

      except RuntimeError as exception:
        if "out of memory" in str(
            exception).lower() and method_name == "VMFSoftmax":
          log.info(
              "Out of memory rejection sampling from vMF, stopping training")
          for p in arch.parameters():
            if p.grad is not None:
              del p.grad
          if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Exception: %s", exception)
        log.info("Best epoch: %s", best_epoch)
        return

      optim.step()
      losses.append(ls.item())

    train_loss = np.mean(losses)

    valid_acc = 0.0
    if ds_name in open_set_datasets:
      valid_acc = evaluate.open_set_val_loop(arch, ds, method, cfg)
    else:
      valid_acc = evaluate.fixed_set_val_loop(arch, ds, method, cfg)

    scheduler.step(valid_acc)

    log.info(
        ("[Epoch %d / %d] => Train Loss: %.4f, Validation Accuracy: %.4f, "
         "Temp: %.4f, Z-Kappa-Mult: %.4f"),
        e,
        cfg.mode.epochs - 1,
        train_loss,
        valid_acc,
        np.exp(arch.temp.item()) if hasattr(arch, "temp") else 1.0,
        arch.z_kappa_mult.item()
        if isinstance(arch.z_kappa_mult, torch.Tensor) else arch.z_kappa_mult,
    )

    if valid_acc > best_epoch["valid_acc"]:
      previous_epoch = best_epoch["epoch"]
      best_epoch["epoch"] = e
      best_epoch["train_loss"] = train_loss
      best_epoch["valid_acc"] = valid_acc
      utils.save_model(e, best_epoch, arch, optim, scheduler, previous_epoch)

    if e - best_epoch["epoch"] >= cfg.mode.patience:
      log.info("Early stopping epoch: %d\n", e)
      break

  log.info("Best epoch: %s", best_epoch)


def get_embeddings(cfg):
  """Extracts embeddings, logits, class weights, and other data on test set."""
  utils.set_seeds(cfg.random_seed)
  if torch.cuda.is_available():
    gpu_idx = utils.get_gpu_by_usage()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
  open_set_datasets = ["cars196", "stanfordonlineproducts", "cub200"]

  ds = hydra.utils.instantiate(cfg.dataset)
  # Remove the "Dataset" suffix from the class name
  ds_name = ds.__class__.__name__.lower()[:-7]
  arch = hydra.utils.instantiate(cfg.arch)
  method = hydra.utils.instantiate(cfg.method, arch=arch)
  method_name = method.__class__.__name__
  if torch.cuda.is_available():
    arch = arch.cuda()

  checkpoint = utils.load_model()
  arch.load_state_dict(checkpoint["arch_state_dict"])
  arch.train(False)

  if method_name == "VMFSoftmax":
    assert arch.use_vmf, "use_vmf must be true if using VMFSoftmax."
    # Need to set z_kappa_mult, so that it can be restored from checkpoint
    # if the method is vMF Softmax
    if "z_mult" in checkpoint:
      arch.z_kappa_mult = torch.tensor(
          [checkpoint["z_mult"]],
          dtype=torch.float32,
          device="cuda" if torch.cuda.is_available() else "cpu",
          requires_grad=False,
      )
    elif "z_kappa_mult" in checkpoint:
      arch.z_kappa_mult = torch.tensor(
          [checkpoint["z_kappa_mult"]],
          dtype=torch.float32,
          device="cuda" if torch.cuda.is_available() else "cpu",
          requires_grad=False,
      )
    else:
      utils.load_z_kappa_mult_vmf(arch, os.getcwd())
  else:
    assert not arch.use_vmf, ("use_vmf should be False if not using the vMF "
                              "method.")

  for split in ds.get_test_splits():
    ds.switch_split(split)
    loader = utils.get_data_loader("sequential", ds, cfg.num_workers,
                                   {"batch_size": cfg.mode.batch_size})

    data = {"embeddings": []}
    if ds_name not in open_set_datasets:
      data["logits"] = []

    # Hyperbolic SCE doesn't use the classification weights
    # it uses a separate parameterization
    if method_name != "HyperbolicSoftmaxCE":
      data["weights"] = method.get_class_weights().data.cpu().numpy()

    data["temp"] = 1.0
    if hasattr(arch, "temp"):
      data["temp"] = np.exp(arch.temp.item())

    with torch.no_grad():
      for batch in loader:
        examples = batch["examples"]
        ids = batch["ids"]

        if torch.cuda.is_available():
          examples = examples.cuda()
          ids = ids.cuda()

        for k in batch:
          # Don't store input examples unless dataset is some variation of MNIST
          if k == "examples" and "mnist" not in ds_name:
            continue
          if k in data:
            data[k].append(batch[k].detach().cpu().numpy())
          else:
            data[k] = [batch[k].detach().cpu().numpy()]

        if ds_name in open_set_datasets:
          data["embeddings"].append(
              method.get_embeddings(examples).detach().cpu().numpy())
        else:
          if method_name == "VMFSoftmax":
            logits, embs = method(
                examples,
                ids,
                get_predictions=True,
                n_samples=method.n_samples,
            )
          else:
            logits, embs = method(
                examples,
                ids,
                get_predictions=True,
            )
          data["logits"].append(logits.detach().cpu().numpy())
          data["embeddings"].append(embs.detach().cpu().numpy())

    for k in data:
      if k == "weights" or k == "temp":
        continue
      data[k] = np.concatenate(data[k], axis=0)

    log.info("Split %s %s", split, data["embeddings"].shape)

    name = "{}_{}.npz".format(ds_name, split)
    np.savez(os.path.join(os.getcwd(), name), **data)


@hydra.main(config_path="conf", config_name="config.yaml")
def main(cfg):
  log.info("\n %s", cfg)
  if cfg.mode.name == "train":
    train(cfg)
  elif cfg.mode.name == "get_embeddings":
    get_embeddings(cfg)
  else:
    raise ValueError("Unknown mode: {}".format(cfg.mode.name))


if __name__ == "__main__":
  main()
