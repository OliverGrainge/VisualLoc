import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from tqdm import tqdm
import numpy as np

from PlaceRec.Training.GSV_Cities.utils import get_loss, get_miner
from PlaceRec.utils import get_config

config = get_config()


class L1UnstructuredPruner:
    def __init__(self, model, prune_step=0.1):
        self.model = model
        self.N = 0
        self.pruner_step = prune_step
        self.current_amount = None

    def step(self):
        self.N += 1
        if self.current_amount is None:
            self.current_amount = self.pruner_step
        else:
            self.current_amount = self.current_amount / (1 - self.current_amount)

        if self.current_amount > 1.0:
            self.current_amount = 1.0

        for module in self.model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=self.current_amount)


def sparsity(method):
    model = method.model
    total_zeros = 0
    total_elements = 0

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            weight_mask = getattr(module, "weight_mask", None)
            if weight_mask is not None:
                total_zeros += torch.sum(weight_mask == 0)
                total_elements += weight_mask.numel()
    overall_sparsity = total_zeros / total_elements if total_elements > 0 else 0
    return overall_sparsity


class TaylorUnstructuredPruner:
    def __init__(self, model, datamodule, prune_step=0.1, n_batch_acc=3):
        self.model = model
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.custom_from_mask(
                    module,
                    name="weight",
                    mask=torch.ones(module.weight.shape).to(
                        next(model.parameters()).device
                    ),
                )

        for param in self.model.parameters():
            param.requires_grad = True

        self.n_batch_acc = n_batch_acc
        self.prune_step = prune_step
        self.datamodule = datamodule
        self.current_amount = None  # Start with no pruning
        self.miner = get_miner("MultiSimilarityMiner")
        self.loss_fn = get_loss("MultiSimilarityLoss")

    def loss_function(self, descriptors, labels):
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
        else:
            loss = self.loss_fn(descriptors, labels)
            if type(loss) == tuple:
                loss, _ = loss
        return loss

    def compute_validation_gradients(self, val_loader):
        self.model.eval()
        self.model.zero_grad()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(dev)

        gradients_accumulated = {}
        count = 0
        for batch in tqdm(
            val_loader,
            total=self.n_batch_acc,
            desc="Computing Gradients for Taylor Importance",
        ):
            count += 1
            places, labels = batch
            BS, N, ch, h, w = places.shape
            images = places.view(BS * N, ch, h, w)
            labels = labels.view(-1)
            descriptors = self.model(images.to(dev))
            loss = self.loss_function(descriptors, labels)
            loss.backward(retain_graph=True)

            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    if module.weight_orig.grad is not None:
                        grad = module.weight_orig.grad.data**2
                        if name in gradients_accumulated:
                            gradients_accumulated[name] += grad
                        else:
                            gradients_accumulated[name] = grad.clone()

            self.model.zero_grad()  # Clear gradients after processing each batch

            if count >= self.n_batch_acc:
                break

        # Averaging the accumulated gradients
        for name in gradients_accumulated:
            gradients_accumulated[name] /= self.n_batch_acc

        self.model.to("cpu")
        return gradients_accumulated

    def compute_taylor_importance(self, gradients_accumulated):
        # Compute importance scores based on the averaged gradients
        importance_scores = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if name in gradients_accumulated.keys():
                    importance = module.weight_orig.data.abs() * gradients_accumulated[
                        name
                    ].to(module.weight_orig.data.device)
                    importance_scores[name] = importance.view(-1).abs()
        return importance_scores

    def prune_by_taylor_scores(self, importance_scores):
        for name, module in self.model.named_modules():
            if name in importance_scores:
                num_weights_to_prune = int(
                    self.current_amount * module.weight_orig.data.numel()
                )
                _, weights_to_prune = torch.topk(
                    importance_scores[name], num_weights_to_prune, largest=False
                )

                mask = torch.ones(
                    module.weight_orig.data.numel(),
                    dtype=torch.bool,
                    device=module.weight_orig.device,
                )
                mask[weights_to_prune] = False
                prune.custom_from_mask(
                    module, name="weight", mask=mask.reshape(module.weight.shape)
                )
                module.weight.data[~mask.reshape(module.weight.shape)] = 0.0

    def step(self, val_loader=None):
        # Increase pruning threshold, compute gradients, compute importance, prune
        if self.current_amount is None:
            self.current_amount = self.prune_step
        else:
            self.current_amount += self.prune_step

        if self.current_amount > 1.0 - self.prune_step:
            self.current_amount = 1.0 - self.prune_step

        val_loader = self.datamodule.train_dataloader()
        gradients_accumulated = self.compute_validation_gradients(val_loader)
        importance_scores = self.compute_taylor_importance(gradients_accumulated)
        self.prune_by_taylor_scores(importance_scores)
        self.model.zero_grad()


class HessianUnstructuredPruner:
    def __init__(self, model, datamodule, prune_step=0.1, n_batch_acc=3):
        self.model = model
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.custom_from_mask(
                    module,
                    name="weight",
                    mask=torch.ones(module.weight.shape).to(
                        next(self.model.parameters()).device
                    ),
                )

        for param in self.model.parameters():
            param.requires_grad = True

        self.n_batch_acc = n_batch_acc
        self.datamodule = datamodule
        self.prune_step = prune_step
        self.current_amount = None  # Start with no pruning
        self.miner = get_miner("MultiSimilarityMiner")
        self.loss_fn = get_loss("MultiSimilarityLoss")

    def loss_function(self, descriptors, labels):
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
        else:
            loss = self.loss_fn(descriptors, labels)
            if type(loss) == tuple:
                loss, _ = loss
        return loss

    def compute_validation_gradients(self, val_loader):
        self.model.eval()
        self.model.zero_grad()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(dev)

        gradients_squared = {}
        count = 0
        for batch in tqdm(
            val_loader,
            total=self.n_batch_acc,
            desc="Computing Gradients for Hessian Approximation",
        ):
            count += 1
            self.model.zero_grad()
            places, labels = batch
            BS, N, ch, h, w = places.shape
            images = places.view(BS * N, ch, h, w)
            labels = labels.view(-1)
            descriptors = self.model(images.to(dev))
            loss = self.loss_function(descriptors, labels)
            loss.backward(retain_graph=True)

            # Accumulate squared gradients

            for name, module in self.model.named_modules():
                if (
                    isinstance(module, (nn.Conv2d, nn.Linear))
                    and module.weight_orig.grad is not None
                ):
                    grad = module.weight_orig.grad.data
                    if name in gradients_squared:
                        gradients_squared[name] += (grad**2).cpu()
                    else:
                        gradients_squared[name] = (grad**2).cpu()

            if count > self.n_batch_acc:
                break

        # Average the squared gradients over the batches
        for name in gradients_squared:
            gradients_squared[name] /= self.n_batch_acc
        self.model.to("cpu")
        return gradients_squared

    def compute_taylor_importance(self, gradients_squared):
        importance_scores = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if name in gradients_squared.keys():
                    # Second-order importance approximation (Taylor series expansion)
                    importance = (
                        module.weight_orig.data.abs()
                        * module.weight_orig.grad.data.abs()
                        * gradients_squared[name].to(module.weight_orig.data.device)
                    )
                    importance_scores[name] = importance.view(-1).abs()
        return importance_scores

    def prune_by_taylor_scores(self, importance_scores):
        for name, module in self.model.named_modules():
            if name in importance_scores:
                num_weights_to_prune = int(
                    self.current_amount * module.weight_orig.data.numel()
                )
                _, weights_to_prune = torch.topk(
                    importance_scores[name], num_weights_to_prune, largest=False
                )
                mask = torch.ones(
                    module.weight_orig.data.numel(),
                    dtype=torch.bool,
                    device=module.weight_orig.device,
                )
                mask[weights_to_prune] = False
                prune.custom_from_mask(
                    module, name="weight", mask=mask.reshape(module.weight.shape)
                )
                module.weight.data[~mask.reshape(module.weight.shape)] = 0.0

    def step(self, val_loader=None):
        if self.current_amount is None:
            self.current_amount = self.prune_step
        else:
            self.current_amount += self.prune_step
        if self.current_amount > 1.0 - self.prune_step:
            self.current_amount = 1.0 - self.prune_step
        loader = self.datamodule.train_dataloader()
        gradients_squared = self.compute_validation_gradients(loader)
        importance_scores = self.compute_taylor_importance(gradients_squared)
        self.prune_by_taylor_scores(importance_scores)
        self.model.zero_grad()


def get_cities(args):
    if args.finetune == True:
        cities = ["London"]
    else:
        cities = [
            "Bangkok",
            "BuenosAires",
            "LosAngeles",
            "MexicoCity",
            "OSL",  # refers to Oslo
            "Rome",
            "Barcelona",
            "Chicago",
            "Madrid",
            "Miami",
            "Phoenix",
            "TRT",  # refers to Toronto
            "Boston",
            "Lisbon",
            "Medellin",
            "Minneapolis",
            "PRG",  # refers to Prague
            "WashingtonDC",
            "Brussels",
            "London",
            "Melbourne",
            "Osaka",
            "PRS",  # refers to Paris
        ]
    return cities


def pruning_schedule(epoch: int, cumulative=False):
    start = config["train"]["initial_sparsity"]
    end = config["train"]["final_sparsity"]
    max_epochs = config["train"]["max_epochs"]

    if cumulative:
        if epoch == 0:
            return start
        elif epoch >= max_epochs:
            return end
    else:
        if epoch == 0:
            return 0  # No pruning at the very start if not cumulative
        elif epoch >= max_epochs:
            return end

    # Calculate the current epoch's sparsity based on the pruning schedule
    if not cumulative:
        if config["train"]["pruning_schedule"] == "linear":
            # Linear increase in sparsity from start to end
            return start + (end - start) * (epoch / max_epochs)
        elif config["train"]["pruning_schedule"] == "cosine":
            # Cosine annealing schedule
            cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))
            return end + (start - end) * cosine_decay
        elif config["train"]["pruning_schedule"] == "exp":
            # Ensure the decay_rate is negative to have a true decay
            k = -np.log(1 - end) / max_epochs
            curr = 1 - np.exp(-k * epoch)
            return curr
    else:
        if config["train"]["pruning_schedule"] == "linear":
            # Linear increase in sparsity from start to end
            curr = start + (end - start) * (epoch / max_epochs)
            prev = start + (end - start) * ((epoch - 1) / max_epochs)
            return curr - prev
        elif config["train"]["pruning_schedule"] == "cosine":
            # Cosine annealing schedule
            curr = 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))
            prev = 0.5 * (1 + np.cos(np.pi * (epoch - 1) / max_epochs))
            curr = end + (start - end) * curr
            prev = end + (start - end) * prev
            return curr - prev
        elif config["train"]["pruning_schedule"] == "exp":
            k = -np.log(1 - end) / max_epochs
            curr = 1 - np.exp(-k * epoch)
            prev = 1 - np.exp(-k * (epoch - 1))
            return curr - prev
