import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch_pruning as tp
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm

from PlaceRec.Training.GSV_Cities.utils import get_loss, get_miner


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
                    module, name="weight", mask=torch.ones(module.weight.shape)
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
        # Ensure model is in evaluation mode to maintain correctness of things like dropout, batchnorm, etc.
        self.model.eval()
        self.model.zero_grad()
        if torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"

        self.model.to(dev)
        count = 0
        for batch in tqdm(
            val_loader, total=self.n_batch_acc, desc="Taylor Importance Gradients"
        ):
            count += 1
            places, labels = batch
            BS, N, ch, h, w = places.shape
            images = places.view(BS * N, ch, h, w)
            labels = labels.view(-1)
            descriptors = self.model(images.to(dev))
            loss = self.loss_function(descriptors, labels)
            loss.backward()
            if count > self.n_batch_acc:
                break
        self.model.to("cpu")

    def compute_taylor_importance(self):
        # Compute the product of gradients and weights for each parameter
        importance_scores = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Ensure gradients are not None
                if module.weight_orig.grad is not None:
                    # Importance score based on Taylor expansion: only consider active weights as already pruned
                    # weights are not important
                    importance = (
                        module.weight_orig.data * module.weight_mask.data
                    ) * module.weight_orig.grad.data
                    # Store the importance score flattened (since unstructured)
                    importance_scores[name] = importance.view(-1).abs()
        return importance_scores

    def prune_by_taylor_scores(self, importance_scores):
        for name, module in self.model.named_modules():
            if name in importance_scores:
                # Calculate the number of weights to prune at this step
                num_weights_to_prune = int(
                    self.current_amount * module.weight_orig.data.numel()
                )
                # Get indices of the least important weights
                _, weights_to_prune = torch.topk(
                    importance_scores[name], num_weights_to_prune, largest=False
                )
                # Pruning mask creation and application
                mask = torch.ones(
                    module.weight_orig.data.numel(),
                    dtype=torch.bool,
                    device=module.weight_orig.device,
                ).to(next(self.model.parameters()).device)

                mask[weights_to_prune] = False
                prune.custom_from_mask(
                    module, name="weight", mask=mask.reshape(module.weight.shape)
                )
                module.weight.data[~mask.reshape(module.weight.shape)] = 0.0

    def step(self, val_loader=None):
        # Increase the pruning threshold
        if self.current_amount is None:
            self.current_amount = self.prune_step
        else:
            self.current_amount += self.prune_step

        if self.current_amount > 1.0 - self.prune_step:
            self.current_amount = 1.0 - self.prune_step

        # Compute the Taylor importance scores
        # need gradients for all layers to prune all layers
        val_loader = self.datamodule.train_dataloader()
        self.compute_validation_gradients(val_loader)
        importance_scores = self.compute_taylor_importance()
        self.prune_by_taylor_scores(importance_scores)
        self.model.zero_grad()


class HessianUnstructuredPruner:
    def __init__(self, model, datamodule, prune_step=0.1, n_batch_acc=3):
        self.model = model
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.custom_from_mask(
                    module, name="weight", mask=torch.ones(module.weight.shape)
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
            places, labels = batch
            BS, N, ch, h, w = places.shape
            images = places.view(BS * N, ch, h, w)
            labels = labels.view(-1)
            descriptors = self.model(images.to(dev))
            loss = self.loss_function(descriptors, labels)
            loss.backward()

            # Accumulate squared gradients
            for name, module in self.model.named_modules():
                if (
                    isinstance(module, (nn.Conv2d, nn.Linear))
                    and module.weight_orig.grad is not None
                ):
                    grad = module.weight_orig.grad.data
                    if name in gradients_squared:
                        gradients_squared[name] += grad**2
                    else:
                        gradients_squared[name] = grad**2
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
                if module.weight_orig.grad is not None:
                    # Second-order importance approximation (Taylor series expansion)
                    importance = (
                        module.weight_orig.data.abs()
                        * module.weight_orig.grad.data.abs()
                        * gradients_squared[name]
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
        val_loader = self.datamodule.train_dataloader()
        gradients_squared = self.compute_validation_gradients(val_loader)
        importance_scores = self.compute_taylor_importance(gradients_squared)
        self.prune_by_taylor_scores(importance_scores)
        self.model.zero_grad()
