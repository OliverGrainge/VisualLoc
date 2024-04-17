import os

import pytorch_lightning as pl
import torch
import torch.nn.utils.prune as prune
import torch_pruning as tp
from pytorch_lightning.callbacks import Callback
import torch.nn as nn
from tqdm import tqdm


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
    def __init__(self, model, prune_step=0.1):
        self.model = model
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.custom_from_mask(
                    module, name="weight", mask=torch.ones(module.weight.shape)
                )

        self.prune_step = prune_step
        self.current_amount = None  # Start with no pruning

    def compute_validation_gradients(self, val_loader, loss_function):
        # Ensure model is in evaluation mode to maintain correctness of things like dropout, batchnorm, etc.
        self.model.eval()
        self.model.zero_grad()
        with torch.set_grad_enabled(True):
            for batch in tqdm(self.val_loader, desc="Taylor Importance Gradients"):
                places, labels = batch
                BS, N, ch, h, w = places.shape
                images = places.view(BS * N, ch, h, w)
                labels = labels.view(-1)
                descriptors = self.model(images)
                loss = loss_function(descriptors, labels)
                loss.backward()

    def compute_taylor_importance(self):
        # Compute the product of gradients and weights for each parameter
        importance_scores = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):

                # Ensure gradients are not None
                if module.weight_orig.grad is not None:
                    # Importance score based on Taylor expansion
                    importance = module.weight_orig.data * module.weight_orig.grad.data
                    # Store the importance score flattened (since unstructured)
                    importance_scores[name] = importance.view(-1).abs()
        return importance_scores

    def prune_by_taylor_scores(self, importance_scores):
        for name, module in self.model.named_modules():
            if name in importance_scores:
                # Calculate the number of weights to prune at this step
                num_weights_to_prune = int(
                    self.current_amount * module.weight.data.numel()
                )
                # Get indices of the least important weights
                _, weights_to_prune = torch.topk(
                    importance_scores[name], num_weights_to_prune, largest=False
                )
                # Pruning mask creation and application
                # print(importance_scores[name].shape, module.weight.numel(), module.weight_orig.numel())
                mask = torch.ones(
                    module.weight_orig.data.numel(),
                    dtype=torch.bool,
                    device=module.weight_orig.device,
                )
                mask[weights_to_prune] = False
                prune.custom_from_mask(
                    module, name="weight", mask=mask.reshape(module.weight.shape)
                )
                # module.weight_orig.retain_grad()

    def step(self, val_loader=None, loss_function=None):
        # Increase the pruning threshold
        if self.current_amount is None:
            self.current_amount = self.prune_step
        else:
            self.current_amount += self.prune_step
        # print(self.current_amount, self.prune_step)

        if self.current_amount > 1.0:
            self.current_amount = 1.0
        # Compute the Taylor importance scores
        grad_enables = []
        for param in self.model.parameters():
            grad_enables.append(param.requires_grad)
            param.requires_grad = False

        if val_loader is not None and loss_function is not None:
            self.compute_validation_gradients(val_loader, loss_function)
        importance_scores = self.compute_taylor_importance()
        self.prune_by_taylor_scores(importance_scores)
        self.model.zero_grad()
        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = grad_enables[i]


class HessianUnstructuredPruner:
    def __init__(self, model, val_loader, prune_step=0.1):
        self.model = model
        self.prune_step = prune_step
        self.current_amount = 0.0  # Start with no pruning
        self.val_loader = val_loader

    def compute_validation_gradients(self, val_loader, loss_function):
        # Ensure model is in evaluation mode
        self.model.eval()
        self.model.zero_grad()
        with torch.set_grad_enabled(True):
            for batch in tqdm(val_loader, desc="Hessian Importance Gradients"):
                places, labels = batch
                BS, N, ch, h, w = places.shape
                images = places.view(BS * N, ch, h, w)
                labels = labels.view(-1)
                descriptors = self.model(images)
                loss = loss_function(descriptors, labels)
                loss.backward(
                    create_graph=True
                )  # Enable computation graph for second derivatives

    def compute_hessian_importance(self):
        importance_scores = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if module.weight.grad is not None:
                    # Compute Hessian by taking the gradient of gradients
                    grad2 = torch.autograd.grad(
                        module.weight.grad.flatten().sum(),
                        module.weight,
                        retain_graph=True,
                    )[0]
                    # Use the diagonal of the Hessian as importance scores
                    importance = grad2.pow(2)  # Squaring to focus on magnitude
                    importance_scores[name] = importance.view(-1).abs()
        return importance_scores

    def prune_by_importance_scores(self, importance_scores):
        for name, module in self.model.named_modules():
            if name in importance_scores:
                total_weights = importance_scores[name].numel()
                num_weights_to_prune = int(self.current_amount * total_weights)
                _, weights_to_prune = torch.topk(
                    importance_scores[name], num_weights_to_prune, largest=False
                )
                mask = torch.ones(
                    total_weights, dtype=torch.bool, device=module.weight.device
                )
                mask[weights_to_prune] = False
                prune.custom_from_mask(
                    module, name="weight", mask=mask.reshape(module.weight.shape)
                )

    def step(self, val_loader, loss_function):
        self.current_amount += self.prune_step
        if self.current_amount > 1.0:
            self.current_amount = 1.0
        self.compute_validation_gradients(val_loader, loss_function)
        importance_scores = self.compute_hessian_importance()
        self.prune_by_importance_scores(importance_scores)
        self.model.zero_grad()
