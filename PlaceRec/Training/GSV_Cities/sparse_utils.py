import os

import pytorch_lightning as pl
import torch
import torch.nn.utils.prune as prune
import torch_pruning as tp
from pytorch_lightning.callbacks import Callback


class GlobalL1PruningCallback(Callback):
    def __init__(self, prune_amount=0.1):
        super().__init__()
        self.prune_amount = prune_amount

    def on_train_start(self, trainer, pl_module):
        parameters_to_prune = []
        for module in pl_module.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                parameters_to_prune.append((module, "weight"))

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.prune_amount,
        )

    def on_train_end(self, trainer, pl_module):
        for module in pl_module.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                prune.remove(module, "weight")


class SaveLastModelCallback(Callback):
    def __init__(self, dirpath, filename="last-model.ckpt"):
        self.dirpath = dirpath
        self.filename = filename

    def on_train_end(self, trainer, pl_module):
        file_path = f"{self.dirpath}/{self.filename}"
        trainer.save_checkpoint(file_path)
        print(f"Model saved to {file_path}")


class SaveFullModelCallback(pl.Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path

    def on_validation_end(self, trainer, pl_module):
        _, nparams = tp.utils.count_ops_and_params(
            pl_module.model, pl_module.example_img.cuda()
        )
        sparsity = nparams / pl_module.orig_nparams
        filepath = f"{self.save_path}/{pl_module.name}/{pl_module.name}_R1[{pl_module.val_R1:.03f}]_SPARSITY[{sparsity:.03f}].ckpt"

        if not os.path.exists(f"{self.save_path}/{pl_module.name}"):
            os.makedirs(f"{self.save_path}/{pl_module.name}")
        # Save the entire model
        torch.save(pl_module.model, filepath)
        print(f"Full model saved to {filepath}")


def calculate_sparsity(model):
    total_weights = 0
    zero_weights = 0

    # Iterate over all parameters in the model
    for param in model.parameters():
        # Flatten the parameter tensor and convert it to a numpy array
        param_copy = param.clone().detach().cpu()
        param_data = param_copy.data.view(-1).numpy()

        # Update total and zero-valued weights counts
        total_weights += param_data.size
        zero_weights += (param_data == 0).sum()

    # Calculate sparsity
    sparsity = zero_weights / total_weights
    return sparsity
