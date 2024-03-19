import torch
import torch.nn.utils.prune as prune
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
        print("================================== permanently pruning")
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
