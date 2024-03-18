import torch.nn.utils.prune as prune
from pytorch_lightning.callbacks import Callback


class IterativePruningCallback(Callback):
    def __init__(self, prune_amount=0.1, prune_interval=10):
        self.prune_amount = prune_amount
        self.prune_interval = prune_interval

    def on_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if (current_epoch + 1) % self.prune_interval == 0:
            for name, module in pl_module.named_modules():
                # Assuming we're pruning Conv2d and Linear layers
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    prune.l1_unstructured(
                        module, name="weight", amount=self.prune_amount
                    )
            print(
                f"Pruned {self.prune_amount*100}% of weights at epoch {current_epoch+1}."
            )
