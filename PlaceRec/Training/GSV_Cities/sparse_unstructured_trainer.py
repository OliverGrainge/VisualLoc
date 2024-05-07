from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch.optim.optimizer import Optimizer

import PlaceRec.Training.GSV_Cities.utils as utils
from PlaceRec.Methods.resnet50_gem import Resnet50gemModel
from PlaceRec.Training.GSV_Cities.dataloaders.GSVCitiesDataloader import (
    GSVCitiesDataModule,
)
from PlaceRec.Training.GSV_Cities.sparse_utils import (
    L1UnstructuredPruner,
    TaylorUnstructuredPruner,
    HessianUnstructuredPruner,
    get_cities,
)
from PlaceRec.utils import get_method


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.
    """

    def __init__(
        self,
        method,
        lr=0.05,
        optimizer="sgd",
        weight_decay=1e-3,
        momentum=0.9,
        warmup_steps=500,
        milestones=[5, 10, 15],
        lr_mult=0.3,
        loss_name="MultiSimilarityLoss",
        miner_name="MultiSimilarityMiner",
        miner_margin=0.1,
        faiss_gpu=False,
    ):
        super().__init__()
        self.name = method.name

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin

        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = []

        self.faiss_gpu = faiss_gpu
        self.model = method.model
        self.model.train()
        assert isinstance(self.model, torch.nn.Module)

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        if self.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        elif self.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == "adam":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(
                f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"'
            )
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.lr_mult
        )
        warmup_scheduler = {
            "scheduler": LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: min(1.0, (epoch + 1) / self.warmup_steps),
            ),
            "interval": "step",
        }
        return [optimizer], [warmup_scheduler, scheduler]

    def loss_function(self, descriptors, labels):
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)

            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)

        else:
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple:
                loss, batch_acc = loss

        self.batch_acc.append(batch_acc)
        self.log(
            "b_acc",
            sum(self.batch_acc) / len(self.batch_acc),
            prog_bar=True,
            logger=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        places, labels = batch
        BS, N, ch, h, w = places.shape

        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)
        descriptors = self(images)
        loss = self.loss_function(descriptors, labels)

        self.log("loss", loss.item(), logger=True)
        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        """
        Hook called at the end of a training epoch to reset or update certain parameters.
        """
        self.batch_acc = []

    def on_validation_epoch_start(self) -> None:
        """
        Hook called at the start of a validation epoch to initialize or reset parameters.
        """
        if len(self.trainer.datamodule.val_set_names) == 1:
            self.val_step_outputs = []
        else:
            self.val_step_outputs = [
                [] for _ in range(len(self.trainer.datamodule.val_set_names))
            ]

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, Optional[torch.Tensor]],
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Processes a single batch of data during the validation phase.

        Args:
            batch (tuple): A tuple containing input data and optionally labels.
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader (used when multiple validation dataloaders are present).

        Returns:
            torch.Tensor: The descriptor vectors computed for the batch.
        """
        places, _ = batch
        descriptors = self(places).detach().cpu()
        if len(self.trainer.datamodule.val_set_names) == 1:
            self.val_step_outputs.append(descriptors)
        else:
            self.val_step_outputs[dataloader_idx].append(descriptors)
        return descriptors

    def on_validation_epoch_end(self) -> None:
        """
        Hook called at the end of a validation epoch to compute and log validation metrics.
        """
        val_step_outputs = self.val_step_outputs
        self.val_step_outputs = []
        dm = self.trainer.datamodule
        if len(dm.val_datasets) == 1:
            val_step_outputs = [val_step_outputs]

        for i, (val_set_name, val_dataset) in enumerate(
            zip(dm.val_set_names, dm.val_datasets)
        ):
            feats = torch.concat(val_step_outputs[i], dim=0)
            num_references = val_dataset.num_references
            num_queries = val_dataset.num_queries
            ground_truth = val_dataset.ground_truth
            r_list = feats[:num_references]
            q_list = feats[num_references:]

            recalls_dict, predictions = utils.get_validation_recalls(
                r_list=r_list,
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 25],
                gt=ground_truth,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu,
            )
            del r_list, q_list, feats, num_references, ground_truth
            self.log(f"{val_set_name}/R1", recalls_dict[1], prog_bar=False, logger=True)
            self.log(f"{val_set_name}/R5", recalls_dict[5], prog_bar=False, logger=True)
            self.log(
                f"{val_set_name}/R10", recalls_dict[10], prog_bar=False, logger=True
            )
        print("\n\n")


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


def setup_pruner(method, datamodule, args, prune_step=0.05, n_batch_acc=100):
    if args.pruning_type == None:
        raise Exception("Must choose a pruning type for unstructured sparse trainer")
    elif args.pruning_type == "magnitude":
        return L1UnstructuredPruner(method.model, prune_step=prune_step)
    elif args.pruning_type == "first-order":
        datamodule.setup("fit")
        return TaylorUnstructuredPruner(
            method.model, datamodule, prune_step=prune_step, n_batch_acc=n_batch_acc
        )
    elif args.pruning_type == "second-order":
        datamodule.setup("fit")
        return HessianUnstructuredPruner(
            method.model, datamodule, prune_step=prune_step, n_batch_acc=n_batch_acc
        )
    else:
        raise Exception(f"Pruning method {args.pruning_type} Not implemented")


def sparse_unstructured_trainer(args):
    pl.seed_everything(seed=1, workers=True)
    torch.set_float32_matmul_precision("medium")

    datamodule = GSVCitiesDataModule(
        cities=get_cities(),
        batch_size=int(args.batch_size / 16),
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False,
        random_sample_from_each_place=True,
        image_size=args.image_resolution,
        num_workers=args.num_workers,
        show_data_stats=False,
        val_set_names=["pitts30k_val"],
    )

    method = get_method(args.method, True)
    pruner = setup_pruner(method, datamodule, args, prune_step=0.05)

    for training_round in range(20):
        print(
            "==============================================================================="
        )
        print(
            "==============================================================================="
        )
        print(
            "==============================================================================="
        )
        print(
            f"=================    Training Round: {training_round}: Sparsity {sparsity(method)} ========================="
        )
        print(
            "==============================================================================="
        )
        print(
            "==============================================================================="
        )
        print(
            "==============================================================================="
        )
        model = VPRModel(
            method=method,
            lr=0.0001,
            weight_decay=0,
            momentum=0.9,
            warmup_steps=600,
            milestones=[5, 10, 15, 25],
            lr_mult=0.3,
            loss_name="MultiSimilarityLoss",
            miner_name="MultiSimilarityMiner",
            miner_margin=0.1,
            faiss_gpu=False,
            optimizer=args.optimizer,
        )

        sparse_count = sparsity(method)

        checkpoint_cb = ModelCheckpoint(
            dirpath=f"Checkpoints/gsv_cities_sparse_unstructured/{method.name}/{args.pruning_type}/",
            filename=f"{method.name}"
            + "_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_sparsity["
            + f"{sparse_count:.2f}"
            + "]",
            auto_insert_metric_name=False,
            save_weights_only=True,
            save_top_k=1,
            mode="max",
        )

        earlystopping_cb = EarlyStopping(
            monitor="pitts30k_val/R1",
            min_delta=0.00,
            patience=3,
            verbose=False,
            mode="max",
        )

        trainer = pl.Trainer(
            enable_progress_bar=False,
            devices="auto",
            accelerator="auto",
            strategy="auto",
            default_root_dir=f"./LOGS/{method.name}",
            num_sanity_val_steps=0,
            precision="16-mixed",
            max_epochs=1,
            check_val_every_n_epoch=1,
            callbacks=[checkpoint_cb, earlystopping_cb],
            reload_dataloaders_every_n_epochs=1,
            # limit_train_batches=10,
        )

        trainer.fit(model=model, datamodule=datamodule)

        pruner.step()
