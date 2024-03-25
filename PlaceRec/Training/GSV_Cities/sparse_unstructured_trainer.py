from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, ModelPruning
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch.optim.optimizer import Optimizer
from utils import get_method

import PlaceRec.Training.GSV_Cities.utils as utils
from PlaceRec.Training.GSV_Cities.dataloaders.GSVCitiesDataloader import \
    GSVCitiesDataModule
from PlaceRec.Training.GSV_Cities.sparse_utils import (GlobalL1PruningCallback,
                                                       SaveLastModelCallback)


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.
    """

    def __init__(
        self,
        # ---- VPR Method to Train
        method,
        # ---- Train hyperparameters
        lr=0.05,
        optimizer="sgd",
        weight_decay=1e-3,
        momentum=0.9,
        warmup_steps=500,
        milestones=[5, 10, 15],
        lr_mult=0.3,
        # ----- Loss
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

        # self.save_hyperparameters()  # write hyperparams into a file

        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = (
            []
        )  # we will keep track of the % of trivial pairs/triplets at the loss level

        self.faiss_gpu = faiss_gpu

        # ----------------------------------
        # get the backbone and the aggregator
        self.model = method.model
        self.model.train()
        assert isinstance(self.model, torch.nn.Module)

    # the forward pass of the lightning model
    def forward(self, x):
        x = self.model(x)
        return x

    # configure the optimizer
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
            "interval": "step",  # Step-wise scheduler
        }
        return [optimizer], [warmup_scheduler, scheduler]

    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)

            # calculate the % of trivial pairs/triplets
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)

        else:  # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple:
                # somes losses do the online mining inside (they don't need a miner objet),
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class,
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log(
            "b_acc",
            sum(self.batch_acc) / len(self.batch_acc),
            prog_bar=True,
            logger=True,
        )
        return loss

    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels = batch

        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape

        # reshape places and labels
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)
        # Feed forward the batch to the model
        descriptors = self(
            images
        )  # Here we are calling the method forward that we defined above
        loss = self.loss_function(
            descriptors, labels
        )  # Call the loss_function we defined above

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
        print("======> Sparsity: ", self.calculate_sparsity_with_masks(self.model))
        self.log(
            "sparsity", self.calculate_sparsity_with_masks(self.model), on_epoch=True
        )

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
        # calculate descriptors
        descriptors = self(places).detach().cpu()
        if len(self.trainer.datamodule.val_set_names) == 1:
            self.val_step_outputs.append(descriptors)
        else:
            self.val_step_outputs[dataloader_idx].append(descriptors)
        return descriptors

    @staticmethod
    def calculate_sparsity_with_masks(model):
        total_params = 0
        zero_params = 0
        for module in model.modules():
            if hasattr(module, "weight"):  # Check if the module has a weight parameter
                total_params += module.weight.numel()
                # Check for a pruning mask
                mask = None
                for hook in module._forward_pre_hooks.values():
                    if hasattr(hook, "mask"):
                        mask = hook.mask
                        break
                if mask is not None:
                    # If there's a mask, use it to count zero parameters
                    zero_params += torch.sum(mask == 0).item()
                else:
                    # If no mask, just count the zeros in the weights directly
                    zero_params += torch.sum(module.weight == 0).item()
        sparsity = (zero_params / total_params) * 100
        return sparsity

    def on_validation_epoch_end(self) -> None:
        """
        Hook called at the end of a validation epoch to compute and log validation metrics.
        """
        val_step_outputs = self.val_step_outputs
        self.val_step_outputs = []
        dm = self.trainer.datamodule
        if len(dm.val_datasets) == 1:  # we need to put the outputs in a list
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


# =================================== Training Loop ================================
def unstructured_sparse_trainer(args):
    method = get_method(args.method, True)

    pl.seed_everything(seed=1, workers=True)
    torch.set_float32_matmul_precision("medium")

    datamodule = GSVCitiesDataModule(
        batch_size=int(args.batch_size / 4),
        img_per_place=4,
        min_img_per_place=4,
        # cities=['London', 'Boston', 'Melbourne'], # you can sppecify cities here or in GSVCitiesDataloader.py
        shuffle_all=False,  # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=True,
        image_size=args.image_resolution,
        num_workers=16,
        show_data_stats=False,
        val_set_names=["pitts30k_val"],  # pitts30k_val
    )

    model = VPRModel(
        method=method,
        lr=0.0001,  # 0.03 for sgd
        optimizer="adam",  # sgd, adam or adamw
        weight_decay=0,  # 0.001 for sgd or 0.0 for adam
        momentum=0.9,
        warmup_steps=50,
        milestones=[1000],
        lr_mult=0.3,
        # ---------------------------------
        # ---- Training loss function -----
        loss_name="MultiSimilarityLoss",
        miner_name="MultiSimilarityMiner",  # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False,
    )

    # Pruning callback
    checkpoint_cb = ModelCheckpoint(
        dirpath=f"Checkpoints/gsv_cities_sparse_unstructured/{method.name}/",
        monitor="pitts30k_val/R1",
        filename=f"{method.name}"
        + "_epoch[{epoch:02d}]_step[{step:04d}]_R1[{pitts30k_val/R1:.4f}]_sparsity[{sparsity:.2f}]",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=-1,
        mode="max",
    )

    def prune_every_10_epochs(epoch):
        return epoch > 0 and (epoch + 1) % 10 == 0

    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            parameters_to_prune.append((module, "weight"))

    # Define the model pruning callback
    pruning_cb = ModelPruning(
        parameters_to_prune=parameters_to_prune,
        pruning_fn="l1_unstructured",  # Type of pruning
        amount=0.15,  # Pruning amount
        use_global_unstructured=True,  # Apply pruning globally across parameters
        verbose=True,
        make_pruning_permanent=True,  # Make pruning permanent during training
        apply_pruning=prune_every_10_epochs,
        prune_on_train_epoch_end=False,
    )

    # we instantiate a trainer
    trainer = pl.Trainer(
        logger=True,
        enable_progress_bar=False,
        accelerator="gpu",
        devices=[0],
        default_root_dir=f"./LOGS/{method.name}",  # Tensorflow can be used to viz
        num_sanity_val_steps=0,  # runs N validation steps before stating traininclg
        precision="16-mixed",  # we use half precision to reduce  memory usage (and 2x speed on RTX)
        max_epochs=300,
        callbacks=[
            pruning_cb,
            checkpoint_cb,
        ],  # we run the checkpointing callback (you can add more)
        enable_checkpointing=True,
        reload_dataloaders_every_n_epochs=10,  # we reload the dataset to shuffle the order
        log_every_n_steps=5,
        check_val_every_n_epoch=10,
        # limit_train_batches=10
        # fast_dev_run=True # comment if you want to start training the network and saving checkpoints
    )

    trainer.fit(model=model, datamodule=datamodule)
