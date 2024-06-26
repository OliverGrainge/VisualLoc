from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch_pruning as tp
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelPruning
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
import torch.nn as nn

import PlaceRec.Training.GSV_Cities.utils as utils
from PlaceRec.Training.GSV_Cities.dataloaders.GSVCitiesDataloader import (
    GSVCitiesDataModuleDistillation,
)
from PlaceRec.Training.GSV_Cities.sparse_utils import get_cities, pruning_schedule
from PlaceRec.utils import get_config, get_method

config = get_config()


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RkdDistance(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction="mean")
        return loss


class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            td = teacher.unsqueeze(0) - teacher.unsqueeze(1)
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = student.unsqueeze(0) - student.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction="elementwise_mean")
        return loss


def get_scheduler(args):
    pruning_freq = args.pruning_freq

    def schd(pruning_ratio_dict, steps):
        return [
            pruning_schedule(i * pruning_freq, cumulative=False) * pruning_ratio_dict
            for i in range(steps + 1)
        ]

    return schd


def setup_pruner(method, args):
    example_img = method.example_input().to(method.device)
    orig_macs, orig_nparams = tp.utils.count_ops_and_params(method.model, example_img)
    print(
        "===============> macs: ",
        orig_macs / 1e6,
        "    nparams",
        orig_nparams,
    )

    dont_prune = []
    for name, layer in method.model.named_modules():
        if (
            "attention" in name
            or "attn" in name
            or "aggregator" in name
            or "aggregation" in name
            or "proj" in name
        ):
            dont_prune.append(layer)

    if args.pruning_type is None:
        raise Exception(" For structured pruning, Must choose pruning method.")
    elif args.pruning_type == "magnitude":
        importance = tp.importance.MagnitudeImportance(p=2, group_reduction="mean")
    elif args.pruning_type == "first-order":
        importance = tp.importance.GroupTaylorImportance()
    elif args.pruning_type == "second-order":
        importance = tp.importance.GroupHessianImportance()
    else:
        raise Exception(f"Pruning method {args.pruning_type} is not found")

    pruner = tp.pruner.MagnitudePruner(
        method.model,
        example_img,
        importance,
        iterative_steps=args.max_epochs // args.pruning_freq,
        iterative_pruning_ratio_scheduler=get_scheduler(args),
        pruning_ratio=args.final_sparsity,
        ignored_layers=dont_prune,
        global_pruning=False,
    )

    return method, pruner, orig_nparams


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.
    """

    def __init__(
        self,
        args,
        student_method,
        teacher_method,
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
        contrastive_factor=1.0,
        rkd_distance_factor=1.0,
        rkd_angle_factor=0.0,
        darkrank_factor=0.0,
        pruning_freq=5,
    ):
        super().__init__()

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult
        self.contrastive_factor = contrastive_factor
        self.rkd_distance_factor = rkd_distance_factor
        self.rkd_angle_factor = rkd_angle_factor
        self.darkrank_factor = darkrank_factor
        self.pruning_freq = pruning_freq

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin

        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = []

        self.faiss_gpu = faiss_gpu

        self.student_method, self.pruner, self.orig_nparams = setup_pruner(
            student_method, args
        )
        self.student_model = student_method.model
        self.teacher_model = teacher_method.model
        self.student_model.train()
        self.teacher_model.eval()
        self.student_model.train()
        self.rkd_distance_loss_fn = RkdDistance()
        self.rkd_angle_loss_fn = RKdAngle()
        self.epoch = 0
        self.feature_adapter = nn.Linear(
            student_method.features_dim["global_feature_shape"][0],
            teacher_method.features_dim["global_feature_shape"][0],
        )
        self.save_hyperparameters(args)
        self.hparams.update(
            {
                "feature_size": self.student_method.features_dim[
                    "global_feature_shape"
                ],
                "teacher_feature_size": self.teacher_method.features_dim[
                    "global_feature_shape"
                ],
            }
        )
        assert isinstance(self.student_model, torch.nn.Module)

    def forward(self, x):
        x = self.student_model(x)
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
        teacher_places, student_places, labels = batch
        BS, N, ch, h, w = teacher_places.shape
        teacher_images = teacher_places.view(BS * N, ch, h, w)
        BS, N, ch, h, w = student_places.shape
        student_images = student_places.view(BS * N, ch, h, w)
        labels = labels.view(-1)
        with torch.no_grad():
            teacher_descriptors = self.teacher_model(teacher_images)
        student_descriptors = self.student_model(student_images)
        student_descriptors = self.feature_adapter(student_descriptors)
        rkd_distance_loss = self.rkd_distance_loss_fn(
            student_descriptors, teacher_descriptors
        )
        rkd_angle_loss = self.rkd_angle_loss_fn(
            student_descriptors, teacher_descriptors
        )

        cont_loss = self.loss_function(student_descriptors, labels)

        loss = (
            self.contrastive_factor * cont_loss
            + self.rkd_distance_factor * rkd_distance_loss
            + self.rkd_angle_factor * rkd_angle_loss
        )

        self.log("rkd_distance_loss", rkd_distance_loss)
        self.log("rkd_angle_loss", rkd_angle_loss)
        self.log("contrastive_loss", cont_loss)
        self.log("loss", loss.item(), logger=True)
        return {"loss": loss}

    def on_train_epoch_start(self):
        """
        Completes the pruning step if required
        """
        if self.epoch % self.pruning_freq == 0 and self.epoch != 0:
            self.pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(
            self.student_model,
            self.student_method.example_input().to(
                next(self.student_model.parameters()).device
            ),
        )

        self.log("sparsity", 1 - (nparams / self.orig_nparams))
        self.epoch += 1

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
        # calculate descriptors
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
        self.val_R1 = recalls_dict[1]
        macs, nparams = tp.utils.count_ops_and_params(
            self.student_model,
            self.student_method.example_input().to(
                next(self.student_model.parameters()).device
            ),
        )

        self.log("sparsity", 1 - (nparams / self.orig_nparams))
        self.log("macs", macs / 1e6)
        self.log("nparams", nparams)


# =================================== Training Loop ================================
def sparse_structured_distillation_trainer(args):
    method = get_method(args.method, False)
    pl.seed_everything(seed=1, workers=True)
    torch.set_float32_matmul_precision("medium")

    student_method = get_method(args.method, False)
    teacher_method = get_method(args.teacher_method, True)

    wandb_logger = WandbLogger(project="GSVCities", config=config["train"])

    datamodule = GSVCitiesDataModuleDistillation(
        cities=get_cities(args),
        batch_size=int(args.batch_size / 4),
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False,
        random_sample_from_each_place=True,
        student_image_size=args.image_resolution,
        teacher_image_size=args.teacher_resolution,
        num_workers=args.num_workers,
        show_data_stats=False,
        val_set_names=["pitts30k_val"],
    )

    if args.checkpoint:
        checkpoint_cb = ModelCheckpoint(
            dirpath=f"Checkpoints/gsv_cities_sparse_structured/{method.name}/{args.pruning_type}/",
            filename=f"{student_method.name}"
            + "_epoch_{epoch:02d}_step_{step:04d}_R1_{pitts30k_val/R1:.4f}_sparsity_"
            + f"_{sparsity:.2f}",
            auto_insert_metric_name=False,
            save_weights_only=True,
            every_n_epochs=args.pruning_freq,
        )

    module = VPRModel(
        args=args,
        student_method=student_method,
        teacher_method=teacher_method,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        warmup_steps=args.warmup_steps,
        milestones=args.milestones,
        lr_mult=args.lr_mult,
        loss_name=args.loss_name,
        miner_name=args.miner_name,
        miner_margin=args.miner_margin,
        faiss_gpu=False,
        optimizer=args.optimizer,
        rkd_angle_factor=args.rkd_angle_factor,
        rkd_distance_factor=args.rkd_distance_factor,
        pruning_freq=args.pruning_freq,
    )

    if args.debug:
        trainer = pl.Trainer(
            enable_progress_bar=args.enable_progress_bar,
            devices="auto",
            accelerator="auto",
            strategy="auto",
            default_root_dir=f"./LOGS/{method.name}",
            num_sanity_val_steps=0,
            precision="16-mixed",
            max_epochs=args.max_epochs,
            callbacks=[
                checkpoint_cb,
            ]
            if args.checkpoint
            else [],
            reload_dataloaders_every_n_epochs=1,
            logger=wandb_logger,
            log_every_n_steps=1,
            limit_train_batches=2,
            check_val_every_n_epoch=20,
        )
    else:
        trainer = pl.Trainer(
            enable_progress_bar=args.enable_progress_bar,
            devices="auto",
            accelerator="auto",
            strategy="auto",
            default_root_dir=f"./LOGS/{method.name}",
            num_sanity_val_steps=0,
            precision="16-mixed",
            max_epochs=args.max_epochs,
            callbacks=[
                checkpoint_cb,
            ]
            if args.checkpoint
            else [],
            reload_dataloaders_every_n_epochs=1,
            logger=wandb_logger,
        )

    trainer.fit(model=module, datamodule=datamodule)
