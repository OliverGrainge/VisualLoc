from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch.optim.optimizer import Optimizer
from pytorch_lightning.loggers import WandbLogger

import PlaceRec.Training.GSV_Cities.utils as utils
from PlaceRec.Training.GSV_Cities.dataloaders.GSVCitiesDataloader import (
    GSVCitiesDataModuleDistillation,
)
from PlaceRec.Training.GSV_Cities.sparse_utils import get_cities
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


class HardDarkRank(nn.Module):
    def __init__(self, alpha=3, beta=3, permute_len=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.permute_len = permute_len

    def forward(self, student, teacher):
        score_teacher = -1 * self.alpha * pdist(teacher, squared=False).pow(self.beta)
        score_student = -1 * self.alpha * pdist(student, squared=False).pow(self.beta)

        permute_idx = score_teacher.sort(dim=1, descending=True)[1][
            :, 1 : (self.permute_len + 1)
        ]
        ordered_student = torch.gather(score_student, 1, permute_idx)

        log_prob = (
            ordered_student
            - torch.stack(
                [
                    torch.logsumexp(ordered_student[:, i:], dim=1)
                    for i in range(permute_idx.size(1))
                ],
                dim=1,
            )
        ).sum(dim=1)
        loss = (-1 * log_prob).mean()

        return loss


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


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.
    """

    def __init__(
        self,
        teacher_method,
        student_method,
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
        rkd_angle_factor=1.0,
        rkd_distance_factor=0.0,
        darkrank_factor=0.0,
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
        self.rkd_angle_factor = rkd_angle_factor
        self.rkd_distance_factor = rkd_distance_factor
        self.darkrank_factor = darkrank_factor

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin

        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = []

        self.faiss_gpu = faiss_gpu
        self.student_model = student_method.model
        self.teacher_model = teacher_method.model
        self.student_model.train()
        self.teacher_model.eval()
        self.rkd_distance_loss_fn = RkdDistance()
        self.rkd_angle_loss_fn = RKdAngle()
        self.darkrank_loss_fn = HardDarkRank()

        self.feature_adapter = nn.Linear(
            student_method.features_dim["global_feature_shape"][0],
            teacher_method.features_dim["global_feature_shape"][0],
        )
        assert isinstance(self.student_model, torch.nn.Module)

    def forward(self, x):
        x = self.student_model(x)
        return x

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:

        model_parameters = list(self.student_model.parameters()) + list(
            self.feature_adapter.parameters()
        )

        if self.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                model_parameters,
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        elif self.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                model_parameters, lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == "adam":
            optimizer = torch.optim.AdamW(
                model_parameters, lr=self.lr, weight_decay=self.weight_decay
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
        return sum(self.batch_acc) / len(self.batch_acc)

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
        darkrank_loss = self.darkrank_loss_fn(student_descriptors, teacher_descriptors)

        cont_loss = self.loss_function(student_descriptors, labels)

        loss = (
            self.contrastive_factor * cont_loss
            + self.rkd_distance_factor * rkd_distance_loss
            + self.rkd * rkd_angle_loss
            + self.darkrank_factor * darkrank_loss
        )

        self.log("rkd_distance_loss", rkd_distance_loss)
        self.log("rkd_angle_loss", rkd_angle_loss)
        self.log("darkrank_loss", darkrank_loss)
        self.log("contrastive_loss", cont_loss)
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


def distillation_trainer(args):
    pl.seed_everything(seed=1, workers=True)
    torch.set_float32_matmul_precision("medium")

    student_method = get_method(args.method, False)
    teacher_method = get_method(args.teacher_method, True)

    wandb_logger = WandbLogger(project="GSVCities", config=config)

    datamodule = GSVCitiesDataModuleDistillation(
        cities=get_cities(args.finetune),
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

    model = VPRModel(
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
        contrastive_factor=args.contrastive_factor,
        rkd_angle_factor=args.rkd_angle_factor,
        rkd_distance_factor=args.rkd_distance_factor,
        darkrank_factor=args.darkrank_factor,
    )

    if args.checkpoint:
        checkpoint_cb = ModelCheckpoint(
            dirpath="Checkpoints/gsv_cities_rdk/"
            + student_method.name
            + "_"
            + teacher_method.name
            + "/",
            monitor="pitts30k_val/R1",
            filename=f"{student_method.name}"
            + "_epoch({epoch:02d})_step({step:04d})_R1[{pitts30k_val/R1:.4f}]_R5[{pitts30k_val/R5:.4f}]",
            auto_insert_metric_name=False,
            save_weights_only=True,
            save_top_k=1,
            mode="max",
        )

    if args.debug:
        trainer = pl.Trainer(
            enable_progress_bar=args.enable_progress_bar,
            devices="auto",
            accelerator="auto",
            strategy="auto",
            default_root_dir=f"./LOGS/{student_method.name}_{teacher_method.name}",
            num_sanity_val_steps=0,
            precision="16-mixed",
            max_epochs=args.max_epochs,
            callbacks=[checkpoint_cb] if args.checkpoint else [],
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
            default_root_dir=f"./LOGS/{student_method.name}_{teacher_method.name}",
            num_sanity_val_steps=0,
            precision="16-mixed",
            max_epochs=args.max_epochs,
            callbacks=[checkpoint_cb] if args.checkpoint else [],
            reload_dataloaders_every_n_epochs=1,
            log_every_n_steps=20,
            logger=wandb_logger,
        )

    trainer.fit(model=model, datamodule=datamodule)
