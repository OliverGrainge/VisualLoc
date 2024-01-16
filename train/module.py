from argparse import Namespace
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F

from train import utils
from train.dataloaders.GSVCitiesDataloader import (
    GSVCitiesDataModule,
)


class VPRModel(pl.LightningModule):
    """
    A PyTorch Lightning Module for Visual Place Recognition (VPR).

    This model is designed to handle tasks related to identifying and recognizing places based on visual data.
    It utilizes PyTorch Lightning for efficient training and validation with a modular approach.

    Attributes:
        args (Namespace): A configuration object containing parameters such as learning rate, weight decay, etc.
        model (torch.nn.Module): The underlying neural network model for feature extraction or VPR.
        loss_fn (callable): The loss function used for training.
        miner (callable): An optional online mining strategy for hard examples in the dataset.
        batch_acc (list): A list to track the percentage of trivial pairs/triplets at the loss level.
        faiss_gpu (bool): Flag indicating whether to use GPU with FAISS for efficient similarity search.

    Methods:
        forward(x): Defines the forward pass of the model.
        configure_optimizers(): Configures optimizers and learning rate schedulers.
        loss_function(descriptors, labels): Computes the loss based on model outputs and labels.
        training_step(batch, batch_idx): Processes a single batch during training.
        on_train_epoch_end(): Hook called at the end of a training epoch.
        on_validation_epoch_start(): Hook called at the start of a validation epoch.
        validation_step(batch, batch_idx, dataloader_idx=None): Processes a single batch during validation.
        on_validation_epoch_end(): Hook called at the end of a validation epoch.
    """

    def __init__(self, args: Namespace, model: nn.Module):
        """
        Initializes the VPRModel with the specified arguments and model.

        Args:
            args (Namespace): A configuration object containing various training and model parameters.
            model (torch.nn.Module): The neural network model to be used for VPR.
        """
        super().__init__()
        self.args = args
        self.save_hyperparameters()  # write hyperparams into a file
        self.loss_fn = utils.get_loss(args.loss_name)
        self.miner = utils.get_miner(args.miner_name, args.miner_margin)
        self.batch_acc = []  # we will keep track of the % of trivial pairs/triplets at the loss level
        self.faiss_gpu = args.faiss_gpu
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The model's output tensor.
        """
        x = self.model(x)
        return x

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        """
        Configures the model's optimizers and learning rate schedulers.

        Returns:
            tuple: A tuple containing the list of optimizers and the list of learning rate schedulers.
        """
        if self.args.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay, momentum=self.args.momentum
            )
        elif self.args.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        elif self.args.optimizer.lower() == "adam":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones, gamma=self.args.lr_mult)
        warmup_scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, (epoch + 1) / self.args.warmup_steps)),
            "interval": "step",  # Step-wise scheduler
        }
        return [optimizer], [warmup_scheduler, scheduler]

    def loss_function(self, student_descriptors: torch.Tensor, teacher_descriptors: List[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss for a batch of data.

        Args:
            descriptors (torch.Tensor): The descriptor vectors outputted by the model.
            labels (torch.Tensor): Ground truth labels corresponding to the input data.

        Returns:
            torch.Tensor: The computed loss value.
        """
        if self.miner is not None:
            miner_outputs = self.miner(student_descriptors, labels)
            #task_loss = self.loss_fn(student_descriptors, labels, miner_outputs)
            rkd_loss = self.rkd_loss(student_descriptors, teacher_descriptors, labels)
            loss = torch.sum(self.args.rkd_weight * rkd_loss)/len(rkd_loss)
            nb_samples = student_descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)

        else:  # no online mining
            loss = self.loss_fn(student_descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple:
                loss, batch_acc = task_loss

        self.batch_acc.append(batch_acc)
        self.log("b_acc", sum(self.batch_acc) / len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    
    def rkd_loss(self, student_descriptors: torch.Tensor, teacher_descriptors: List[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        if self.miner is not None:
            if self.args.rkd_loss == "pairwise_l2_distance":
                miner_outputs = self.miner(student_descriptors, labels)
                s_desc_i = student_descriptors[miner_outputs[0]]
                s_desc_j = student_descriptors[miner_outputs[1]]
                student_structure = (s_desc_i - s_desc_j) ** 2
                student_structure = torch.sum(student_structure, dim=1)
                student_structure /= s_desc_i.shape[1]
                student_structure *= 1/student_structure.mean()
                rkd_losses = []
                for t_desc in teacher_descriptors:
                    t_desc_i = t_desc[miner_outputs[0]]
                    t_desc_j = t_desc[miner_outputs[1]]
                    teacher_structure = (t_desc_i - t_desc_j) ** 2
                    teacher_structure = torch.sum(teacher_structure, dim=1)
                    teacher_structure /= t_desc_i.shape[1]
                    teacher_structure *= 1/teacher_structure.mean()
                    rkd_losses.append(F.smooth_l1_loss(student_structure, teacher_structure))
                return torch.tensor(rkd_losses.item())
            elif self.args.rkd_loss == "pairwise_cosine_distance":
                miner_outputs = self.miner(student_descriptors, labels)
                s_desc_i = student_descriptors[miner_outputs[0]]
                s_desc_j = student_descriptors[miner_outputs[1]]
                student_similarity_matrix = torch.mm(s_desc_i, s_desc_j.T)
                rkd_losses = []
                for t_desc in teacher_descriptors:
                    t_desc_i = t_desc[miner_outputs[0]]
                    t_desc_j = t_desc[miner_outputs[1]]
                    teacher_similarity_matrix = torch.mm(t_desc_i, t_desc_j.T)
                    rkd_loss = F.smooth_l1_loss(student_similarity_matrix.flatten(), teacher_similarity_matrix.flatten())
                    rkd_losses.append(rkd_loss)
                return torch.tensor(rkd_losses)
            else:
                raise NotImplementedError

                    


            
                

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Processes a single batch of data during the training phase.

        Args:
            batch (tuple): A tuple containing input data and corresponding labels.
            batch_idx (int): Index of the batch.

        Returns:
            dict: A dictionary containing the loss value for the batch.
        """
        places, labels, teacher_desc = batch
        for desc in teacher_desc:
            desc.to("cuda")
        BS, N, ch, h, w = places.shape
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)
        student_desc = self(images)  # Here we are calling the method forward that we defined above
        teacher_desc = [desc.view(BS * N, -1) for desc in teacher_desc]
        print("################################3", student_desc.device, teacher_desc[0].device, labels.device)
        loss = self.loss_function(student_desc, teacher_desc, labels)  # Call the loss_function we defined above
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
            self.val_step_outputs = [[] for _ in range(len(self.trainer.datamodule.val_set_names))]

    def validation_step(
        self, batch: Tuple[torch.Tensor, Optional[torch.Tensor]], batch_idx: int, dataloader_idx: Optional[int] = None
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

        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
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
                faiss_gpu=self.args.faiss_gpu,
            )
            del r_list, q_list, feats, num_references, ground_truth
            self.log(f"{val_set_name}/R1", recalls_dict[1], prog_bar=False, logger=True)
            self.log(f"{val_set_name}/R5", recalls_dict[5], prog_bar=False, logger=True)
            self.log(f"{val_set_name}/R10", recalls_dict[10], prog_bar=False, logger=True)
        print("\n\n")