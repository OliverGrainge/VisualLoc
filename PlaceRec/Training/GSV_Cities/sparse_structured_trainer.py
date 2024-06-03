from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch_pruning as tp
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelPruning
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch.optim.optimizer import Optimizer
from pytorch_lightning.loggers import WandbLogger
import torch.nn as nn

import PlaceRec.Training.GSV_Cities.utils as utils
from PlaceRec.Training.GSV_Cities.dataloaders.GSVCitiesDataloader import (
    GSVCitiesDataModule,
)
from PlaceRec.Training.GSV_Cities.sparse_utils import get_cities, pruning_schedule
from PlaceRec.utils import get_config, get_method

config = get_config()


def get_scheduler(args):
    pruning_freq = args.pruning_freq

    def schd(pruning_ratio_dict, steps):
        return [
            pruning_schedule(i * pruning_freq, cumulative=False) * pruning_ratio_dict
            for i in range(steps + 1)
        ]

    return schd


def get_dont_prune(method, args):
    if "netvlad" in method.name.lower():
        dont_prune = []
        for name, module in method.model.named_modules():
            if "channel_pool" in name:
                dont_prune.append(module)
        return dont_prune
    return []


def get_channel_groups(method, args):
    if "vit" in method.name:
        channel_groups = {}
        for m in method.model.modules():
            if isinstance(m, nn.MultiheadAttention):
                channel_groups[m] = m.num_heads
        return channel_groups
    return []


def get_pruning_ratio_dict(method, args):
    print("==============================", args.aggregation_pruning_rate)
    if "convap" in method.name:
        layer_dict = {name: module for name, module in method.model.named_modules()}
        # Define the pruning ratio dictionary
        pruning_ratio_dict = {
            layer: args.final_sparsity for layer in layer_dict.values()
        }
        for name in layer_dict:
            if "aggregator" in name:
                pruning_ratio_dict[layer_dict[name]] = args.aggregation_pruning_rate
        return pruning_ratio_dict

    if "mixvpr" in method.name:
        layer_dict = {name: module for name, module in method.model.named_modules()}
        # Define the pruning ratio dictionary
        pruning_ratio_dict = {
            layer: args.final_sparsity for layer in layer_dict.values()
        }
        for name in layer_dict:
            if "aggregator" in name:
                pruning_ratio_dict[layer_dict[name]] = args.aggregation_pruning_rate
        return pruning_ratio_dict

    if "gem" in method.name:
        layer_dict = {name: module for name, module in method.model.named_modules()}

        # Define the pruning ratio dictionary
        pruning_ratio_dict = {
            layer: args.final_sparsity for layer in layer_dict.values()
        }
        for name in layer_dict:
            if "aggregation" in name or "proj" in name:
                pruning_ratio_dict[layer_dict[name]] = args.aggregation_pruning_rate
        return pruning_ratio_dict

    if "netvlad" in method.name:
        layer_dict = {name: module for name, module in method.model.named_modules()}
        # Define the pruning ratio dictionary
        pruning_ratio_dict = {
            layer: args.final_sparsity for layer in layer_dict.values()
        }
        for name in layer_dict:
            if "aggregator" in name or "linear" in name:
                pruning_ratio_dict[layer_dict[name]] = args.aggregation_pruning_rate
        return pruning_ratio_dict

    if "vit" in method.name:
        layer_dict = {name: module for name, module in method.model.named_modules()}
        pruning_ratio_dict = {
            layer: args.final_sparsity for layer in layer_dict.values()
        }
        for name in layer_dict:
            if "encoder.layers.encoder_layer_11.mlp" in name or "encoder.ln" in name:
                pruning_ratio_dict[layer_dict[name]] = args.aggregation_pruning_rate
        return pruning_ratio_dict


def get_mixvpr_in_channel_proj(method):
    layers = {}
    for name, layer in method.model.named_modules():
        # print(name, layer)
        if name == "aggregator.channel_proj":
            layers["layer"] = layer
        if name == "backbone.model.layer3.5.bn3":
            layers["prev_layer"] = layer
    return layers


def get_mixvpr_out_channel_proj(method):
    layers = {}
    for name, layer in method.model.aggregator.named_modules():
        if name in ["channel_proj", "row_proj"]:
            layers[name] = layer
    return layers


def prune_mixvpr_in_channel_proj(method):
    layers = get_mixvpr_in_channel_proj(method)
    num_prune = layers["layer"].in_features - layers["prev_layer"].num_features
    l1_norm = torch.norm(layers["layer"].weight, p=1, dim=0)
    indices_to_prune = torch.topk(l1_norm, num_prune, largest=False).indices
    all_indices = torch.arange(layers["layer"].in_features)
    indices_to_keep = torch.tensor(
        [idx for idx in all_indices if idx not in indices_to_prune]
    )
    new_weight = layers["layer"].weight[:, indices_to_keep].detach()

    if layers["layer"].bias is not None:
        new_bias = layers[
            "layer"
        ].bias.detach()  # Bias remains unchanged because output features are unchanged
    else:
        new_bias = None
    new_layer = nn.Linear(new_weight.size(1), layers["layer"].out_features)
    new_layer.weight = nn.Parameter(new_weight)
    new_layer.bias = nn.Parameter(new_bias) if new_bias is not None else None
    method.model.aggregator.channel_proj = new_layer


def prune_mixvpr_out_channel_proj(method, step):
    layers = get_mixvpr_out_channel_proj(method)
    amount = pruning_schedule(step, True) * args.aggregation_pruning_rate
    num_out_features = layers["channel_proj"].out_features
    num_prune = int(amount * num_out_features)
    l1_norm = torch.norm(layers["channel_proj"].weight.data, p=1, dim=1)
    indices_to_prune = torch.topk(l1_norm, num_prune, largest=False).indices
    mask = torch.ones(num_out_features, dtype=bool)
    mask[indices_to_prune] = False
    new_weight = layers["channel_proj"].weight.data[mask, :].detach()
    if layers["channel_proj"].bias is not None:
        new_bias = layers["channel_proj"].bias[mask].detach()
    else:
        new_bias = None
    new_layer = nn.Linear(layers["channel_proj"].in_features, new_weight.size(0))
    new_layer.weight = nn.Parameter(new_weight)
    if new_bias is not None:
        new_layer.bias = nn.Parameter(new_bias)
    else:
        new_layer.bias = None
    method.model.aggregator.channel_proj = new_layer


def prune_netvlad_centroids(method, step):
    # Prune centroids
    centroids = method.model.aggregation.centroids
    centroids_data = centroids.detach().cpu().numpy()
    amount = pruning_schedule(step, True) * args.aggregation_pruning_rate
    num_clusters = int((1 - amount) * centroids_data.shape[0])
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(centroids_data)
    clustered_data = kmeans.cluster_centers_
    new_centroids = torch.tensor(clustered_data, dtype=centroids.dtype).to(
        centroids.device
    )
    method.model.aggregation.centroids = torch.nn.Parameter(new_centroids)
    method.model.aggregation.clusters_num = int(new_centroids.shape[0])

    # Calculate L1 norm of each filter in the convolutional layer using torch.norm
    conv_filters = method.model.aggregation.conv.weight
    filter_importance = torch.norm(
        conv_filters, p=1, dim=(1, 2, 3)
    )  # Apply L1 norm across channel, height, and width dimensions

    # Prune convolutional filters
    _, important_indices = torch.topk(
        filter_importance, num_clusters
    )  # Get indices of top filters
    new_conv_weights = conv_filters[important_indices, :, :, :]
    method.model.aggregation.conv.weight = torch.nn.Parameter(new_conv_weights)

    # Optionally, adjust the bias if it exists
    if method.model.aggregation.conv.bias is not None:
        new_conv_biases = method.model.aggregation.conv.bias[important_indices]
        method.model.aggregation.conv.bias = torch.nn.Parameter(new_conv_biases)


def setup_pruner(method, args):
    example_img = method.example_input().to(method.device)
    orig_macs, orig_nparams = tp.utils.count_ops_and_params(method.model, example_img)

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

    if args.method.lower() == "mixvpr":
        pruner = tp.pruner.GroupNormPruner(
            method.model.backbone,
            example_img,
            importance,
            iterative_steps=args.max_epochs // args.pruning_freq,
            iterative_pruning_ratio_scheduler=get_scheduler(args),
            pruning_ratio_dict=get_pruning_ratio_dict(method, args),
            ignored_layers=get_dont_prune(method, args),
            channel_groups=get_channel_groups(method, args),
            global_pruning=False,
        )

        class MixVPRPruner:
            def __init__(self, pruner, method):
                self.pruner = pruner
                self.method = method
                self.epoch = 0

            def step(self):
                self.pruner.step()
                prune_mixvpr_in_channel_proj(self.method)
                prune_mixvpr_out_channel_proj(self.method, self.epoch)
                self.epoch += 1

        pruner = MixVPRPruner(pruner, method)
        return method, pruner, orig_nparams

    elif args.method.lower() == "netvlad":
        pruner = tp.pruner.GroupNormPruner(
            method.model.backbone,
            example_img,
            importance,
            iterative_steps=args.max_epochs // args.pruning_freq,
            iterative_pruning_ratio_scheduler=get_scheduler(args),
            pruning_ratio_dict=get_pruning_ratio_dict(method, args),
            ignored_layers=get_dont_prune(method, args),
            channel_groups=get_channel_groups(method, args),
            global_pruning=False,
        )

        class NetVLADPruner:
            def __init__(self, pruner, method):
                self.pruner = pruner
                self.method = method
                self.epoch = 0

            def step(self):
                self.pruner.step()
                prune_netvlad_centroids(self.method, self.epoch)
                self.epoch += 1

        pruner = NetVLADPruner(pruner, method)

        return method, pruner, orig_nparams

    else:
        pruner = tp.pruner.GroupNormPruner(
            method.model,
            example_img,
            importance,
            iterative_steps=args.max_epochs // args.pruning_freq,
            iterative_pruning_ratio_scheduler=get_scheduler(args),
            pruning_ratio_dict=get_pruning_ratio_dict(method, args),
            ignored_layers=get_dont_prune(method, args),
            channel_groups=get_channel_groups(method, args),
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
        method_name,
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
        pruning_freq=5,
    ):
        super().__init__()
        self.name = method_name
        self.method = get_method(method_name, pretrained=True)

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult
        self.pruning_freq = pruning_freq
        self.pruning_type = args.pruning_type
        self.pruning_schedule = args.pruning_schedule
        self.pruning_freq = args.pruning_freq
        self.initial_sparsity = args.initial_sparsity
        self.final_sparsity = args.final_sparsity
        self.eval_distance = args.eval_distance

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin

        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = []

        self.faiss_gpu = faiss_gpu

        self.method, self.pruner, self.orig_nparams = setup_pruner(self.method, args)
        self.model = self.method.model
        self.preprocess = self.method.preprocess
        self.model.train()
        self.epoch = 0
        self.save_hyperparameters(args)
        self.hparams.update(
            {"feature_size": self.method.features_dim["global_feature_shape"]}
        )

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

    def on_train_epoch_start(self):
        """
        Completes the pruning step if required
        """
        if self.epoch % self.pruning_freq == 0 and self.epoch != 0:
            self.pruner.step()
            self.trainer.optimizers = self.configure_optimizers()[0]
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

        cpu_lat1 = utils.measure_cpu_latency(
            self.model, self.method.example_input(), batch_size=1
        )
        gpu_lat1 = utils.measure_gpu_latency(
            self.model, self.method.example_input(), batch_size=1
        )
        cpu_lat50 = utils.measure_cpu_latency(
            self.model, self.method.example_input(), batch_size=50
        )
        gpu_lat50 = utils.measure_gpu_latency(
            self.model, self.method.example_input(), batch_size=50
        )

        self.log("cpu_bs1_lat_ms", cpu_lat1)
        self.log("gpu_bs1_lat_ms", gpu_lat1)
        self.log("cpu_bs50_lat_ms", cpu_lat50)
        self.log("gpu_bs50_lat_ms", gpu_lat50)

        for i, (val_set_name, val_dataset) in enumerate(
            zip(dm.val_set_names, dm.val_datasets)
        ):
            feats = torch.concat(val_step_outputs[i], dim=0)
            num_references = val_dataset.num_references
            num_queries = val_dataset.num_queries
            ground_truth = val_dataset.ground_truth
            r_list = feats[:num_references]
            q_list = feats[num_references:]

            macs, nparams = tp.utils.count_ops_and_params(
                self.model,
                self.method.example_input().to(next(self.model.parameters()).device),
            )
            sparsity = 1 - (nparams / self.orig_nparams)

            rat100p = utils.get_validation_recall_at_100precision(
                r_list=r_list,
                q_list=q_list,
                gt=ground_truth,
                print_results=True,
                dataset_name=val_set_name,
                distance=self.eval_distance,
            )

            self.log(f"{val_set_name}/recall@100p", rat100p)

            recalls_dict, predictions, ret_latency = utils.get_validation_recalls(
                r_list=r_list,
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 25],
                gt=ground_truth,
                print_results=True,
                dataset_name=val_set_name,
                distance=self.eval_distance,
                sparsity=sparsity,
                descriptor_dim=val_step_outputs[0][0].shape[1],
            )

            del r_list, q_list, ground_truth

            self.log(
                f"{val_set_name}/map_memory_mb",
                (num_references * feats.shape[1] * 2) / (1024 * 1024),
            )

            self.log(f"{val_set_name}/retrieval_lat_ms", ret_latency)

            self.log(f"{val_set_name}/total_cpu_lat_bs50", cpu_lat50 + ret_latency)

            self.log(f"{val_set_name}/total_cpu_lat_bs1", cpu_lat1 + ret_latency)

            self.log(f"{val_set_name}/total_gpu_lat_bs50", gpu_lat50 + ret_latency)

            self.log(f"{val_set_name}/total_gpu_lat_bs1", gpu_lat1 + ret_latency)

            print(
                f"{val_set_name}____references {num_references} dim {feats.shape[1]} map_memory {(num_references * feats.shape[1] * 4) / (1024 * 1024)} total memory {(nparams * 2 + (num_references * feats.shape[1] * 4)) / (1024 * 1024)}"
            )
            self.log(
                f"{val_set_name}/total_memory_mb",
                (nparams * 2 + (num_references * feats.shape[1] * 4)) / (1024 * 1024),
            )
            self.log(f"{val_set_name}/R1", recalls_dict[1], prog_bar=False, logger=True)
            self.log(f"{val_set_name}/R5", recalls_dict[5], prog_bar=False, logger=True)
            self.log(
                f"{val_set_name}/R10", recalls_dict[10], prog_bar=False, logger=True
            )
            print("\n\n")

        self.log("flops", macs)
        self.log("descriptor_dim", val_step_outputs[0][0].shape[1])
        self.log("sparsity", sparsity)
        self.log("macs", macs / 1e6)
        self.log("nparams", nparams)
        self.log("model_memory_mb", (nparams * 4) / (1024 * 1024))


# =================================== Training Loop ================================
def sparse_structured_trainer(args):
    pl.seed_everything(seed=1, workers=True)
    torch.set_float32_matmul_precision("medium")

    wandb_logger = WandbLogger(project="GSVCities", name=args.method)

    datamodule = GSVCitiesDataModule(
        cities=get_cities(args),
        batch_size=args.batch_size,
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False,
        random_sample_from_each_place=True,
        image_size=args.image_resolution,
        num_workers=args.num_workers,
        show_data_stats=False,
        val_set_names=[
            "pitts30k_val",
            "inria",
            "spedtest",
            "mapillarysls",
            "essex3in1",
            "nordland",
            "crossseasons",
        ],
        # val_set_names=["spedtest"],
    )

    if args.checkpoint:
        checkpoint_cb = ModelCheckpoint(
            dirpath=f"Checkpoints/gsv_cities_sparse_structured/{args.method}/{args.pruning_type}/",
            filename=f"{args.method}"
            + "_epoch_{epoch:02d}_step_{step:04d}_R1_{pitts30k_val/R1:.4f}_sparsity_"
            + f"_{sparsity:.2f}",
            auto_insert_metric_name=False,
            save_weights_only=True,
            every_n_epochs=args.pruning_freq,
        )

    module = VPRModel(
        args=args,
        method_name=args.method,
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
    )

    if args.debug:
        trainer = pl.Trainer(
            enable_progress_bar=args.enable_progress_bar,
            devices="auto",
            accelerator="auto",
            strategy="auto",
            default_root_dir=f"./LOGS/{args.method}",
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
            default_root_dir=f"./LOGS/{args.method}",
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
            check_val_every_n_epoch=args.pruning_freq,
        )

    trainer.fit(model=module, datamodule=datamodule)
