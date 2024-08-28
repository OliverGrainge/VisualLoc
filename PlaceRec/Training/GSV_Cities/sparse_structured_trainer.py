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
from sklearn.cluster import KMeans

import PlaceRec.Training.GSV_Cities.utils as utils
from PlaceRec.Training.GSV_Cities.dataloaders.GSVCitiesDataloader import (
    GSVCitiesDataModule,
)
from PlaceRec.Training.GSV_Cities.sparse_utils import get_cities, pruning_schedule
from PlaceRec.utils import get_config, get_method

config = get_config()


def get_all_milestones(pruning_freq, milestones, max_epochs):
    all_milestones = milestones
    start = pruning_freq
    while all_milestones[-1] <= max_epochs + pruning_freq:
        new_milestones = [start + ms for ms in milestones]
        start = start + pruning_freq
        all_milestones += new_milestones
    return all_milestones


def get_scheduler(args):
    pruning_freq = args.pruning_freq

    def schd(pruning_ratio_dict, steps):
        return [
            pruning_schedule(args, i * pruning_freq, cumulative=False)
            * pruning_ratio_dict
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
        if (
            name == "backbone.model.layer3.5.bn3"
            and "resnet34" not in method.name.lower()
        ):
            layers["prev_layer"] = layer

        if name == "backbone.model.layer3.5.bn2" and "resnet34" in method.name.lower():
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


def prune_mixvpr_out_channel_proj(method, step, args):
    layers = get_mixvpr_out_channel_proj(method)
    amount = pruning_schedule(args, step * args.pruning_freq, True)
    # num_out_features = layers["channel_proj"].out_features
    if "resnet34" in method.name.lower():
        num_out_features = 256
        # (num_out_features, layers["channel_proj"].weight.data.shape)

    else:
        num_out_features = 1024
        # print(num_out_features, layers["channel_proj"].weight.data.shape)

    num_prune = int(amount * num_out_features)
    l1_norm = torch.norm(layers["channel_proj"].weight.data, p=1, dim=1)
    indices_to_prune = torch.topk(l1_norm, num_prune, largest=False).indices
    mask = torch.ones(layers["channel_proj"].weight.data.shape[0], dtype=bool)
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


def prune_netvlad_centroids(method, step, args):
    # Prune centroids
    centroids = method.model.aggregation.centroids
    centroids_data = centroids.detach().cpu().numpy()
    amount = pruning_schedule(args, step * args.pruning_freq, True)
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


def prune_linear_layer_by_l2(in_features: int, layer: nn.Linear, epoch, args):
    """
    Prune the input and output dimensions of a linear layer based on L2 norm of the weights.

    Args:
        in_features (int): The target input features to be retained.
        layer (nn.Linear): The linear layer to be pruned.
        epoch (int): The current training epoch.
        args (argparse.Namespace): The arguments including pruning settings.

    Returns:
        None
    """
    # Check if the layer is an instance of nn.Linear
    if not isinstance(layer, nn.Linear):
        raise ValueError("The layer to be pruned must be an instance of nn.Linear")

    # Calculate the pruning amount for output dimensions
    amount = pruning_schedule(args, epoch * args.pruning_freq, True)

    if in_features < layer.in_features:
        l2_norms_in = torch.norm(layer.weight.data, p=2, dim=0)
        num_neurons_to_prune_in = layer.in_features - in_features
        prune_indices_in = torch.argsort(l2_norms_in)[:num_neurons_to_prune_in]
        keep_indices_in = torch.argsort(l2_norms_in)[num_neurons_to_prune_in:]

        # Update weights for input pruning
        layer.weight.data = layer.weight.data[:, keep_indices_in].clone()

        # Update the in_features attribute
        layer.in_features = in_features

    if amount == 0:
        return

    # Prune output dimensions
    l2_norms_out = torch.norm(layer.weight.data, p=2, dim=1)
    # num_out_features = layer.out_features
    num_out_features = 2048
    num_neurons_to_prune_out = int(amount * num_out_features)
    prune_indices_out = torch.argsort(l2_norms_out)[:num_neurons_to_prune_out]
    keep_indices_out = torch.argsort(l2_norms_out)[num_neurons_to_prune_out:]

    # Update weights and biases for output pruning
    layer.weight.data = layer.weight.data[keep_indices_out].clone()
    if layer.bias is not None:
        layer.bias.data = layer.bias.data[keep_indices_out].clone()

    # Update the out_features attribute
    layer.out_features -= num_neurons_to_prune_out


def prune_conv_layer_by_l2(in_channels: int, layer: nn.Conv2d, epoch, args):

    # Check if the layer is an instance of nn.Conv2d
    if not isinstance(layer, nn.Conv2d):
        raise ValueError("The layer to be pruned must be an instance of nn.Conv2d")

    # Calculate the pruning amount for output channels
    amount = pruning_schedule(args, epoch * args.pruning_freq, True)

    if in_channels < layer.in_channels:
        # Compute L2 norm across the spatial dimensions of the kernel weights
        l2_norms_in = torch.norm(
            layer.weight.data.view(layer.out_channels, layer.in_channels, -1),
            dim=2,
            p=2,
        ).mean(dim=0)
        num_channels_to_prune_in = layer.in_channels - in_channels
        prune_indices_in = torch.argsort(l2_norms_in)[:num_channels_to_prune_in]
        keep_indices_in = torch.argsort(l2_norms_in)[num_channels_to_prune_in:]

        # Update weights for input channel pruning
        layer.weight.data = layer.weight.data[:, keep_indices_in, :, :].clone()

        # Update the in_channels attribute
        layer.in_channels = in_channels

    if amount == 0:
        return

    # Prune output channels
    l2_norms_out = torch.norm(
        layer.weight.data.view(layer.out_channels, -1), dim=1, p=2
    )

    num_channels_to_prune_out = int(amount * 1024)
    prune_indices_out = torch.argsort(l2_norms_out)[:num_channels_to_prune_out]
    keep_indices_out = torch.argsort(l2_norms_out)[num_channels_to_prune_out:]

    # Update weights and biases for output channel pruning
    layer.weight.data = layer.weight.data[keep_indices_out, :, :, :].clone()
    if layer.bias is not None:
        layer.bias.data = layer.bias.data[keep_indices_out].clone()

    # Update the out_channels attribute
    layer.out_channels -= num_channels_to_prune_out


def get_gem_proj_layer(method):
    for name, layer in method.model.named_modules():
        if name == "proj":
            return layer


def get_convap_channel_pool(method):
    for name, layer in method.model.named_modules():
        if "channel_pool" in name:
            return layer


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

    if "mixvpr" in args.method.lower():
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
            def __init__(self, pruner, method, args):
                self.pruner = pruner
                self.method = method
                self.args = args
                self.epoch = 0

            def step(self):
                self.pruner.step()
                prune_mixvpr_in_channel_proj(self.method)
                prune_mixvpr_out_channel_proj(self.method, self.epoch, self.args)
                self.epoch += 1

        pruner = MixVPRPruner(pruner, method, args)
        return method, pruner, orig_nparams

    elif "netvlad" in args.method.lower():
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
            def __init__(self, pruner, method, args):
                self.pruner = pruner
                self.method = method
                self.args = args
                self.epoch = 0

            def step(self):
                self.pruner.step()
                prune_netvlad_centroids(self.method, self.epoch, args)
                self.epoch += 1

        pruner = NetVLADPruner(pruner, method, args)

        return method, pruner, orig_nparams

    elif "gem" in args.method.lower():
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

        class GemPruner:
            def __init__(self, pruner, method, args):
                self.pruner = pruner
                self.method = method
                self.args = args
                self.epoch = 0

            def step(self):
                self.pruner.step()
                x = torch.randn(1, 3, 320, 320).to(
                    next(method.model.parameters()).device
                )
                x = self.method.model.backbone(x)
                x = self.method.model.aggregation(x)
                in_dim = x.shape[1]
                layer = get_gem_proj_layer(self.method)
                prune_linear_layer_by_l2(in_dim, layer, self.epoch, self.args)
                self.epoch += 1

        pruner = GemPruner(pruner, method, args)

        return method, pruner, orig_nparams

    elif "convap" in args.method.lower():
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

        class ConvAPPruner:
            def __init__(self, pruner, method, args):
                self.pruner = pruner
                self.method = method
                self.args = args
                self.epoch = 0
                self.initial_out_channels = None

            def step(self):
                if self.initial_out_channels is None:
                    x = torch.randn(1, 3, 320, 320).to(
                        next(method.model.parameters()).device
                    )
                    x = self.method.model.backbone(x)

                self.pruner.step()
                x = torch.randn(1, 3, 320, 320).to(
                    next(method.model.parameters()).device
                )
                x = self.method.model.backbone(x)
                in_channels = x.shape[1]
                layer = get_convap_channel_pool(self.method)
                prune_conv_layer_by_l2(in_channels, layer, self.epoch, self.args)
                self.epoch += 1

        pruner = ConvAPPruner(pruner, method, args)

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
        checkpoint=True,
        aggregation_pruning_rate=None,
    ):
        super().__init__()
        self.name = method_name
        self.method = get_method(method_name, pretrained=True)

        self.lr = lr
        self.optimizer_type = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmup_steps = warmup_steps

        self.lr_mult = lr_mult
        self.pruning_freq = pruning_freq
        self.pruning_type = args.pruning_type
        self.pruning_schedule = args.pruning_schedule
        self.pruning_freq = args.pruning_freq
        self.milestones = get_all_milestones(self.pruning_freq, milestones, 30)
        self.initial_sparsity = args.initial_sparsity
        self.final_sparsity = args.final_sparsity
        self.eval_distance = args.eval_distance
        self.aggregation_pruning_rate = aggregation_pruning_rate

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
        self.save_hyperparameters(args)
        self.hparams.update(
            {"feature_size": self.method.features_dim["global_feature_shape"]}
        )
        self.checkpoint = checkpoint

        assert isinstance(self.model, torch.nn.Module)

        # print("=== filepath", f"/home/oeg1n18/VisualLoc/Checkpoints/test_save.ckpt")
        torch.save(self.model, f"/home/oeg1n18/VisualLoc/Checkpoints/test_save.ckpt")

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(
        self, last_epoch=-1
    ) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        if self.optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        elif self.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == "adam":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(
                f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"'
            )

        return [optimizer]

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
        self.feats_shape = descriptors.shape[1]
        loss = self.loss_function(descriptors, labels)
        self.log("loss", loss.item(), logger=True)
        return {"loss": loss}

    def on_train_epoch_start(self):
        """
        Completes the pruning step if required
        """
        print("================== Starting Train Epoch", self.current_epoch)
        if self.current_epoch % self.pruning_freq == 0:
            self.pruner.step()
            print(
                "===============================================================  EPOCH PRUNING:",
                self.current_epoch,
            )

        if self.current_epoch % self.pruning_freq == 0:
            self.reset_optimizer()
            print("Optimizer reset at train epoch start")
        else:
            self.adjust_learning_rate(self.trainer.optimizers[0], self.current_epoch)
            print("learning rate adjusted at train epoch start")
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"])

    def reset_optimizer(self):
        # Reinitialize the optimizer

        self.trainer.optimizers = self.configure_optimizers()

    def adjust_learning_rate(self, optimizer, epoch):
        # Custom learning rate adjustment logic
        # milestones = [ms + epoch for ms in self.milestones]
        milestones = self.milestones
        if epoch in milestones:
            new_lr = optimizer.param_groups[0]["lr"] * 0.5
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
            print(f"Learning rate adjusted to {new_lr}")

    def on_train_epoch_end(self) -> None:
        """
        Hook called at the end of a training epoch to reset or update certain parameters.
        """
        print("=========== Ending Training Epoch: ", self.current_epoch)
        self.batch_acc = []

    def on_validation_epoch_start(self) -> None:
        """
        Hook called at the start of a validation epoch to initialize or reset parameters.
        """
        print("=========== Starting Validation Epoch: ", self.current_epoch)
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
        print("================ Validation Epoch End: ", self.current_epoch)
        val_step_outputs = self.val_step_outputs
        dm = self.trainer.datamodule
        self.val_step_outputs = []
        if len(dm.val_datasets) == 1:  # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]
        """
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
        """

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

            # self.log(f"{val_set_name}/retrieval_lat_ms", ret_latency)

            # self.log(f"{val_set_name}/total_cpu_lat_bs50", cpu_lat50 + ret_latency)

            # self.log(f"{val_set_name}/total_cpu_lat_bs1", cpu_lat1 + ret_latency)

            # self.log(f"{val_set_name}/total_gpu_lat_bs50", gpu_lat50 + ret_latency)

            # self.log(f"{val_set_name}/total_gpu_lat_bs1", gpu_lat1 + ret_latency)

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

        print("-----")
        print("-----")
        print("-----")
        print("==== SPARSITY: ", sparsity, "feature_dim", self.feats_shape)
        print("-----")
        print("-----")
        print("-----")

        # print("=== filepath", f"/home/oeg1n18/VisualLoc/Checkpoints/{self.name}_agg_{self.aggregation_pruning_rate:.2f}_sparsity_{sparsity:.3f}_R1_{recalls_dict[1]:.3f}.ckpt")
        print("===================== EPOCH SAVING:", self.current_epoch)
        torch.save(
            self.model,
            f"/home/oeg1n18/VisualLoc/Checkpoints/{self.name}_agg_{self.aggregation_pruning_rate:.2f}_sparsity_{sparsity:.3f}_R1_{recalls_dict[1]:.3f}.ckpt",
        )


# =================================== Training Loop ================================
def sparse_structured_trainer(args):
    pl.seed_everything(seed=1, workers=True)
    torch.set_float32_matmul_precision("medium")

    wandb_logger = WandbLogger(project="GSVCities", name=args.method)

    print(
        "===========================================================================",
        args.aggregation_pruning_rate,
    )
    raise Exception

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
            #    "inria",
            ##    "essex3in1",
            #    "spedtest",
            #    "mapillarysls",
            #    "nordland",
            #    "crossseasons",
        ],
        # val_set_names=["spedtest"],
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
        checkpoint=args.checkpoint,
        aggregation_pruning_rate=args.aggregation_pruning_rate,
    )
    if "netvlad" in args.method:
        prec = 32
    else:
        prec = 16
    if args.debug:
        trainer = pl.Trainer(
            enable_progress_bar=args.enable_progress_bar,
            devices="auto",
            accelerator="auto",
            strategy="auto",
            default_root_dir=f"./LOGS/{args.method}",
            num_sanity_val_steps=0,
            # precision=32,
            max_epochs=args.max_epochs,
            reload_dataloaders_every_n_epochs=1,
            logger=wandb_logger,
            log_every_n_steps=1,
            limit_train_batches=1,
            check_val_every_n_epoch=args.pruning_freq,
            # limit_val_batches=1,
        )
    else:
        print("======================== Precision: ", prec)
        trainer = pl.Trainer(
            enable_progress_bar=args.enable_progress_bar,
            devices="auto",
            accelerator="auto",
            strategy="auto",
            default_root_dir=f"./LOGS/{args.method}",
            num_sanity_val_steps=0,
            # precision=32,
            max_epochs=args.max_epochs,
            reload_dataloaders_every_n_epochs=1,
            logger=wandb_logger,
            check_val_every_n_epoch=args.pruning_freq,
        )

    trainer.fit(model=module, datamodule=datamodule)
