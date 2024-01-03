import math
import os
from collections import OrderedDict

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
from torch.hub import load_state_dict_from_url
from torch.nn import (
    Dropout,
    Identity,
    LayerNorm,
    Linear,
    Module,
    ModuleList,
    Parameter,
    init,
)
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm

from PlaceRec.Methods import BaseModelWrapper
from PlaceRec.utils import L2Norm

filepath = os.path.dirname(os.path.abspath(__file__))


model_urls = {
    "cct_7_3x1_32": "https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_cifar10_300epochs.pth",
    "cct_7_3x1_32_sine": "https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_sine_cifar10_5000epochs.pth",
    "cct_7_3x1_32_c100": "https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_cifar100_300epochs.pth",
    "cct_7_3x1_32_sine_c100": "https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_3x1_32_sine_cifar100_5000epochs.pth",
    "cct_7_7x2_224_sine": "https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_7_7x2_224_flowers102.pth",
    "cct_14_7x2_224": "https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_14_7x2_224_imagenet.pth",
    "cct_14_7x2_384": "https://shi-labs.com/projects/cct/checkpoints/finetuned/cct_14_7x2_384_imagenet.pth",
    "cct_14_7x2_384_fl": "https://shi-labs.com/projects/cct/checkpoints/finetuned/cct_14_7x2_384_flowers102.pth",
}


class CCT(nn.Module):
    def __init__(
        self,
        img_size=224,
        embedding_dim=768,
        n_input_channels=3,
        n_conv_layers=1,
        kernel_size=7,
        stride=2,
        padding=3,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        dropout=0.0,
        attention_dropout=0.1,
        stochastic_depth=0.1,
        num_layers=14,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=1000,
        positional_embedding="learnable",
        aggregation=None,
        *args,
        **kwargs,
    ):
        super(CCT, self).__init__()

        self.tokenizer = Tokenizer(
            n_input_channels=n_input_channels,
            n_output_channels=embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            pooling_padding=pooling_padding,
            max_pool=True,
            activation=nn.ReLU,
            n_conv_layers=n_conv_layers,
            conv_bias=False,
        )

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels, height=img_size, width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding,
        )
        if aggregation in ["cls", "seqpool"]:
            self.aggregation = aggregation
        else:
            self.aggregation = None

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.classifier(x)
        if self.aggregation == "cls":
            return x[:, 0]
        elif self.aggregation == "seqpool":
            x = torch.matmul(F.softmax(self.classifier.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
            return x
        else:
            # x = x.permute(0, 2, 1)
            return x


def _cct(
    arch,
    pretrained,
    progress,
    num_layers,
    num_heads,
    mlp_ratio,
    embedding_dim,
    kernel_size=3,
    stride=None,
    padding=None,
    aggregation=None,
    *args,
    **kwargs,
):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    model = CCT(
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        embedding_dim=embedding_dim,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        aggregation=aggregation,
        *args,
        **kwargs,
    )

    if pretrained:
        if arch in model_urls:
            state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
            state_dict = pe_check(model, state_dict)
            model.load_state_dict(state_dict, strict=False)
        else:
            raise RuntimeError(f"Variant {arch} does not yet have pretrained weights.")
    return model


def cct_2(arch, pretrained, progress, aggregation=None, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128, aggregation=aggregation, *args, **kwargs)


def cct_4(arch, pretrained, progress, aggregation=None, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128, aggregation=aggregation, *args, **kwargs)


def cct_6(arch, pretrained, progress, aggregation=None, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256, aggregation=aggregation, *args, **kwargs)


def cct_7(arch, pretrained, progress, aggregation=None, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256, aggregation=aggregation, *args, **kwargs)


def cct_14(arch, pretrained, progress, aggregation=None, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384, aggregation=aggregation, *args, **kwargs)


@register_model
def cct_2_3x2_32(pretrained=False, progress=False, img_size=32, positional_embedding="learnable", num_classes=10, aggregation=None, *args, **kwargs):
    return cct_2(
        "cct_2_3x2_32",
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        aggregation=aggregation,
        *args,
        **kwargs,
    )


@register_model
def cct_2_3x2_32_sine(pretrained=False, progress=False, img_size=32, positional_embedding="sine", num_classes=10, aggregation=None, *args, **kwargs):
    return cct_2(
        "cct_2_3x2_32_sine",
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        aggregation=aggregation,
        *args,
        **kwargs,
    )


@register_model
def cct_4_3x2_32(pretrained=False, progress=False, img_size=32, positional_embedding="learnable", num_classes=10, aggregation=None, *args, **kwargs):
    return cct_4(
        "cct_4_3x2_32",
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        aggregation=aggregation,
        *args,
        **kwargs,
    )


@register_model
def cct_4_3x2_32_sine(pretrained=False, progress=False, img_size=32, positional_embedding="sine", num_classes=10, aggregation=None, *args, **kwargs):
    return cct_4(
        "cct_4_3x2_32_sine",
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        aggregation=aggregation,
        *args,
        **kwargs,
    )


@register_model
def cct_6_3x1_32(pretrained=False, progress=False, img_size=32, positional_embedding="learnable", num_classes=10, aggregation=None, *args, **kwargs):
    return cct_6(
        "cct_6_3x1_32",
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        aggregation=aggregation,
        *args,
        **kwargs,
    )


@register_model
def cct_6_3x1_32_sine(pretrained=False, progress=False, img_size=32, positional_embedding="sine", num_classes=10, aggregation=None, *args, **kwargs):
    return cct_6(
        "cct_6_3x1_32_sine",
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        aggregation=aggregation,
        *args,
        **kwargs,
    )


@register_model
def cct_6_3x2_32(pretrained=False, progress=False, img_size=32, positional_embedding="learnable", num_classes=10, aggregation=None, *args, **kwargs):
    return cct_6(
        "cct_6_3x2_32",
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        aggregation=aggregation,
        *args,
        **kwargs,
    )


@register_model
def cct_6_3x2_32_sine(pretrained=False, progress=False, img_size=32, positional_embedding="sine", num_classes=10, aggregation=None, *args, **kwargs):
    return cct_6(
        "cct_6_3x2_32_sine",
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        aggregation=aggregation,
        *args,
        **kwargs,
    )


@register_model
def cct_7_3x1_32(pretrained=False, progress=False, img_size=32, positional_embedding="learnable", num_classes=10, aggregation=None, *args, **kwargs):
    return cct_7(
        "cct_7_3x1_32",
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        aggregation=aggregation,
        *args,
        **kwargs,
    )


@register_model
def cct_7_3x1_32_sine(pretrained=False, progress=False, img_size=32, positional_embedding="sine", num_classes=10, aggregation=None, *args, **kwargs):
    return cct_7(
        "cct_7_3x1_32_sine",
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        aggregation=aggregation,
        *args,
        **kwargs,
    )


@register_model
def cct_7_3x1_32_c100(
    pretrained=False, progress=False, img_size=32, positional_embedding="learnable", num_classes=100, aggregation=None, *args, **kwargs
):
    return cct_7(
        "cct_7_3x1_32_c100",
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        aggregation=aggregation,
        *args,
        **kwargs,
    )


@register_model
def cct_7_3x1_32_sine_c100(
    pretrained=False, progress=False, img_size=32, positional_embedding="sine", num_classes=100, aggregation=None, *args, **kwargs
):
    return cct_7(
        "cct_7_3x1_32_sine_c100",
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        aggregation=aggregation,
        *args,
        **kwargs,
    )


@register_model
def cct_7_3x2_32(pretrained=False, progress=False, img_size=32, positional_embedding="learnable", num_classes=10, aggregation=None, *args, **kwargs):
    return cct_7(
        "cct_7_3x2_32",
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        aggregation=aggregation,
        *args,
        **kwargs,
    )


@register_model
def cct_7_3x2_32_sine(pretrained=False, progress=False, img_size=32, positional_embedding="sine", num_classes=10, aggregation=None, *args, **kwargs):
    return cct_7(
        "cct_7_3x2_32_sine",
        pretrained,
        progress,
        kernel_size=3,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        aggregation=aggregation,
        *args,
        **kwargs,
    )


@register_model
def cct_7_7x2_224(
    pretrained=False, progress=False, img_size=224, positional_embedding="learnable", num_classes=102, aggregation=None, *args, **kwargs
):
    return cct_7(
        "cct_7_7x2_224",
        pretrained,
        progress,
        kernel_size=7,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        aggregation=aggregation,
        *args,
        **kwargs,
    )


@register_model
def cct_7_7x2_224_sine(
    pretrained=False, progress=False, img_size=224, positional_embedding="sine", num_classes=102, aggregation=None, *args, **kwargs
):
    return cct_7(
        "cct_7_7x2_224_sine",
        pretrained,
        progress,
        kernel_size=7,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        aggregation=aggregation,
        *args,
        **kwargs,
    )


@register_model
def cct_14_7x2_224(
    pretrained=False, progress=False, img_size=224, positional_embedding="learnable", num_classes=1000, aggregation=None, *args, **kwargs
):
    return cct_14(
        "cct_14_7x2_224",
        pretrained,
        progress,
        kernel_size=7,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        aggregation=aggregation,
        *args,
        **kwargs,
    )


@register_model
def cct_14_7x2_384(
    pretrained=False, progress=False, img_size=384, positional_embedding="learnable", num_classes=1000, aggregation=None, *args, **kwargs
):
    return cct_14(
        "cct_14_7x2_384",
        pretrained,
        progress,
        kernel_size=7,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        aggregation=aggregation,
        *args,
        **kwargs,
    )


@register_model
def cct_14_7x2_384_fl(
    pretrained=False, progress=False, img_size=384, positional_embedding="learnable", num_classes=102, aggregation=None, *args, **kwargs
):
    return cct_14(
        "cct_14_7x2_384_fl",
        pretrained,
        progress,
        kernel_size=7,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        aggregation=aggregation,
        *args,
        **kwargs,
    )


class Embedder(nn.Module):
    def __init__(self, word_embedding_dim=300, vocab_size=100000, padding_idx=1, pretrained_weight=None, embed_freeze=False, *args, **kwargs):
        super(Embedder, self).__init__()
        self.embeddings = (
            nn.Embedding.from_pretrained(pretrained_weight, freeze=embed_freeze)
            if pretrained_weight is not None
            else nn.Embedding(vocab_size, word_embedding_dim, padding_idx=padding_idx)
        )
        self.embeddings.weight.requires_grad = not embed_freeze

    def forward_mask(self, mask):
        bsz, seq_len = mask.shape
        new_mask = mask.view(bsz, seq_len, 1)
        new_mask = new_mask.sum(-1)
        new_mask = new_mask > 0
        return new_mask

    def forward(self, x, mask=None):
        embed = self.embeddings(x)
        embed = embed if mask is None else embed * self.forward_mask(mask).unsqueeze(-1).float()
        return embed, mask

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        else:
            nn.init.normal_(m.weight)


def resize_pos_embed(posemb, posemb_new, num_tokens=1):
    # Copied from `timm` by Ross Wightman:
    # github.com/rwightman/pytorch-image-models
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def pe_check(model, state_dict, pe_key="classifier.positional_emb"):
    if pe_key is not None and pe_key in state_dict.keys() and pe_key in model.state_dict().keys():
        if model.state_dict()[pe_key].shape != state_dict[pe_key].shape:
            state_dict[pe_key] = resize_pos_embed(state_dict[pe_key], model.state_dict()[pe_key], num_tokens=model.classifier.num_tokens)
    return state_dict


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Tokenizer(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        n_conv_layers=1,
        n_input_channels=3,
        n_output_channels=64,
        in_planes=64,
        activation=None,
        max_pool=True,
        conv_bias=False,
    ):
        super(Tokenizer, self).__init__()

        n_filter_list = [n_input_channels] + [in_planes for _ in range(n_conv_layers - 1)] + [n_output_channels]

        self.conv_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        n_filter_list[i],
                        n_filter_list[i + 1],
                        kernel_size=(kernel_size, kernel_size),
                        stride=(stride, stride),
                        padding=(padding, padding),
                        bias=conv_bias,
                    ),
                    nn.Identity() if activation is None else activation(),
                    nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride, padding=pooling_padding) if max_pool else nn.Identity(),
                )
                for i in range(n_conv_layers)
            ]
        )

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return self.flattener(self.conv_layers(x)).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class TextTokenizer(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        embedding_dim=300,
        n_output_channels=128,
        activation=None,
        max_pool=True,
        *args,
        **kwargs,
    ):
        super(TextTokenizer, self).__init__()

        self.max_pool = max_pool
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, n_output_channels, kernel_size=(kernel_size, embedding_dim), stride=(stride, 1), padding=(padding, 0), bias=False),
            nn.Identity() if activation is None else activation(),
            nn.MaxPool2d(kernel_size=(pooling_kernel_size, 1), stride=(pooling_stride, 1), padding=(pooling_padding, 0))
            if max_pool
            else nn.Identity(),
        )

        self.apply(self.init_weight)

    def seq_len(self, seq_len=32, embed_dim=300):
        return self.forward(torch.zeros((1, seq_len, embed_dim)))[0].shape[1]

    def forward_mask(self, mask):
        new_mask = mask.unsqueeze(1).float()
        cnn_weight = torch.ones((1, 1, self.conv_layers[0].kernel_size[0]), device=mask.device, dtype=torch.float)
        new_mask = F.conv1d(new_mask, cnn_weight, None, self.conv_layers[0].stride[0], self.conv_layers[0].padding[0], 1, 1)
        if self.max_pool:
            new_mask = F.max_pool1d(
                new_mask, self.conv_layers[2].kernel_size[0], self.conv_layers[2].stride[0], self.conv_layers[2].padding[0], 1, False, False
            )
        new_mask = new_mask.squeeze(1)
        new_mask = new_mask > 0
        return new_mask

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.transpose(1, 3).squeeze(1)
        x = x if mask is None else x * self.forward_mask(mask).unsqueeze(-1).float()
        return x, mask

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class Attention(nn.Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MaskedAttention(Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(attn.dtype).max
            assert mask.shape[-1] == attn.shape[-1], "mask has incorrect dimensions"
            mask = mask[:, None, :] * mask[:, :, None]
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn.masked_fill_(~mask, mask_value)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead, attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


class MaskedTransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, attention_dropout=0.1, drop_path_rate=0.1):
        super(MaskedTransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = MaskedAttention(dim=d_model, num_heads=nhead, attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, mask=None, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src), mask))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


class TransformerClassifier(Module):
    def __init__(
        self,
        seq_pool=True,
        embedding_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=1000,
        dropout=0.1,
        attention_dropout=0.1,
        stochastic_depth=0.1,
        positional_embedding="learnable",
        sequence_length=None,
    ):
        super().__init__()
        positional_embedding = positional_embedding if positional_embedding in ["sine", "learnable", "none"] else "sine"
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        assert sequence_length is not None or positional_embedding == "none", (
            f"Positional embedding is set to {positional_embedding} and" f" the sequence length was not specified."
        )

        if not seq_pool:
            sequence_length += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim), requires_grad=True)
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != "none":
            if positional_embedding == "learnable":
                self.positional_emb = Parameter(torch.zeros(1, sequence_length, embedding_dim), requires_grad=True)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim), requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path_rate=dpr[i],
                )
                for i in range(num_layers)
            ]
        )
        self.norm = LayerNorm(embedding_dim)

        # self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode="constant", value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # TODO: TOREMOVE
        # if self.seq_pool:
        #    x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        # else:
        #    x = x[:, 0]
        # x = self.fc(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)] for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


class MaskedTransformerClassifier(Module):
    def __init__(
        self,
        seq_pool=True,
        embedding_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=1000,
        dropout=0.1,
        attention_dropout=0.1,
        stochastic_depth=0.1,
        positional_embedding="sine",
        seq_len=None,
        *args,
        **kwargs,
    ):
        super().__init__()
        positional_embedding = positional_embedding if positional_embedding in ["sine", "learnable", "none"] else "sine"
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.seq_pool = seq_pool

        assert seq_len is not None or positional_embedding == "none", (
            f"Positional embedding is set to {positional_embedding} and" f" the sequence length was not specified."
        )

        if not seq_pool:
            seq_len += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim), requires_grad=True)
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != "none":
            if positional_embedding == "learnable":
                seq_len += 1  # padding idx
                self.positional_emb = Parameter(torch.zeros(1, seq_len, embedding_dim), requires_grad=True)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(seq_len, embedding_dim, padding_idx=True), requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList(
            [
                MaskedTransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path_rate=dpr[i],
                )
                for i in range(num_layers)
            ]
        )
        self.norm = LayerNorm(embedding_dim)

        self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x, mask=None):
        if self.positional_emb is None and x.size(1) < self.seq_len:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode="constant", value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            if mask is not None:
                mask = torch.cat([torch.ones(size=(mask.shape[0], 1), device=mask.device), mask.float()], dim=1)
                mask = mask > 0

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x, mask=mask)
        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]

        x = self.fc(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim, padding_idx=False):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)] for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        pe = pe.unsqueeze(0)
        if padding_idx:
            return torch.cat([torch.zeros((1, 1, dim)), pe], dim=1)
        return pe


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, clusters_num=64, dim=128, normalize_input=True, work_with_tokens=False):
        """
        Args:
            clusters_num : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super().__init__()
        self.clusters_num = clusters_num
        self.dim = dim
        self.alpha = 0
        self.normalize_input = normalize_input
        self.work_with_tokens = work_with_tokens
        if work_with_tokens:
            self.conv = nn.Conv1d(dim, clusters_num, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(dim, clusters_num, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(clusters_num, dim))

    def init_params(self, centroids, descriptors):
        centroids_assign = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        dots = np.dot(centroids_assign, descriptors.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending

        self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
        self.centroids = nn.Parameter(torch.from_numpy(centroids))
        if self.work_with_tokens:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2))
        else:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * centroids_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x):
        if self.work_with_tokens:
            x = x.permute(0, 2, 1)
            N, D, _ = x.shape[:]
        else:
            N, D, H, W = x.shape[:]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # Across descriptor dim
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = torch.zeros([N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device)
        for D in range(self.clusters_num):  # Slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - self.centroids[D : D + 1, :].expand(x_flatten.size(-1), -1, -1).permute(
                1, 2, 0
            ).unsqueeze(0)
            residual = residual * soft_assign[:, D : D + 1, :].unsqueeze(2)
            vlad[:, D : D + 1, :] = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)  # Flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

    def initialize_netvlad_layer(self, args, cluster_ds, backbone):
        descriptors_num = 50000
        descs_num_per_image = 100
        images_num = math.ceil(descriptors_num / descs_num_per_image)
        random_sampler = SubsetRandomSampler(np.random.choice(len(cluster_ds), images_num, replace=False))
        random_dl = DataLoader(dataset=cluster_ds, num_workers=args.num_workers, batch_size=args.infer_batch_size, sampler=random_sampler)
        with torch.no_grad():
            backbone = backbone.eval()
            descriptors = np.zeros(shape=(descriptors_num, args.features_dim), dtype=np.float32)
            for iteration, (inputs, _) in enumerate(tqdm(random_dl, ncols=100)):
                inputs = inputs.to(args.device)
                outputs = backbone(inputs)
                norm_outputs = F.normalize(outputs, p=2, dim=1)
                image_descriptors = norm_outputs.view(norm_outputs.shape[0], args.features_dim, -1).permute(0, 2, 1)
                image_descriptors = image_descriptors.cpu().numpy()
                batchix = iteration * args.infer_batch_size * descs_num_per_image
                for ix in range(image_descriptors.shape[0]):
                    sample = np.random.choice(image_descriptors.shape[1], descs_num_per_image, replace=False)
                    startix = batchix + ix * descs_num_per_image
                    descriptors[startix : startix + descs_num_per_image, :] = image_descriptors[ix, sample, :]
        kmeans = faiss.Kmeans(args.features_dim, self.clusters_num, niter=100, verbose=False)
        kmeans.train(descriptors)
        self.init_params(kmeans.centroids, descriptors)
        self = self.to(args.device)


#################################### Building the Model ##############################################

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(384, antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def cct_8(arch, pretrained, progress, aggregation=None, *args, **kwargs):
    return _cct(arch, pretrained, progress, num_layers=8, num_heads=6, mlp_ratio=3, embedding_dim=384, aggregation=aggregation, *args, **kwargs)


@register_model
def cct_8_7x2_384(
    pretrained=False, progress=False, img_size=384, positional_embedding="learnable", num_classes=1000, aggregation=None, *args, **kwargs
):
    return cct_8(
        "cct_14_7x2_384",
        pretrained,
        progress,
        kernel_size=7,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        aggregation=aggregation,
        *args,
        **kwargs,
    )


class cct384_netvlad(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = cct_8_7x2_384(pretrained=False, progress=True, aggregation="netvlad")
        self.aggregation = NetVLAD(clusters_num=64, dim=384, work_with_tokens=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


class CCT384_NetVLAD(BaseModelWrapper):
    def __init__(self, pretrained: bool = True):
        model = cct384_netvlad()
        if pretrained:
            model.load_state_dict(torch.load(os.path.join(filepath, "weights", "msls_cct384tr8fz1_netvlad.pth")))

        super().__init__(model=model, preprocess=preprocess, name="cct384_netvlad")

        self.model.to(self.device)
        self.model.eval()


model = cct384_netvlad()


sd = torch.load("/Users/olivergrainge/Documents/github/VisualLoc/PlaceRec/Methods/weights/msls_cct384tr8fz1_netvlad.pth", map_location="cpu")[
    "model_state_dict"
]

loaded_sd = sd
model_sd = model.state_dict()
if list(loaded_sd.keys())[0].startswith("module"):
    loaded_sd = OrderedDict({k.replace("module.", ""): v for (k, v) in loaded_sd.items()})

loaded_sd_keys = list(loaded_sd.keys())
model_sd_keys = list(model_sd.keys())


model.load_state_dict(loaded_sd)


image = torch.randn(1, 3, 384, 384)
output = model(image)
print(output.shape)
