import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from torch.utils.data import DataLoader, TensorDataset
import faiss
import numpy as np
import torchvision
from PlaceRec.Training.models import get_backbone, get_aggregator
from typing import Literal
from torchvision import transforms as tvf
import os 
from onedrivedownloader import download
import fast_pytorch_kmeans as fpk
from PIL import Image
import einops as ein
from typing import Union, List, Literal
from torchvision.transforms import functional as T
from PlaceRec.Quantization import quantize_model
from PlaceRec.Training.dataloaders.val.MapillaryDataset import MSLS
from PlaceRec.Training import valid_transform
from torch.utils.data import Subset
from tqdm import tqdm 
from torch.utils.data import SubsetRandomSampler
from torch.hub import load_state_dict_from_url
from torchvision import transforms


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
    if (
        pe_key is not None
        and pe_key in state_dict.keys()
        and pe_key in model.state_dict().keys()
    ):
        if model.state_dict()[pe_key].shape != state_dict[pe_key].shape:
            state_dict[pe_key] = resize_pos_embed(
                state_dict[pe_key],
                model.state_dict()[pe_key],
                num_tokens=model.classifier.num_tokens,
            )
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
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
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

        n_filter_list = (
            [n_input_channels]
            + [in_planes for _ in range(n_conv_layers - 1)]
            + [n_output_channels]
        )

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
                    nn.MaxPool2d(
                        kernel_size=pooling_kernel_size,
                        stride=pooling_stride,
                        padding=pooling_padding,
                    )
                    if max_pool
                    else nn.Identity(),
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


class Attention(nn.Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MaskedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
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


class TransformerEncoderLayer(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        attention_dropout=0.1,
        drop_path_rate=0.1,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = Attention(
            dim=d_model,
            num_heads=nhead,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src

class TransformerClassifier(nn.Module):
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
        positional_embedding = (
            positional_embedding
            if positional_embedding in ["sine", "learnable", "none"]
            else "sine"
        )
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        assert sequence_length is not None or positional_embedding == "none", (
            f"Positional embedding is set to {positional_embedding} and"
            f" the sequence length was not specified."
        )

        if not seq_pool:
            sequence_length += 1
            self.class_emb = nn.Parameter(
                torch.zeros(1, 1, self.embedding_dim), requires_grad=True
            )
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)

        if positional_embedding != "none":
            if positional_embedding == "learnable":
                self.positional_emb = nn.Parameter(
                    torch.zeros(1, sequence_length, embedding_dim), requires_grad=True
                )
                nn.init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = nn.Parameter(
                    self.sinusoidal_embedding(sequence_length, embedding_dim),
                    requires_grad=False,
                )
        else:
            self.positional_emb = None

        self.dropout = nn.Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = nn.ModuleList(
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
        self.norm = nn.LayerNorm(embedding_dim)

        # self.fc = Linear(embedding_dim, num_classes)
        self.apply(self.init_weight)

    def forward(self, x):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(
                x, (0, 0, 0, self.n_channels - x.size(1)), mode="constant", value=0
            )

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
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor(
            [
                [p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                for p in range(n_channels)
            ]
        )
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)

    


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
            sequence_length=self.tokenizer.sequence_length(
                n_channels=n_input_channels, height=img_size, width=img_size
            ),
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
            x = torch.matmul(
                F.softmax(self.classifier.attention_pool(x), dim=1).transpose(-1, -2), x
            ).squeeze(-2)
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





def cct_8(arch, pretrained, progress, aggregation=None, *args, **kwargs):
    return _cct(
        arch,
        pretrained,
        progress,
        num_layers=8,
        num_heads=6,
        mlp_ratio=3,
        embedding_dim=384,
        aggregation=aggregation,
        *args,
        **kwargs,
    )



def cct_8_7x2_384(
    pretrained=False,
    progress=False,
    img_size=384,
    positional_embedding="learnable",
    num_classes=1000,
    aggregation=None,
    *args,
    **kwargs,
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


preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((384, 384), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
