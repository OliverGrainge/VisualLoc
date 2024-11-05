import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import (CosineSimilarity,
                                               DotProductSimilarity,
                                               LpDistance)


class MyMultiSimilarity(nn.Module):
    def __init__(self, alpha=1.0, beta=50.0, base=0.0):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.base = base

    def forward(self, descriptors, labels, miner_outputs):
        desc = F.normalize(descriptors, p=2, dim=1)
        S = desc @ desc.T
        pos_exp = self.apply_margin(S, self.base)
        neg_exp = self.apply_margin(self.base, S)

        pos_mask, neg_mask = self.get_masks(S, miner_outputs)
        print(
            miner_outputs[0].shape,
            pos_exp[pos_mask].shape,
            miner_outputs[2].shape,
            neg_exp[neg_mask].shape,
        )
        pos_loss = (1.0 / self.alpha) * self.logsumexp(
            self.alpha * pos_exp, keep_mask=pos_mask.bool(), add_one=True
        )
        neg_loss = (1.0 / self.beta) * self.logsumexp(
            self.beta * neg_exp, keep_mask=neg_mask.bool(), add_one=True
        )
        return torch.mean(pos_loss + neg_loss)

    def get_masks(self, similarity, miner_outputs):
        a1, p, a2, n = miner_outputs
        pos_mask, neg_mask = torch.zeros_like(similarity), torch.zeros_like(similarity)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        return pos_mask.bool(), neg_mask.bool()

    def apply_margin(self, x, margin):
        return margin - x

    def logsumexp(self, x, keep_mask=None, add_one=True, dim=1):
        if keep_mask is not None:
            x = x.masked_fill(~keep_mask, torch.finfo(x.dtype).min)
        if add_one:
            zeros = torch.zeros(
                x.size(dim - 1), dtype=x.dtype, device=x.device
            ).unsqueeze(dim)
            x = torch.cat([x, zeros], dim=dim)

        output = torch.logsumexp(x, dim=dim, keepdim=True)
        if keep_mask is not None:
            output = output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
        return output


def get_loss(loss_name):
    if loss_name == "SupConLoss":
        return losses.SupConLoss(temperature=0.07)
    if loss_name == "CircleLoss":
        return losses.CircleLoss(
            m=0.4, gamma=80
        )  # these are params for image retrieval
    if loss_name == "MultiSimilarityLoss":
        return losses.MultiSimilarityLoss(
            alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity()
        )
    if loss_name == "MyMultiSimilarityLoss":
        return MyMultiSimilarity()
    if loss_name == "ContrastiveLoss":
        return losses.ContrastiveLoss(pos_margin=0, neg_margin=0.5)
    if loss_name == "Lifted":
        return losses.GeneralizedLiftedStructureLoss(
            neg_margin=0, pos_margin=1, distance=DotProductSimilarity()
        )
    if loss_name == "FastAPLoss":
        return losses.FastAPLoss(num_bins=30)
    if loss_name == "NTXentLoss":
        return losses.NTXentLoss(
            temperature=0.07
        )  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
    if loss_name == "TripletMarginLoss":
        return losses.TripletMarginLoss(
            margin=0.1,
            swap=False,
            smooth_loss=False,
            triplets_per_anchor="all",
        )  # or an int, for example 100
    if loss_name == "CentroidTripletLoss":
        return losses.CentroidTripletLoss(
            margin=0.05,
            swap=False,
            smooth_loss=False,
            triplets_per_anchor="all",
        )
    raise NotImplementedError(f"Sorry, <{loss_name}> loss function is not implemented!")


class DummyMiner(nn.Module):
    def forward(self, embeddings, labels):
        return None


def get_miner(miner_name, margin=0.1):
    if miner_name == "TripletMarginMiner":
        return miners.TripletMarginMiner(
            margin=margin, type_of_triplets="semihard"
        )  # all, hard, semihard, easy
    if miner_name == "MultiSimilarityMiner":
        return miners.MultiSimilarityMiner(epsilon=margin, distance=CosineSimilarity())
    if miner_name == "PairMarginMiner":
        return miners.PairMarginMiner(
            pos_margin=0.7, neg_margin=0.3, distance=DotProductSimilarity()
        )
    if miner_name == "none":
        return DummyMiner()
    return None
