import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineTripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity()

    def forward(self, anchor, positive, negative):
        # Calculate cosine similarity
        positive_similarity = self.cosine_similarity(anchor, positive)
        negative_similarity = self.cosine_similarity(anchor, negative)

        # Since cosine similarity is higher for closer vectors,
        # we subtract it from 1 to follow the conventional triplet loss approach
        # where lower distance values are desired.
        loss = F.relu((1 - positive_similarity) - (1 - negative_similarity) + self.margin)

        return loss.mean()


a = torch.tensor([1.0, 0.0, 0.0])
p = torch.tensor([0.1, 0.9, 0.0])
n = torch.tensor([0.9, 0.1, 0.0])

cosine_similarity = nn.CosineSimilarity()


loss_fn = nn.TripletMarginLoss(margin=0.01)

print(loss_fn(a, p, n))


def cosine_distance(x1, x2):
    # Cosine similarity ranges from -1 to 1, so we add 1 to make it non-negative
    # and then normalize it to range from 0 to 1
    cosine_sim = nn.CosineSimilarity(dim=0)(x1, x2)
    return 1 - cosine_sim


triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=0.01)

print(triplet_loss(a, p, n))
