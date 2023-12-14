import torch
import torch.nn as nn
from transformers import ViTModel

class NetVLAD(nn.Module):
    def __init__(self, num_clusters, dim):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.clusters = nn.Parameter(torch.randn(num_clusters, dim))
        self.clusters2 = nn.Parameter(torch.randn(1, num_clusters, dim))

    def forward(self, x):
        # x: tensor of shape [batch_size, num_tokens, feature_dim]
        # self.clusters: tensor of shape [num_clusters, feature_dim]
        
        # Expand the clusters to match the batch size and number of tokens
        clusters = self.clusters.unsqueeze(0).unsqueeze(0)
        clusters = clusters.expand(x.size(0), x.size(1), self.num_clusters, self.dim)
        x_expanded = x.unsqueeze(2)
        x_expanded = x_expanded.expand_as(clusters)
        assignment = torch.softmax(torch.sum(x_expanded * clusters, dim=3), dim=2)
        a_sum = assignment.sum(1).unsqueeze(1).unsqueeze(3)
        a = a_sum * self.clusters2
        vlad = torch.sum((x.unsqueeze(2) - a) * assignment.unsqueeze(3), dim=1)
        return vlad.flatten(1)


class ViTNetVLAD(nn.Module):
    def __init__(self, num_clusters):
        super(ViTNetVLAD, self).__init__()
        self.vit_backbone = ViTModel.from_pretrained("google/vit-base-patch16-224")
        dim = self.vit_backbone.config.hidden_size  # Dimension of the token embeddings from ViT
        self.netvlad = NetVLAD(num_clusters, dim)

    def forward(self, x):
        outputs = self.vit_backbone(x)
        tokens = outputs.last_hidden_state
        vlad = self.netvlad(tokens)
        return vlad

# Example usage
model = ViTNetVLAD(num_clusters=64)
input_tensor = torch.randn(2, 3, 224, 224)  # Example input
output = model(input_tensor)
print(output.shape)