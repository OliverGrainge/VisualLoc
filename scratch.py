import time

import torch
import torch_pruning as tp

from PlaceRec.utils import get_method
import torch
from torch.nn.utils import prune
import torch.nn as nn


class NMPruningMethod(prune.BasePruningMethod):
    """Prunes (zeros out) the weights that are closest to the median of the absolute values."""
    def __init__(self, N: int=2, M: int=4):
        self.N = N
        self.M = M
        super(NMPruningMethod, self).__init__()
    
    PRUNING_TYPE = 'unstructured'
    
    def compute_mask(self, t, default_mask):
        """
        Compute the mask of weights to prune.
        
        Args:
            t (torch.Tensor): The tensor to prune.
            default_mask (torch.Tensor): The default (previous) mask.
        
        Returns:
            torch.Tensor: The updated mask.
        """
        """
        Compute the mask of weights to prune using N:M sparsity.
        
        Args:
            t (torch.Tensor): The tensor to prune.
            default_mask (torch.Tensor): The default (previous) mask.
        
        Returns:
            torch.Tensor: The updated mask.
        """
        # Ensure t is a flattened version of the original tensor
        flat_tensor = t.flatten()
        num_elements = flat_tensor.size(0)
        
        # Calculate the number of blocks
        num_blocks = num_elements // self.M
        
        # Create a mask of ones
        mask = torch.ones(num_elements, dtype=torch.float32, device=t.device)
        # Apply N:M sparsity
        for i in range(num_blocks):
            block_start = i * self.M
            block_end = block_start + self.M
            block = flat_tensor[block_start:block_end]
            
            # Find the indices of the N smallest elements in the block
            _, indices_to_prune = torch.topk(block.abs(), self.N, largest=False)
            
            # Zero out the corresponding positions in the mask
            mask[block_start:block_end][indices_to_prune] = 0
        
        # Reshape the mask back to the shape of t
        mask = mask.reshape(t.shape)
        return mask



def apply_NM_model(model):
    for module in model.modules():
    # Check if the module is a convolutional or linear layer
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Apply custom pruning to the 'weight' parameter
            NMPruningMethod.apply(module, "weight")
            # If you also want to prune biases or other parameters, add similar lines here
    return model


method = get_method("resnet50_eigenplaces", pretrained=True)

model = method.model
model.eval()
model.to("cpu")

model = apply_NM_model(model)

