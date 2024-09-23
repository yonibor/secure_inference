from typing import Union
import numpy as np
import torch
from torch import nn


class ClusterRelu(nn.Module):
    def __init__(self, C, H, W, prototype=None, inter: Union[float, int, np.ndarray, torch.Tensor]=0):
        super(ClusterRelu, self).__init__()
        self.C, self.H, self.W = C, H, W
        self._prototype = self._inter = None
        self.prototype = prototype
        self.inter = inter

        self.channel_indices = nn.Parameter(torch.arange(C).view(C, 1, 1).repeat(1, H, W),
                                            requires_grad=False) # technical, used for selecting the prototypes
        
    def forward(self, x):
        # Extract row and col indices from prototype
        rows, cols = self.prototype[0], self.prototype[1]

        # Gather prototype values using ellipsis indexing
        prototype_x = x[:, self.channel_indices, rows, cols]

        # Calculate ReLU map and apply it
        relu_map = (x * (1 - self.inter) + prototype_x * self.inter).gt_(0)
        output = x * relu_map

        return output
        
    @property
    def prototype(self) -> torch.Tensor:
        return self._prototype
    
    @prototype.setter
    def prototype(self, new_prototype):
        if new_prototype is not None:
            if isinstance(new_prototype, np.ndarray):
                new_prototype = torch.from_numpy(new_prototype)
            else:
                new_prototype = new_prototype.clone() 
        else:
            new_prototype = _create_default_prototype(C=self.C, H=self.H, W=self.W)
        if self._prototype is None:
            self._prototype = nn.Parameter(new_prototype, requires_grad=False)
        else:
            self._prototype.copy_(new_prototype)

    @property
    def inter(self) -> torch.Tensor:
        return self._inter
    
    @inter.setter
    def inter(self, new_inter):
        if new_inter is None:
            self._inter = new_inter
        else:
            if isinstance(new_inter, np.ndarray):
                new_inter = torch.from_numpy(new_inter)
            elif isinstance(new_inter, torch.Tensor):
                new_inter = new_inter.clone()
            else:
                new_inter = torch.full((self.C, self.H, self.W), new_inter, dtype=torch.float)
            assert torch.all(torch.logical_and(new_inter >= 0, new_inter <= 1)) # TODO: remove at real time
            if self._inter is None:
                self._inter = nn.Parameter(new_inter, requires_grad=False)
            else:
                self._inter.copy_(new_inter)
    
def _create_default_prototype(C, H, W):
    prototype = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    prototype = torch.stack(prototype, dim=0).unsqueeze(1).repeat(1, C, 1, 1)
    return prototype
    
def prototype_from_clusters(C, H, W, channel_clusters={}) -> ClusterRelu:
    prototype = _create_default_prototype(C=C, H=H, W=W)
    for channel, cluster_res in channel_clusters.items():
        cluster_centers_indices = cluster_res.cluster_centers_indices_
        labels = cluster_res.labels_.reshape(H, W)
        for label, cluster_idx in enumerate(cluster_centers_indices):
            label_rows, label_cols = np.nonzero(labels == label)
            center_row = cluster_idx // W 
            center_col = cluster_idx % W
            prototype[0, channel, label_rows, label_cols] = center_row
            prototype[1, channel, label_rows, label_cols] = center_col
    return prototype