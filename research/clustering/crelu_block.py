from typing import Union
import numpy as np
import torch
from torch import nn


class ClusterRelu(nn.Module):
    def __init__(self, C, H, W, prototype=None, inter: Union[float, int, np.ndarray, torch.Tensor]=0):
        super(ClusterRelu, self).__init__()
        self.C, self.H, self.W = C, H, W
        self._prototype = self._inter = self._active_channels = None
        self.prototype = prototype
        self.inter = inter
        self.active_channels = torch.arange(C)
        
        self.channel_indices = nn.Parameter(torch.arange(C).view(C, 1, 1).repeat(1, H, W),
                                            requires_grad=False) # technical, used for selecting the prototypes
        
    def forward(self, x):
        not_active_channels = [c for c in range(x.shape[1]) if c not in self.active_channels]
        not_active_x = x[:, not_active_channels]
        pos_not_active_x = not_active_x.gt(0)
        # Extract row and col indices from prototype
        active_prototype = self.prototype[:, self.active_channels]
        rows, cols = active_prototype[0], active_prototype[1]

        # Gather prototype values using ellipsis indexing
        prototype_x = x[:, self.channel_indices[self.active_channels], rows, cols]

        # Calculate ReLU map and apply it
        active_inter = self.inter[self.active_channels]
        relu_map = torch.zeros(size=x.shape, dtype=torch.bool, device=x.device)
        x_inter = x[:, self.active_channels] * (1 - active_inter) + prototype_x * active_inter
        relu_map[:, self.active_channels] = x_inter.gt(0)
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
            new_prototype = create_default_prototype(C=self.C, H=self.H, W=self.W)
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
    
    @property
    def active_channels(self):
        return self._active_channels
    
    @active_channels.setter
    def active_channels(self, new_channels):
        if not isinstance(new_channels, torch.Tensor):
            new_channels = torch.tensor(new_channels, dtype=torch.long)
        else:
            new_channels = new_channels.long()
        if self._active_channels is not None:
            new_channels = new_channels.to(self._active_channels.device)
        assert torch.all(torch.diff(new_channels) > 0), 'need to be sorted'
        self._active_channels = nn.Parameter(new_channels, requires_grad=False)
    
def create_default_prototype(C, H, W):
    prototype = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    prototype = torch.stack(prototype, dim=0).unsqueeze(1).repeat(1, C, 1, 1)
    return prototype