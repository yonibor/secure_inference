from typing import Optional, Union

import numpy as np
import torch
from torch import nn


class ClusterRelu(nn.Module):
    def __init__(
        self,
        C: int,
        H: int,
        W: int,
        prototype: Optional[Union[np.ndarray, torch.Tensor]] = None,
        labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        inter: Union[float, int, np.ndarray, torch.Tensor] = 0,
        use_cluster_mean: bool = False,
    ) -> None:
        super(ClusterRelu, self).__init__()
        self.C, self.H, self.W = C, H, W
        self.use_cluster_mean = use_cluster_mean
        self.register_buffer("_prototype", self._format_labels(prototype))
        self.register_buffer("_inter", self._format_inter(inter))
        self.register_buffer(
            "_active_channels", self._format_active_channels(torch.arange(self.C))
        )
        self.register_buffer("_labels", self._format_labels(labels))

        self.register_buffer(
            "channel_indices", torch.arange(C).view(C, 1, 1).repeat(1, H, W)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cluster_values = self._get_cluster_values(x)

        # Calculate ReLU map and apply it
        active_inter = self.inter[self.active_channels]
        relu_map = torch.zeros(size=x.shape, dtype=torch.bool, device=x.device)
        x_inter = (
            x[:, self.active_channels] * (1 - active_inter)
            + cluster_values * active_inter
        )
        relu_map[:, self.active_channels] = x_inter.gt(0)
        output = x * relu_map
        return output

    def _get_cluster_values(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cluster_mean:
            return self._get_cluster_mean(x)
        else:
            return self._get_cluster_examplar(x)

    # def _get_cluster_mean(self, x: torch.Tensor) -> torch.Tensor:
    #     clusters_mean = torch.zeros(
    #         (x.shape[0], len(self.active_channels), *x.shape[2:]),
    #         dtype=x.dtype,
    #         device=x.device,
    #     )
    #     labels = self.labels[self.active_channels]
    #     for label in labels.unique():
    #         mask = (labels == label).unsqueeze(dim=0)
    #         label_mean = (x * mask).sum(dim=[2, 3], keepdim=True) / mask.sum(
    #             dim=[2, 3], keepdim=True
    #         )
    #         label_mean = mask * label_mean
    #         clusters_mean += label_mean
    #     return clusters_mean

    def _get_cluster_mean(self, x: torch.Tensor) -> torch.Tensor:
        # Expand the labels tensor along the batch dimension to match x
        x_active = x[:, self.active_channels]
        expanded_labels = (
            self.labels[self.active_channels].unsqueeze(0).expand_as(x_active)
        )

        # Determine the number of unique labels
        num_clusters = expanded_labels.max().item() + 1

        # Sum of values for each cluster
        cluster_sums = torch.zeros(
            (x_active.shape[0], num_clusters, *x_active.shape[2:]),
            dtype=x_active.dtype,
            device=x_active.device,
        ).scatter_add_(1, expanded_labels, x_active)

        # Count occurrences for each cluster
        cluster_counts = torch.zeros_like(cluster_sums).scatter_add_(
            1, expanded_labels, torch.ones_like(x_active)
        )

        # Calculate the mean for each cluster and assign it to the appropriate positions
        mean_per_cluster = (cluster_sums / (cluster_counts + 1e-10)).gather(
            1, expanded_labels
        )

        return mean_per_cluster

    def _get_cluster_examplar(self, x: torch.Tensor) -> torch.Tensor:
        # Extract row and col indices from prototype
        active_prototype = self.prototype[:, self.active_channels]
        rows, cols = active_prototype[0], active_prototype[1]

        # Gather prototype values using ellipsis indexing
        prototype_x = x[:, self.channel_indices[self.active_channels], rows, cols]
        return prototype_x

    @property
    def prototype(self) -> torch.Tensor:
        return self._prototype

    @prototype.setter
    def prototype(
        self, new_prototype: Optional[Union[np.ndarray, torch.Tensor]]
    ) -> None:
        self._prototype = self._format_prototype(new_prototype)

    @property
    def inter(self) -> torch.Tensor:
        return self._inter

    @inter.setter
    def inter(self, new_inter: Optional[Union[np.ndarray, float, torch.Tensor]]):
        self._inter = self._format_inter(new_inter)

    @property
    def active_channels(self) -> torch.Tensor:
        return self._active_channels

    @active_channels.setter
    def active_channels(
        self, new_active_channels: Union[torch.Tensor, np.ndarray, list]
    ) -> None:
        self._active_channels = self._format_active_channels(new_active_channels)

    @property
    def labels(self) -> torch.Tensor:
        return self._labels

    @labels.setter
    def labels(self, new_labels: Optional[Union[np.ndarray, torch.Tensor]]) -> None:
        self._labels = self._format_labels(new_labels)

    def _format_prototype(
        self, prototype: Optional[Union[np.ndarray, torch.Tensor]]
    ) -> torch.Tensor:
        if prototype is not None:
            if isinstance(prototype, np.ndarray):
                prototype = torch.from_numpy(prototype)
            else:
                prototype = prototype.clone()
        else:
            prototype = create_default_prototype(C=self.C, H=self.H, W=self.W)
        return prototype

    def _format_inter(
        self, inter: Union[float, int, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(inter, np.ndarray):
            inter = torch.from_numpy(inter)
        elif isinstance(inter, torch.Tensor):
            inter = inter.clone()
        else:
            inter = torch.full((self.C, self.H, self.W), inter, dtype=torch.float)
        assert torch.all(
            torch.logical_and(inter >= 0, inter <= 1)
        )  # TODO: remove at real time
        return inter

    def _format_active_channels(
        self, active_channels: Union[torch.Tensor, np.ndarray, list]
    ) -> torch.Tensor:
        if not isinstance(active_channels, torch.Tensor):
            active_channels = torch.tensor(active_channels, dtype=torch.long)
        else:
            active_channels = active_channels.long()
        assert torch.all(torch.diff(active_channels) > 0), "need to be sorted"
        return active_channels

    def _format_labels(
        self, labels: Optional[Union[np.ndarray, torch.Tensor]]
    ) -> torch.Tensor:
        if labels is not None:
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)
            else:
                labels = labels.clone()
        else:
            labels = create_default_labels(C=self.C, H=self.H, W=self.W)
        return labels


def create_default_prototype(C: int, H: int, W: int) -> torch.Tensor:
    prototype = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    prototype = torch.stack(prototype, dim=0).unsqueeze(1).repeat(1, C, 1, 1)
    return prototype


def create_default_labels(C: int, H: int, W: int) -> torch.Tensor:
    labels = torch.arange(H * W).view(1, H, W).repeat(C, 1, 1)
    return labels
