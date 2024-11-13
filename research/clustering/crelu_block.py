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
        self.register_buffer("_prototype", self._format_prototype(prototype))
        self.register_buffer("_inter", self._format_inter(inter))
        self.register_buffer(
            "_crelu_channels", self._format_channels(torch.arange(self.C))
        )
        self.register_buffer("_original_relu_channels", self._format_channels(None))
        self.register_buffer("_labels", self._format_labels(labels))

        self.register_buffer(
            "channel_indices", torch.arange(C).view(C, 1, 1).repeat(1, H, W)
        )  # helper for slicing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        relu_map = torch.zeros(size=x.shape, dtype=torch.bool, device=x.device)
        if self.crelu_channels.numel() > 0:
            cluster_values = self._get_cluster_values(x)

            # Calculate ReLU map and apply it
            active_inter = self.inter[self.crelu_channels]

            x_inter = (
                x[:, self.crelu_channels] * (1 - active_inter)
                + cluster_values * active_inter
            )
            relu_map[:, self.crelu_channels] = x_inter.gt(0)
        if self.original_relu_channels.numel() > 0:
            relu_map[:, self._original_relu_channels] = x[
                :, self._original_relu_channels
            ].gt(0)

        output = x * relu_map
        return output

    def _get_cluster_values(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cluster_mean:
            return self._get_cluster_mean(x)
        else:
            return self._get_cluster_examplar(x)

    def _get_cluster_mean(self, x: torch.Tensor) -> torch.Tensor:
        x_active = x[:, self.crelu_channels]
        clusters_mean = torch.zeros(
            x_active.shape,
            dtype=x_active.dtype,
            device=x_active.device,
        )
        labels = self.labels[self.crelu_channels]
        for label in labels.unique():
            mask = (labels == label).unsqueeze(dim=0)
            label_mean = (x_active * mask).sum(dim=[2, 3], keepdim=True) / mask.sum(
                dim=[2, 3], keepdim=True
            )
            label_mean = mask * label_mean
            clusters_mean += label_mean
        return clusters_mean

    # def _get_cluster_mean(self, x: torch.Tensor) -> torch.Tensor:
    #     # Get labels for active channels and add batch dimension
    #     x_active = x[:, self.crelu_channels]
    #     channel_labels = self.labels[self.crelu_channels]  # Shape: (C, H, W)
    #     num_labels = int(
    #         channel_labels.max().item() + 1
    #     )  # Total number of unique labels

    #     # Create one-hot encoding for labels using scatter
    #     one_hot_labels = torch.zeros(
    #         (num_labels, *channel_labels.shape), device=x_active.device
    #     )
    #     one_hot_labels.scatter_(
    #         0, channel_labels.long().unsqueeze(0), 1
    #     )  # Shape: (num_labels, 1, C, H, W)

    #     # Compute the sum and count of elements for each label
    #     masked_sums = (x_active.unsqueeze(1) * one_hot_labels.unsqueeze(0)).sum(
    #         dim=[-2, -1], keepdim=True
    #     )
    #     label_counts = one_hot_labels.sum(dim=[-2, -1], keepdim=True)

    #     # Avoid division by zero by adding a small epsilon
    #     mean_values = masked_sums / (label_counts + 1e-10)

    #     # Combine results to get cluster means per label
    #     cluster_means = (one_hot_labels * mean_values).sum(dim=1)

    #     return cluster_means

    # def _get_cluster_mean(self, x: torch.Tensor) -> torch.Tensor:
    #     # Expand the labels tensor along the batch dimension to match x
    #     x_active = x[:, self.crelu_channels]
    #     expanded_labels = (
    #         self.labels[self.crelu_channels].unsqueeze(0).expand_as(x_active)
    #     )

    #     # Determine the number of unique labels
    #     num_clusters = expanded_labels.max().item() + 1

    #     # Sum of values for each cluster
    #     cluster_sums = torch.zeros(
    #         (x_active.shape[0], num_clusters, *x_active.shape[2:]),
    #         dtype=x_active.dtype,
    #         device=x_active.device,
    #     ).scatter_add_(1, expanded_labels, x_active)

    #     # Count occurrences for each cluster
    #     cluster_counts = torch.zeros_like(cluster_sums).scatter_add_(
    #         1, expanded_labels, torch.ones_like(x_active)
    #     )

    #     # Calculate the mean for each cluster and assign it to the appropriate positions
    #     mean_per_cluster = (cluster_sums / (cluster_counts + 1e-10)).gather(
    #         1, expanded_labels
    #     )

    #     return mean_per_cluster

    def _get_cluster_examplar(self, x: torch.Tensor) -> torch.Tensor:
        # Extract row and col indices from prototype
        active_prototype = self.prototype[:, self.crelu_channels]
        channels, rows, cols = (
            active_prototype[0],
            active_prototype[1],
            active_prototype[2],
        )

        # Gather prototype values using ellipsis indexing
        prototype_x = x[:, channels, rows, cols]
        return prototype_x

    @property
    def prototype(self) -> torch.Tensor:
        return self._prototype

    @prototype.setter
    def prototype(
        self, new_prototype: Optional[Union[np.ndarray, torch.Tensor]]
    ) -> None:
        self._prototype = self._format_prototype(new_prototype).to(
            self._prototype.device
        )

    @property
    def inter(self) -> torch.Tensor:
        return self._inter

    @inter.setter
    def inter(self, new_inter: Optional[Union[np.ndarray, float, torch.Tensor]]):
        self._inter = self._format_inter(new_inter).to(self._inter.device)

    @property
    def crelu_channels(self) -> torch.Tensor:
        assert not torch.isin(self._crelu_channels, self._original_relu_channels).any()
        return self._crelu_channels

    @crelu_channels.setter
    def crelu_channels(
        self, new_channels: Optional[Union[torch.Tensor, np.ndarray, list]]
    ) -> None:
        self._crelu_channels = self._format_channels(new_channels).to(
            self._crelu_channels.device
        )

    @property
    def original_relu_channels(self) -> torch.Tensor:
        assert not torch.isin(self.crelu_channels, self._original_relu_channels).any()
        return self._original_relu_channels

    @original_relu_channels.setter
    def original_relu_channels(
        self, new_channels: Optional[Union[torch.Tensor, np.ndarray, list]]
    ) -> None:
        self._original_relu_channels = self._format_channels(new_channels).to(
            self._original_relu_channels.device
        )

    @property
    def labels(self) -> torch.Tensor:
        return self._labels

    @labels.setter
    def labels(self, new_labels: Optional[Union[np.ndarray, torch.Tensor]]) -> None:
        self._labels = self._format_labels(new_labels).to(self._labels.device)

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

    def _format_channels(
        self, channels: Optional[Union[torch.Tensor, np.ndarray, list]]
    ) -> torch.Tensor:
        if channels is None:
            channels = []
        if not isinstance(channels, torch.Tensor):
            channels = torch.tensor(channels, dtype=torch.long)
        else:
            channels = channels.clone().long()
        assert torch.all(torch.diff(channels) > 0), "need to be sorted"
        return channels

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
    prototype = torch.meshgrid(
        torch.arange(C), torch.arange(H), torch.arange(W), indexing="ij"
    )
    prototype = torch.stack(prototype, dim=0)
    return prototype


def create_default_labels(C: int, H: int, W: int) -> torch.Tensor:
    labels = torch.arange(H * W).view(1, H, W).repeat(C, 1, 1)
    return labels
