from typing import Optional, Union

import numpy as np
import torch
from torch import nn


class CReluNotInitError(Exception):
    pass


class ClusterRelu(nn.Module):
    def __init__(
        self,
        is_dummy: bool,
        C: Optional[int] = None,
        H: Optional[int] = None,
        W: Optional[int] = None,
        prototype: Optional[Union[np.ndarray, torch.Tensor]] = None,
        labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        inter: Union[float, int, np.ndarray, torch.Tensor] = 0,
        inter_before_activation: bool = True,
        use_cluster_mean: bool = False,
        use_sigmoid: bool = False,
        sigmoid_factor: Optional[Union[float, int]] = None,
    ) -> None:
        super(ClusterRelu, self).__init__()
        self.is_dummy = is_dummy
        self.C, self.H, self.W = C, H, W
        self.use_cluster_mean = use_cluster_mean
        self.use_sigmoid = use_sigmoid
        self.sigmoid_factor = sigmoid_factor
        self.inter_before_activation = inter_before_activation

        self.register_buffer("_prototype", self._format_prototype(prototype))
        self.register_buffer("_inter", self._format_inter(inter))
        self.register_buffer("_crelu_channels", None)
        self.register_buffer("_original_relu_channels", None)
        self.register_buffer("_id_channels", None)
        self.register_buffer("_labels", self._format_labels(labels))
        self.register_buffer("_device_tracker", torch.empty(0))

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_dummy:
            return self.relu(x)

        drelu = torch.zeros(size=x.shape, dtype=x.dtype, device=x.device)

        if self.crelu_channels.any():
            drelu[:, self.crelu_channels] = self._get_drelu_from_clusters(x)

        if self.original_relu_channels.any():
            drelu[:, self.original_relu_channels] = self._drelu_activation(
                x[:, self.original_relu_channels]
            )
        if self.id_channels.any():
            drelu[:, self.id_channels] = 1
        return x * drelu

    def set_default_values(
        self,
        set_size: bool,
        C: Optional[int] = None,
        H: Optional[int] = None,
        W: Optional[int] = None,
    ):
        if set_size:
            assert not (C is None or H is None or W is None)
            self.C, self.H, self.W = C, H, W
        self.inter = 0
        self.prototype = None
        self.labels = None
        self.original_relu_channels = True
        self.crelu_channels = False
        self.id_channels = False
        self.verify_buffers_init()

    def verify_buffers_init(self):
        for name, buffer in self.named_buffers():
            self._verify_not_none(buffer, name)

    def _get_drelu_from_clusters(self, x):
        cluster_values = self._get_cluster_values(x)

        inter_crelu = self.inter[self.crelu_channels]
        x_crelu = x[:, self.crelu_channels]

        if self.inter_before_activation:
            x_inter = x_crelu * (1 - inter_crelu) + cluster_values * inter_crelu
            drelu = self._drelu_activation(x_inter)
        else:
            cluster_drelu = self._drelu_activation(cluster_values)
            original_drelu = self._drelu_activation(x_crelu)
            drelu = original_drelu * (1 - inter_crelu) + cluster_drelu * inter_crelu
        return drelu

    def _drelu_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_sigmoid:
            result = self.sigmoid(x * self.sigmoid_factor)
        else:
            result = x.gt(0)
        result = result.to(x.dtype)
        return result

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
        self._verify_not_none(self._prototype)
        return self._prototype

    @prototype.setter
    def prototype(
        self, new_prototype: Optional[Union[np.ndarray, torch.Tensor]]
    ) -> None:
        self._prototype = self._format_prototype(new_prototype).to(
            self._device_tracker.device
        )

    @property
    def inter(self) -> torch.Tensor:
        self._verify_not_none(self._inter)
        return self._inter

    @inter.setter
    def inter(self, new_inter: Optional[Union[np.ndarray, float, torch.Tensor]]):
        self._inter = self._format_inter(new_inter).to(self._device_tracker.device)

    @property
    def crelu_channels(self) -> torch.Tensor:
        self._verify_not_none(self._crelu_channels)
        self._verify_channels()
        return self._crelu_channels

    @crelu_channels.setter
    def crelu_channels(
        self, new_channels: Optional[Union[torch.Tensor, np.ndarray, list]]
    ) -> None:
        self._crelu_channels = self._format_channels(new_channels).to(
            self._device_tracker.device
        )

    @property
    def original_relu_channels(self) -> torch.Tensor:
        self._verify_not_none(self._original_relu_channels)
        self._verify_channels()
        return self._original_relu_channels

    @original_relu_channels.setter
    def original_relu_channels(
        self, new_channels: Optional[Union[torch.Tensor, np.ndarray, list]]
    ) -> None:
        self._original_relu_channels = self._format_channels(new_channels).to(
            self._device_tracker.device
        )

    @property
    def id_channels(self) -> torch.Tensor:
        self._verify_not_none(self._id_channels)
        self._verify_channels()
        return self._id_channels

    @id_channels.setter
    def id_channels(
        self, new_channels: Optional[Union[torch.Tensor, np.ndarray, list]]
    ) -> None:
        self._id_channels = self._format_channels(new_channels).to(
            self._device_tracker.device
        )

    @property
    def labels(self) -> torch.Tensor:
        self._verify_not_none(self._labels)
        return self._labels

    @labels.setter
    def labels(self, new_labels: Optional[Union[np.ndarray, torch.Tensor]]) -> None:
        self._labels = self._format_labels(new_labels).to(self._device_tracker.device)

    def _verify_channels(self):
        channels_combined = torch.stack(
            [self._id_channels, self._original_relu_channels, self._crelu_channels]
        )
        channels_combined = channels_combined.int().sum(dim=0)
        if torch.any(channels_combined > 1):
            raise ValueError("channel on multiple lists")

    def _format_prototype(
        self, prototype: Optional[Union[np.ndarray, torch.Tensor]]
    ) -> Optional[torch.Tensor]:
        if prototype is not None:
            if isinstance(prototype, np.ndarray):
                prototype = torch.from_numpy(prototype)
            else:
                prototype = prototype.clone()
        elif self.C is None or self.H is None or self.W is None:
            return None
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
        elif self.C is None or self.H is None or self.W is None:
            return None
        else:
            inter = torch.full((self.C, self.H, self.W), inter, dtype=torch.float)
        assert torch.all(
            torch.logical_and(inter >= 0, inter <= 1)
        )  # TODO: remove at real time
        return inter

    def _format_channels(
        self, channels: Union[torch.Tensor, np.ndarray, list, bool]
    ) -> torch.Tensor:
        if isinstance(channels, bool):
            channels = torch.full((self.C,), channels)
        if not isinstance(channels, torch.Tensor):
            channels = torch.tensor(channels, dtype=torch.bool)
        else:
            channels = channels.clone().bool()
        assert channels.numel() == self.C
        return channels

    def _format_labels(
        self, labels: Optional[Union[np.ndarray, torch.Tensor]]
    ) -> torch.Tensor:
        if labels is not None:
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)
            else:
                labels = labels.clone()
        elif self.C is None or self.H is None or self.W is None:
            return None
        else:
            labels = create_default_labels(C=self.C, H=self.H, W=self.W)
        return labels

    @staticmethod
    def _verify_not_none(val: Optional[torch.Tensor], name: Optional[str] = None):
        if val is None:
            if name is None:
                name = ""
            message = f"Buffer {name} is None, it wasn't initialized properly"
            raise CReluNotInitError(message)


def create_default_prototype(C: int, H: int, W: int) -> torch.Tensor:
    prototype = torch.meshgrid(
        torch.arange(C), torch.arange(H), torch.arange(W), indexing="ij"
    )
    prototype = torch.stack(prototype, dim=0)
    return prototype


def create_default_labels(C: int, H: int, W: int) -> torch.Tensor:
    labels = torch.arange(H * W).view(1, H, W).repeat(C, 1, 1)
    return labels
