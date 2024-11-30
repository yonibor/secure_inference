from typing import Callable, Optional, Union

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
        features_amount: Optional[int] = None,
        prototype: Optional[Union[np.ndarray, torch.Tensor]] = None,
        labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        decisions: Optional[Union[np.ndarray, torch.Tensor]] = None,
        inter: Union[float, int, np.ndarray, torch.Tensor] = 0,
        inter_before_activation: bool = True,
        use_cluster_mean: bool = False,
        use_sigmoid: bool = False,
        sigmoid_factor: Optional[Union[float, int]] = None,
        multi_prototype: bool = False,
    ) -> None:
        super(ClusterRelu, self).__init__()
        self.is_dummy = is_dummy
        self.C, self.H, self.W = C, H, W
        self.features_amount = features_amount
        self.use_cluster_mean = use_cluster_mean
        self.use_sigmoid = use_sigmoid
        self.sigmoid_factor = sigmoid_factor
        self.inter_before_activation = inter_before_activation
        self.multi_prototype = multi_prototype

        self.register_buffer("_device_tracker", torch.empty(0))
        self.register_buffer("_prototype", self._format_prototype(prototype))
        self.register_buffer("_labels", self._format_labels(labels))
        self.register_buffer("_decisions", self._format_decisions(decisions))
        self.register_buffer("_inter", self._format_inter(inter))
        self.register_buffer("_crelu_channels", None)
        self.register_buffer("_original_relu_channels", None)
        self.register_buffer("_id_channels", None)
        self.register_buffer("_binary_bases", None)

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
        features_amount: Optional[int] = None,
    ):
        if set_size:
            assert not (C is None or H is None or W is None or features_amount is None)
            self.C, self.H, self.W = C, H, W
            self.features_amount = features_amount
        self.inter = 0
        self.prototype = None
        self.labels = None
        self.decisions = None
        self.original_relu_channels = True
        self.crelu_channels = False
        self.id_channels = False
        self.verify_buffers_init()
        if self.features_amount is not None:
            self._init_binary_bases()

    def verify_buffers_init(self) -> None:
        for name, buffer in self.named_buffers():
            self._verify_not_none(buffer, name)

    def _get_drelu_from_clusters(self, x: torch.Tensor) -> torch.Tensor:
        if self.multi_prototype:
            return self._get_derlu_multi_prototype(x)
        else:
            return self._get_derlu_single_prototype(x)

    def _get_derlu_single_prototype(self, x: torch.Tensor) -> torch.Tensor:
        cluster_values = self._get_cluster_values(x)

        inter_crelu = self.inter[self.crelu_channels]
        x_crelu = x[:, self.crelu_channels]

        if self.inter_before_activation:
            x_inter = x_crelu * (1 - inter_crelu) + cluster_values * inter_crelu
            drelu = self._drelu_activation(x_inter)
        else:
            cluster_drelu = self._drelu_activation(cluster_values)
            x_drelu = self._drelu_activation(x_crelu)
            drelu = x_drelu * (1 - inter_crelu) + cluster_drelu * inter_crelu
            # id_miss = torch.logical_and(x_drelu == 1, cluster_drelu == 0)
            # print(torch.count_nonzero(id_miss) / torch.count_nonzero(x_drelu == 1))
        return drelu

    def _get_derlu_multi_prototype(self, x: torch.Tensor) -> torch.Tensor:
        prototype_x = self._get_cluster_examplar(x)
        prototype_drelu = prototype_x.gt(0).int()
        prototype_index = (prototype_drelu * self.binary_bases).sum(axis=-1)
        cur_decisions = self.decisions[self.crelu_channels]
        cur_decisions_expanded = cur_decisions.expand(
            prototype_index.shape[0], -1, -1, -1, -1
        )
        prototype_index_expanded = prototype_index.unsqueeze(-1)
        decisions_drelu = torch.gather(
            cur_decisions_expanded,
            dim=-1,
            index=prototype_index_expanded,
        ).squeeze(-1)
        x_drelu = x[:, self.crelu_channels].gt(0).int()
        cur_inter = self.inter[self.crelu_channels]
        drelu = x_drelu * (1 - cur_inter) + decisions_drelu * cur_inter

        # counts = torch.unique(
        #     torch.round(decisions_drelu) == x_drelu, return_counts=True
        # )[1]
        # print(counts / counts.sum())
        # id_miss = torch.logical_and(x_drelu == 1, decisions_drelu.round() == 0)
        # print(torch.count_nonzero(id_miss) / torch.count_nonzero(x_drelu == 1))
        # print("-----------------")

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
        active_prototype = self.prototype[:, self.crelu_channels].squeeze(axis=-1)
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
        self._prototype = self._format_prototype(new_prototype)

    @property
    def inter(self) -> torch.Tensor:
        self._verify_not_none(self._inter)
        return self._inter

    @inter.setter
    def inter(self, new_inter: Optional[Union[np.ndarray, float, torch.Tensor]]):
        self._inter = self._format_inter(new_inter)

    @property
    def crelu_channels(self) -> torch.Tensor:
        self._verify_not_none(self._crelu_channels)
        self._verify_channels()
        return self._crelu_channels

    @crelu_channels.setter
    def crelu_channels(
        self, new_channels: Optional[Union[torch.Tensor, np.ndarray, list]]
    ) -> None:
        self._crelu_channels = self._format_channels(new_channels)

    @property
    def original_relu_channels(self) -> torch.Tensor:
        self._verify_not_none(self._original_relu_channels)
        self._verify_channels()
        return self._original_relu_channels

    @original_relu_channels.setter
    def original_relu_channels(
        self, new_channels: Optional[Union[torch.Tensor, np.ndarray, list]]
    ) -> None:
        self._original_relu_channels = self._format_channels(new_channels)

    @property
    def id_channels(self) -> torch.Tensor:
        self._verify_not_none(self._id_channels)
        self._verify_channels()
        return self._id_channels

    @id_channels.setter
    def id_channels(
        self, new_channels: Optional[Union[torch.Tensor, np.ndarray, list]]
    ) -> None:
        self._id_channels = self._format_channels(new_channels)

    @property
    def labels(self) -> torch.Tensor:
        self._verify_not_none(self._labels)
        return self._labels

    @labels.setter
    def labels(self, new_labels: Optional[Union[np.ndarray, torch.Tensor]]) -> None:
        self._labels = self._format_labels(new_labels)

    @property
    def decisions(self) -> torch.Tensor:
        self._verify_not_none(self._decisions)
        return self._decisions

    @decisions.setter
    def decisions(
        self, new_decisions: Optional[Union[np.ndarray, torch.Tensor]]
    ) -> None:
        self._decisions = self._format_decisions(new_decisions)

    def _verify_channels(self):
        channels_combined = torch.stack(
            [self._id_channels, self._original_relu_channels, self._crelu_channels]
        )
        channels_combined = channels_combined.int().sum(dim=0)
        if torch.any(channels_combined > 1):
            raise ValueError("channel on multiple lists")

    @property
    def binary_bases(self):
        if self._binary_bases is None:
            self._init_binary_bases()
        return self._binary_bases

    def _init_binary_bases(self):
        if self.features_amount is None:
            raise ValueError(
                "can't init binary to decimal if features amount is not set"
            )
        bases = 2 ** torch.arange(self.features_amount - 1, -1, -1, dtype=torch.int)
        self._binary_bases = self._set_device(bases)

    def _format_prototype(
        self, prototype: Optional[Union[np.ndarray, torch.Tensor]]
    ) -> Optional[torch.Tensor]:
        return self._format_data(prototype, create_default_prototype)

    def _format_data(
        self,
        data: Optional[Union[np.ndarray, torch.Tensor]],
        default_func: Callable,
        set_device: bool = True,
    ) -> Optional[torch.Tensor]:
        if data is not None:
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            else:
                data = data.clone()
        elif (
            self.C is None
            or self.H is None
            or self.W is None
            or self.features_amount is None
        ):
            return None
        else:
            data = default_func(
                C=self.C, H=self.H, W=self.W, features_amount=self.features_amount
            )
        if set_device:
            data = self._set_device(data)
        return data

    def _format_decisions(
        self, decisions: Optional[Union[np.ndarray, torch.Tensor]]
    ) -> Optional[torch.Tensor]:
        return self._format_data(decisions, create_default_decisions)

    def _format_inter(
        self,
        inter: Union[float, int, np.ndarray, torch.Tensor],
        set_device: bool = True,
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
        if set_device:
            inter = self._set_device(inter)
        return inter

    def _format_channels(
        self,
        channels: Union[torch.Tensor, np.ndarray, list, bool],
        set_device: bool = True,
    ) -> torch.Tensor:
        if isinstance(channels, bool):
            channels = torch.full((self.C,), channels)
        if not isinstance(channels, torch.Tensor):
            channels = torch.tensor(channels, dtype=torch.bool)
        else:
            channels = channels.clone().bool()
        assert channels.numel() == self.C
        if set_device:
            channels = self._set_device(channels)
        return channels

    def _format_labels(
        self, labels: Optional[Union[np.ndarray, torch.Tensor]]
    ) -> torch.Tensor:
        return self._format_data(labels, create_default_labels)

    @staticmethod
    def _verify_not_none(val: Optional[torch.Tensor], name: Optional[str] = None):
        if val is None:
            if name is None:
                name = ""
            message = f"Buffer {name} is None, it wasn't initialized properly"
            raise CReluNotInitError(message)

    def _set_device(self, buffer: torch.Tensor) -> torch.Tensor:
        return buffer.to(self._device_tracker.device)


def create_default_prototype(
    C: int, H: int, W: int, features_amount: int
) -> torch.Tensor:
    prototype = torch.meshgrid(
        torch.arange(C),
        torch.arange(H),
        torch.arange(W),
        indexing="ij",
    )
    prototype = torch.stack(prototype, dim=0)
    prototype = prototype.unsqueeze(-1).repeat(1, 1, 1, 1, features_amount)
    return prototype


def create_default_decisions(
    C: int, H: int, W: int, features_amount: int
) -> torch.Tensor:
    decisions = torch.zeros(size=(C, H, W, 2**features_amount), dtype=torch.float)
    return decisions


def create_default_labels(C: int, H: int, W: int, **kwargs) -> torch.Tensor:
    labels = torch.arange(H * W).view(1, H, W).repeat(C, 1, 1)
    return labels
