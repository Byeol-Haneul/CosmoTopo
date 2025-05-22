import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from model.aggregators import *
from torch_sparse.tensor import SparseTensor
from torch_sparse.matmul import *

'''    
Notes
-----
- We modify the Higher-order Attention Network (HOAN) from [H23]. 
- Attention mechanisms are removed, and intra-neighborhood aggregation methods are added.
- Cell-Cell E(3)-Invariants (CCI) are added for use.
- This module serve as a base layer for all the TNN layers.

References
----------
.. [H23] Hajij, Zamzmi, Papamarkou, Miolane, Guzmán-Sáenz, Ramamurthy, Birdal, Dey,
    Mukherjee, Samaga, Livesay, Walters, Rosen, Schaub. Topological Deep Learning: Going Beyond Graph Data.
    (2023) https://arxiv.org/abs/2206.00606.

.. [PSHM23] Papillon, Sanborn, Hajij, Miolane.
    Architectures of Topological Deep Learning: A Survey on Topological Neural Networks.
    (2023) https://arxiv.org/abs/2304.10031.

.. [TopoModelX] https://github.com/pyt-team/TopoModelX/blob/main/topomodelx/nn/combinatorial/

'''

def sparse_hadamard(A, B):
    return A*B

def sparse_row_norm(sparse_tensor):
    row_sum = torch.sparse.sum(sparse_tensor, dim=1)
    values = sparse_tensor._values() / row_sum.to_dense()[sparse_tensor._indices()[0]]
    sparse_tensor = torch.sparse_coo_tensor(
        sparse_tensor._indices(), values, sparse_tensor.shape
    )
    return sparse_tensor.coalesce()


class HBNS(torch.nn.Module):
    def __init__(
        self,
        source_in_channels: int,
        source_out_channels: int,
        target_in_channels: int,
        target_out_channels: int,
        update_func=F.relu,
        aggr_func: str = 'sum',
        initialization: str = "xavier_uniform",
    ) -> None:
        super().__init__()

        self.initialization = initialization

        self.source_in_channels, self.source_out_channels = (
            source_in_channels,
            source_out_channels,
        )
        self.target_in_channels, self.target_out_channels = (
            target_in_channels,
            target_out_channels,
        )

        self.activation = update_func

        self.w_s = Parameter(torch.Tensor(self.source_in_channels, self.target_out_channels))
        self.w_t = Parameter(torch.Tensor(self.target_in_channels, self.source_out_channels))

        self.w_s_cci = Parameter(torch.Tensor(self.source_in_channels, self.target_out_channels))
        self.w_t_cci = Parameter(torch.Tensor(self.target_in_channels, self.source_out_channels))

        self.reset_parameters()

        self.source_aggregators = nn.ModuleList([PNAAggregator(source_out_channels, source_out_channels, update_func=self.activation, aggr_func=aggr_func) for _ in range(2)])
        self.target_aggregators = nn.ModuleList([PNAAggregator(target_out_channels, target_out_channels, update_func=self.activation, aggr_func=aggr_func) for _ in range(2)])

        self.layer_norm_source = nn.LayerNorm(source_out_channels)
        self.layer_norm_target = nn.LayerNorm(target_out_channels)

    def get_device(self) -> torch.device:
        return self.w_s.device

    def reset_parameters(self, gain=1.414) -> None:
        if self.initialization == "xavier_uniform":
            for w in [self.w_s, self.w_t, self.w_s_cci, self.w_t_cci]:
                torch.nn.init.xavier_uniform_(w, gain=gain)

        elif self.initialization == "xavier_normal":
            for w in [self.w_s, self.w_t, self.w_s_cci, self.w_t_cci]:
                torch.nn.init.xavier_normal_(w, gain=gain)
        else:
            raise RuntimeError(
                "Initialization method not recognized."
                "Should be either xavier_uniform or xavier_normal."
            )

    def update(
        self, message_on_source: torch.Tensor, message_on_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        message_on_source = self.layer_norm_source(message_on_source)
        message_on_target = self.layer_norm_source(message_on_target)
        return self.activation(message_on_source), self.activation(message_on_target)

    def forward(
        self, x_source: torch.Tensor, x_target: torch.Tensor, neighborhood: SparseTensor, cci = None
    ) -> tuple[torch.Tensor, torch.Tensor]:

        s_message, s_message2  = torch.mm(x_source, self.w_s), torch.mm(x_source, self.w_s_cci)  # [n_source_cells, d_t_out]
        t_message, t_message2  = torch.mm(x_target, self.w_t), torch.mm(x_target, self.w_t_cci)  # [n_target_cells, d_s_out]

        neighborhood_s_to_t = (
            neighborhood.coalesce()
        )  # [n_target_cells, n_source_cells]
        neighborhood_t_to_s = (
            neighborhood.t().coalesce()
        )  # [n_source_cells, n_target_cells]

        # ADD CROSS-CELL INFORMATION
        if cci is not None:
            message_on_source = self.source_aggregators[0](neighborhood_t_to_s, t_message) + self.source_aggregators[1](cci.t(), t_message2)# + self.source_aggregators[2](sparse_hadamard(cci.t(), neighborhood_t_to_s), t_message3) 
            message_on_target = self.target_aggregators[0](neighborhood_s_to_t, s_message) + self.target_aggregators[1](cci, s_message2)# + self.target_aggregators[2](sparse_hadamard(cci, neighborhood_s_to_t), s_message3)
        else:
            message_on_source = self.source_aggregators[0](neighborhood_t_to_s, t_message) 
            message_on_target = self.target_aggregators[0](neighborhood_s_to_t, s_message) 

        return self.update(message_on_source, message_on_target)


class HBS(torch.nn.Module):
    def __init__(
        self,
        source_in_channels: int,
        source_out_channels: int,
        m_hop: int = 1,
        update_func=F.relu,
        aggr_func: str = 'sum',
        initialization: str = "xavier_uniform",
    ) -> None:
        super().__init__()

        self.initialization = initialization

        self.source_in_channels = source_in_channels
        self.source_out_channels = source_out_channels

        self.m_hop = m_hop
        self.activation = update_func

        self.weight = Parameter(torch.Tensor(self.source_in_channels, self.source_out_channels))
        self.weight2 = Parameter(torch.Tensor(self.source_in_channels, self.source_out_channels))

        self.reset_parameters()
        self.source_aggregators = nn.ModuleList([PNAAggregator(source_out_channels, source_out_channels, update_func=self.activation, aggr_func=aggr_func) for _ in range(2)])
        self.layer_norm = nn.LayerNorm(source_out_channels)


    def get_device(self) -> torch.device:
        return self.weight[0].device

    def reset_parameters(self, gain: float = 1.414) -> None:

        def reset_p_hop_parameters(weight, weight2):
            if self.initialization == "xavier_uniform":
                for w in [weight, weight2]:
                    torch.nn.init.xavier_uniform_(w, gain=gain)

            elif self.initialization == "xavier_normal":
                for w in [weight, weight2]:
                    torch.nn.init.xavier_normal_(w, gain=gain)
            else:
                raise RuntimeError(
                    "Initialization method not recognized. "
                    "Should be either xavier_uniform or xavier_normal."
                )

        for w, w2 in zip([self.weight], [self.weight2], strict=True):
            reset_p_hop_parameters(w, w2)

    def update(self, message: torch.Tensor) -> torch.Tensor:
        message = self.activation(message)
        message = self.layer_norm(message)
        return message

    def forward(
        self, x_source: torch.Tensor, neighborhood: SparseTensor, cci = None
    ) -> torch.Tensor:        
        message, message2 = [torch.mm(x_source, w) for w in [self.weight, self.weight2]] 

        if cci is not None:
            message, message2 = [agg(_neighbor, _message) 
                                    for agg, _neighbor, _message in 
                                    zip(self.source_aggregators, [neighborhood, cci], [message, message2])]
        else:
            message = self.source_aggregators[0](neighborhood, message)

        if cci is not None:
            result = message + message2
        else:
            result = message
        
        return self.update(result)