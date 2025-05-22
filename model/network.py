import torch
from torch import nn

from .layers import *

def get_activation(update_func):
    if update_func == "sigmoid":
        return torch.sigmoid
    elif  update_func == "relu":
        return torch.nn.functional.relu
    elif update_func == "tanh":
        return torch.nn.functional.tanh
    else:
        raise NotImplementedError

class Network(nn.Module):
    def __init__(self, layerType, channels_per_layer, final_output_layer, cci_mode: str, update_func: str, aggr_func: str, residual_flag: bool = True, loss_fn_name: str = "log_implicit_likelihood"):
        super().__init__()
        
        self.layerType = layerType
        self.cci_mode = cci_mode
        self.activation = get_activation(update_func)
        self.base_model = CustomHMC(layerType, channels_per_layer, update_func=self.activation, aggr_func=aggr_func, residual_flag=residual_flag)   
        self.loss_fn_name = loss_fn_name

        penultimate_layer = channels_per_layer[-1][-1][0]
        num_aggregators = 4

        if self.layerType == "TNN":
            num_ranks_pooling = 5
        else:
            num_ranks_pooling = 1
        
        # Global feature size: x_0.shape[0], x_1.shape[0], x_2.shape[0], x_3.shape[0]
        if self.layerType == "GNN":
            self.global_feature_size = 4
        else:
            self.global_feature_size = 4 

        # Fully-Connected Layers
        self.fc1 = nn.Linear(penultimate_layer * num_ranks_pooling * num_aggregators + self.global_feature_size, 512)           
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, final_output_layer)

    def forward(self, batch) -> torch.Tensor:
        # Features
        x_0 = batch['x_0']
        x_1 = batch['x_1']
        x_2 = batch['x_2']
        x_3 = batch['x_3']
        x_4 = batch['x_4']

        # (Co)Adjacency
        n0_to_0 = batch['n0_to_0']
        n1_to_1 = batch['n1_to_1']
        n2_to_2 = batch['n2_to_2']
        n3_to_3 = batch['n3_to_3']
        n4_to_4 = batch['n4_to_4']

        # Incidence
        n0_to_1 = batch['n0_to_1']
        n0_to_2 = batch['n0_to_2']
        n0_to_3 = batch['n0_to_3']
        n0_to_4 = batch['n0_to_4']
        n1_to_2 = batch['n1_to_2']
        n1_to_3 = batch['n1_to_3']
        n1_to_4 = batch['n1_to_4']
        n2_to_3 = batch['n2_to_3']
        n2_to_4 = batch['n2_to_4']
        n3_to_4 = batch['n3_to_4']

        keys = [f'{self.cci_mode}_{i}_to_{j}' for i in range(5) for j in range(i, 5)]
        values = [batch[key] if self.cci_mode != 'None' else None for key in keys]
        cci0_to_0, cci0_to_1, cci0_to_2, cci0_to_3, cci0_to_4, cci1_to_1, cci1_to_2, cci1_to_3, cci1_to_4, cci2_to_2, cci2_to_3, cci2_to_4, cci3_to_3, cci3_to_4, cci4_to_4 = values

        # Global Feature
        global_feature = batch['global_feature'][:, :self.global_feature_size]
       
        # Forward pass through the base model
        x_0, x_1, x_2, x_3, x_4 = self.base_model(
            x_0, x_1, x_2, x_3, x_4, 

            n0_to_0, n1_to_1, n2_to_2, n3_to_3, n4_to_4,
            n0_to_1, n0_to_2, n0_to_3, n0_to_4,
            n1_to_2, n1_to_3, n1_to_4,
            n2_to_3, n2_to_4,
            n3_to_4, 

            cci0_to_0, cci1_to_1, cci2_to_2, cci3_to_3, cci4_to_4,
            cci0_to_1, cci0_to_2, cci0_to_3, cci0_to_4,
            cci1_to_2, cci1_to_3, cci1_to_4,
            cci2_to_3, cci2_to_4,
            cci3_to_4
        )

        def global_aggregations(x):
            variance = torch.var(x, dim=0, unbiased=False)
            variance = torch.where(variance == 0, torch.tensor(1e-6, device=x.device), variance) # For numerical stability. See             
            x_std = torch.sqrt(variance).unsqueeze(0)
            x_avg = torch.mean(x, dim=0, keepdim=True)
            x_max, _ = torch.max(x, dim=0, keepdim=True)
            x_min, _ = torch.min(x, dim=0, keepdim=True)
            x = torch.cat((x_avg, x_std, x_max, x_min), dim=1)
            return x
        
        x_0 = global_aggregations(x_0)
        x_1 = global_aggregations(x_1)
        x_2 = global_aggregations(x_2)
        x_3 = global_aggregations(x_3)
        x_4 = global_aggregations(x_4)
       
        if "GNN" in self.layerType or self.layerType == "TetraTNN":
            x = torch.cat((x_0, global_feature), dim=1)
        elif self.layerType == "ClusterTNN":
            x = torch.cat((x_3, global_feature), dim=1)
        elif self.layerType == "TNN":
            x = torch.cat((x_0, x_1, x_2, x_3, x_4, global_feature), dim=1)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        x = self.activation(x)

        x = self.fc4(x)

        if self.loss_fn_name == "mse":
            return x
        else:
            x1 = x[:, :x.shape[1] // 2]
            x2 = torch.square(x[:, x.shape[1] // 2:])
            x = torch.cat([x1, x2], dim=1)
            return x


class CustomHMC(torch.nn.Module):
    def __init__(
        self,
        layerType,
        channels_per_layer,
        update_func=torch.nn.functional.relu,
        aggr_func='sum',
        residual_flag: bool = True
    ) -> None:
        def check_channels_consistency():
            assert len(channels_per_layer) > 0
            for i in range(len(channels_per_layer) - 1):
                assert channels_per_layer[i][1][0] == channels_per_layer[i + 1][0][0]
                assert channels_per_layer[i][1][1] == channels_per_layer[i + 1][0][1]
                assert channels_per_layer[i][1][2] == channels_per_layer[i + 1][0][2]
                assert channels_per_layer[i][1][3] == channels_per_layer[i + 1][0][3]
                assert channels_per_layer[i][1][4] == channels_per_layer[i + 1][0][4]

        super().__init__()
        check_channels_consistency()

        if layerType == "GNN":
            self.base_layer = GNNLayer
        elif layerType == "SimpleGNN":
            self.base_layer = SimpleGNNLayer
        elif layerType == "TetraTNN":
            self.base_layer = TetraTNNLayer
        elif layerType == "ClusterTNN":
            self.base_layer = ClusterTNNLayer
        elif layerType == "TNN":
            self.base_layer = TNNLayer
        else:
            raise Exception("Invalid Model Type. Current Available Options are [Hier, Normal]")

        self.residual_flag = residual_flag
        self.activation = update_func
        self.layers = torch.nn.ModuleList(
            [
                self.base_layer(
                    in_channels=in_channels,
                    inout_channels=inout_channels,
                    update_func=update_func,
                    aggr_func=aggr_func
                )
                for in_channels, inout_channels in channels_per_layer
            ]
        )

    def forward(
        self,
        x_0, x_1, x_2, x_3, x_4,
        neighborhood_0_to_0, neighborhood_1_to_1, neighborhood_2_to_2, neighborhood_3_to_3, neighborhood_4_to_4,
        neighborhood_0_to_1, neighborhood_0_to_2, neighborhood_0_to_3, neighborhood_0_to_4,
        neighborhood_1_to_2, neighborhood_1_to_3, neighborhood_1_to_4,
        neighborhood_2_to_3, neighborhood_2_to_4,
        neighborhood_3_to_4,

        cci_0_to_0, cci_1_to_1, cci_2_to_2, cci_3_to_3, cci_4_to_4,
        cci_0_to_1, cci_0_to_2, cci_0_to_3, cci_0_to_4,
        cci_1_to_2, cci_1_to_3, cci_1_to_4,
        cci_2_to_3, cci_2_to_4,
        cci_3_to_4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for layer_num, layer in enumerate(self.layers):

            h_0, h_1, h_2, h_3, h_4 = layer(
                x_0, x_1, x_2, x_3, x_4,

                neighborhood_0_to_0, neighborhood_1_to_1, neighborhood_2_to_2, neighborhood_3_to_3, neighborhood_4_to_4,
                neighborhood_0_to_1, neighborhood_0_to_2, neighborhood_0_to_3, neighborhood_0_to_4,
                neighborhood_1_to_2, neighborhood_1_to_3, neighborhood_1_to_4,
                neighborhood_2_to_3, neighborhood_2_to_4,
                neighborhood_3_to_4,

                cci_0_to_0, cci_1_to_1, cci_2_to_2, cci_3_to_3, cci_4_to_4,
                cci_0_to_1, cci_0_to_2, cci_0_to_3, cci_0_to_4,
                cci_1_to_2, cci_1_to_3, cci_1_to_4,
                cci_2_to_3, cci_2_to_4,
                cci_3_to_4,
            )

            residual_condition = self.residual_flag and layer_num > 1

            x_0 = h_0 + x_0 if residual_condition else h_0
            x_1 = h_1 + x_1 if residual_condition else h_1
            x_2 = h_2 + x_2 if residual_condition else h_2
            x_3 = h_3 + x_3 if residual_condition else h_3
            x_4 = h_4 + x_4 if residual_condition else h_4

            x_0 = self.activation(x_0)
            x_1 = self.activation(x_1)
            x_2 = self.activation(x_2)
            x_3 = self.activation(x_3)
            x_4 = self.activation(x_4)

        return x_0, x_1, x_2, x_3, x_4
