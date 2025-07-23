'''
Author: Jun-Young Lee

Summary: Aggregation classes.

Notes:
- PNAAggregator: Intra-neighborhood Aggregation 
    (i.e., aggregating messages from all cells y1, y2, ... for yi \in N(x))
- RankAggregator: Inter-neighborhood Aggregation 
    (i.e., aggregating messages from all neighborhoods N1(x), N2(x), N3(x), ...)

References
----------
.. Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Liò, Petar Veličković
    Principal Neighbourhood Aggregation for Graph Nets (2020) https://arxiv.org/abs/2004.05718
   - We use a subset of aggregators proposed from Corso, Gabriele, et al. 2020.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from torch_sparse.matmul import *

class PNAAggregator(nn.Module):
    def __init__(self, in_channels, out_channels, aggr_func='sum', scalers=['identity'], update_func=F.relu):
        super(PNAAggregator, self).__init__()
        self.scalers = scalers
        self.activation = update_func

        if aggr_func == 'all':
            self.aggr_func = ['sum', 'max', 'min', 'mean']
            self.mlp = nn.Linear(len(self.aggr_func) * len(scalers) * in_channels, out_channels)
        else:
            self.aggr_func = [aggr_func]

    def forward(self, neighborhood_matrix, features):
        self.neighbor = neighborhood_matrix
        self.features = features

        aggregated_features = []
        for agg in self.aggr_func:
            agg_features = self.aggregate(agg)
            
            for scaler in self.scalers:
                scaled_features = self.scale(agg_features, scaler)
                aggregated_features.append(scaled_features)

        output = torch.cat(aggregated_features, dim=1)

        if len(self.aggr_func) > 1:
            output = self.mlp(output)

        output = self.activation(output)
        return output

    def aggregate(self, aggregation_type):
        if aggregation_type == 'sum':
            return self.sum_aggregation()
        elif aggregation_type == 'max':
            return self.max_aggregation()
        elif aggregation_type == 'min':
            return self.min_aggregation()
        elif aggregation_type == 'mean':
            return self.mean_aggregation()
        elif aggregation_type == 'std':
            return self.std_aggregation()
        else:
            raise ValueError(f"Unsupported aggregation type: {aggregation_type}")

    def sum_aggregation(self):
        return spmm_sum(self.neighbor, self.features)

    def max_aggregation(self):
        return spmm_max(self.neighbor, self.features)[0]

    def min_aggregation(self):
        return spmm_min(self.neighbor, self.features)[0]
    
    def mean_aggregation(self):
        return spmm_mean(self.neighbor, self.features)

    def scale(self, agg_features, scaler):
        if scaler == 'identity':
            return agg_features
        elif scaler == 'amplification':
            raise NotImplementedError
        elif scaler == 'attenuation':
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported scaler type: {scaler}")

class RankAggregator(nn.Module):
    def __init__(self, in_channels, out_channels, aggr_func='sum', update_func=F.relu):
        super(RankAggregator, self).__init__()

        self.activation = update_func

        if aggr_func == 'all':
            self.aggr_func = ['sum', 'max', 'min', 'std', 'mean']
            self.mlp = nn.Linear(len(self.aggr_func) * in_channels, out_channels)
        else:
            self.aggr_func = [aggr_func]

    def forward(self, node_features_list):
        aggregated_features = []
        self.num_features = node_features_list[0].size(1)  

        for agg in self.aggr_func:
            agg_features = self.aggregate(node_features_list, agg)
            aggregated_features.append(agg_features)

        output = torch.cat(aggregated_features, dim=-1)

        if len(self.aggr_func) > 1:
            output = self.mlp(output)
            
        output = self.activation(output)
        return output

    def aggregate(self, node_features_list, aggregation_type):
        if aggregation_type == 'sum':
            return self.sum_aggregation(node_features_list)
        elif aggregation_type == 'max':
            return self.max_aggregation(node_features_list)
        elif aggregation_type == 'min':
            return self.min_aggregation(node_features_list)
        elif aggregation_type == 'std':
            return self.std_aggregation(node_features_list)
        elif aggregation_type == 'mean':
            return self.mean_aggregation(node_features_list)
        else:
            raise ValueError(f"Unsupported aggregation type: {aggregation_type}")

    def sum_aggregation(self, node_features_list):
        return torch.stack(node_features_list).sum(dim=0)

    def max_aggregation(self, node_features_list):
        return torch.stack(node_features_list).max(dim=0)[0]

    def min_aggregation(self, node_features_list):
        return torch.stack(node_features_list).min(dim=0)[0]

    def std_aggregation(self, node_features_list):
        return torch.stack(node_features_list).std(dim=0)

    def mean_aggregation(self, node_features_list):
        return torch.stack(node_features_list).mean(dim=0)
