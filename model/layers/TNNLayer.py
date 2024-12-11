import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .BaseLayer import sparse_row_norm, HBNS, HBS
from topomodelx.base.aggregation import Aggregation
from model.aggregators import *
import sys

class TNNLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        inout_channels: list[int],
        update_func=F.relu,
        aggr_func="sum",
        initialization="xavier_uniform",
    ):
        super().__init__()

        in_channels_0, in_channels_1, in_channels_2, in_channels_3, in_channels_4 = in_channels
        inout_channels_0, inout_channels_1, inout_channels_2, inout_channels_3, inout_channels_4 = inout_channels

        self.hbs_0_level1 = HBS(
            source_in_channels=in_channels_0,
            source_out_channels=inout_channels_0,
            update_func=update_func,
            aggr_func=aggr_func,
            initialization=initialization,
        )

        self.hbs_1_level1 = HBS(
            source_in_channels=in_channels_1,
            source_out_channels=inout_channels_1,
            update_func=update_func,
            aggr_func=aggr_func,
            initialization=initialization,
        )

        self.hbs_2_level1 = HBS(
            source_in_channels=in_channels_2,
            source_out_channels=inout_channels_2,
            update_func=update_func,
            aggr_func=aggr_func,
            initialization=initialization,
        )

        self.hbs_3_level1 = HBS(
            source_in_channels=in_channels_3,
            source_out_channels=inout_channels_3,
            update_func=update_func,
            aggr_func=aggr_func,
            initialization=initialization,
        )

        self.hbs_4_level1 = HBS(
            source_in_channels=in_channels_4,
            source_out_channels=inout_channels_4,
            update_func=update_func,
            aggr_func=aggr_func,
            initialization=initialization,
        )

        self.hbns_0_1_level1 = HBNS(
            source_in_channels=in_channels_1,
            source_out_channels=inout_channels_1,
            target_in_channels=in_channels_0,
            target_out_channels=inout_channels_0,
            update_func=update_func,
            aggr_func=aggr_func,
            initialization=initialization,
        )

        self.hbns_1_2_level1 = HBNS(
            source_in_channels=in_channels_2,
            source_out_channels=inout_channels_2,
            target_in_channels=in_channels_1,
            target_out_channels=inout_channels_1,
            update_func=update_func,
            aggr_func=aggr_func,
            initialization=initialization,
        )

        self.hbns_2_3_level1 = HBNS(
            source_in_channels=in_channels_3,
            source_out_channels=inout_channels_3,
            target_in_channels=in_channels_2,
            target_out_channels=inout_channels_2,
            update_func=update_func,
            aggr_func=aggr_func,
            initialization=initialization,
        )

        self.hbns_3_4_level1 = HBNS(
            source_in_channels=in_channels_4,
            source_out_channels=inout_channels_4,
            target_in_channels=in_channels_3,
            target_out_channels=inout_channels_3,
            update_func=update_func,
            aggr_func=aggr_func,
            initialization=initialization,
        )

        self.aggr = torch.nn.ModuleList([
            RankAggregator(inout_channels[0], inout_channels[0], update_func=update_func, aggr_func=aggr_func) 
            for _ in range(5)
        ])

    def forward(
        self,
        x_0, x_1, x_2, x_3, x_4,
        adjacency_0, adjacency_1, adjacency_2, adjacency_3, adjacency_4,
        incidence_0_1, incidence_0_2, incidence_0_3, incidence_0_4,
        incidence_1_2, incidence_1_3, incidence_1_4,
        incidence_2_3, incidence_2_4,
        incidence_3_4,

        cci_0_to_0, cci_1_to_1, cci_2_to_2, cci_3_to_3, cci_4_to_4,
        cci_0_to_1, cci_0_to_2, cci_0_to_3, cci_0_to_4,
        cci_1_to_2, cci_1_to_3, cci_1_to_4,
        cci_2_to_3, cci_2_to_4,
        cci_3_to_4,
    ):

        x_0_to_0 = self.hbs_0_level1(x_0, adjacency_0, cci_0_to_0)
        x_1_to_1 = self.hbs_1_level1(x_1, adjacency_1, cci_1_to_1)
        x_2_to_2 = self.hbs_2_level1(x_2, adjacency_2, cci_2_to_2)
        x_3_to_3 = self.hbs_3_level1(x_3, adjacency_3, cci_3_to_3)
        x_4_to_4 = self.hbs_4_level1(x_4, adjacency_4, cci_4_to_4)

        x_0_to_1, x_1_to_0 = self.hbns_0_1_level1(x_1, x_0, incidence_0_1)
        x_1_to_2, x_2_to_1 = self.hbns_1_2_level1(x_2, x_1, incidence_1_2)
        x_2_to_3, x_3_to_2 = self.hbns_2_3_level1(x_3, x_2, incidence_2_3)
        x_3_to_4, x_4_to_3 = self.hbns_3_4_level1(x_4, x_3, incidence_3_4)

        x_0_level1 = self.aggr[0]([x_0_to_0, x_1_to_0])
        x_1_level1 = self.aggr[1]([x_0_to_1, x_1_to_1, x_2_to_1])
        x_2_level1 = self.aggr[2]([x_1_to_2, x_2_to_2, x_3_to_2])
        x_3_level1 = self.aggr[3]([x_2_to_3, x_3_to_3, x_4_to_3])
        x_4_level1 = self.aggr[4]([x_3_to_4, x_4_to_4])
    
        return x_0_level1, x_1_level1, x_2_level1, x_3_level1, x_4_level1
