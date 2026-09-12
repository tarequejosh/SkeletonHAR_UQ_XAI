import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialGraphConv(nn.Module):
    """
    Spatial Graph Convolution layer (Yan et al., 2018).
    Computes: \sum_{k} (X * (A_k \odot M_k)) W_k
    where A_k is the k-th spatial partition adjacency matrix,
    M_k is learnable edge importance weighting,
    and W_k is the convolutional filter.
    """
    def __init__(self, in_channels, out_channels, num_partitions=3, num_nodes=20):
        super().__init__()
        self.num_partitions = num_partitions
        self.num_nodes = num_nodes
        
        # 1x1 convolution spanning all partitions: (out_channels, in_channels * num_partitions, 1, 1)
        self.conv = nn.Conv2d(
            in_channels * num_partitions,
            out_channels,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            bias=True
        )
        
        # Learnable edge importance mask initialized with ones
        self.edge_importance = nn.Parameter(torch.ones(num_partitions, num_nodes, num_nodes))
        
        # Initialization
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x, A):
        """
        x: (N, C, T, V)
        A: (num_partitions, V, V)
        returns: (N, out_channels, T, V)
        """
        N, C, T, V = x.size()
        
        # Multiply adjacency with learnable edge importance
        A_eff = A * self.edge_importance # (K, V, V)
        
        # Spatial graph convolution across partitions
        # Permute x to (N, T, C, V) for matmul with A_eff
        # Or einsum: x_k = einsum('nctv, kvw -> nktcw', x, A_eff)
        # Efficient formulation:
        # x is (N, C, T, V), we do:
        # for each k: x_k = x @ A_eff[k].T -> (N, C, T, V)
        # Stack over k along channel dim: (N, K*C, T, V)
        out_list = []
        for k in range(self.num_partitions):
            # (N, C*T, V) @ (V, V) -> (N, C*T, V)
            x_k = torch.einsum('nctv, vw -> nctw', x, A_eff[k])
            out_list.append(x_k)
            
        x_cat = torch.cat(out_list, dim=1) # (N, K*C, T, V)
        out = self.conv(x_cat)             # (N, out_channels, T, V)
        return out


class STGCNBlock(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Block:
    Spatial GCN -> BN -> ReLU -> Temporal Conv -> BN -> Dropout -> Residual -> ReLU
    """
    def __init__(self, in_channels, out_channels, num_partitions=3, num_nodes=20, 
                 stride=1, temporal_kernel_size=9, dropout=0.0):
        super().__init__()
        
        # Spatial Graph Conv
        self.sgcn = SpatialGraphConv(in_channels, out_channels, num_partitions, num_nodes)
        self.bn_s = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Temporal Convolution (conv along T axis with padding)
        pad = (temporal_kernel_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel_size, 1),
                padding=(pad, 0),
                stride=(stride, 1),
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )
        
        # Configurable Dropout layer (can be enabled at test time for MC-Dropout)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        
        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x, A):
        """
        x: (N, in_channels, T, V)
        A: (num_partitions, V, V)
        """
        res = self.residual(x)
        x = self.relu(self.bn_s(self.sgcn(x, A)))
        x = self.dropout(self.tcn(x))
        x = self.relu(x + res)
        return x
