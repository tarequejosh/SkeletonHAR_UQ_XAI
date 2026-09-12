import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.modules import STGCNBlock
from src.data.skeleton_graph import SkeletonGraph, UTD_EDGES

class STGCN(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network (ST-GCN)
    Reference: Yan et al., AAAI 2018.
    """
    def __init__(
        self,
        in_channels=3,
        num_classes=27,
        num_nodes=20,
        edges=UTD_EDGES,
        graph=None,
        dropout=0.0,
        channel_plan="compact" # "compact" (6 blocks) or "standard" (9 blocks)
    ):
        super().__init__()
        
        # Build skeleton graph
        if graph is None:
            self.graph = SkeletonGraph(num_nodes=num_nodes, edges=edges)
        else:
            self.graph = graph
            
        A = torch.from_numpy(self.graph.A).float() # (3, V, V)
        self.register_buffer('A', A)
        
        # Input batch normalization
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
        
        # Channel configs
        if channel_plan == "compact":
            # 6 ST-GCN blocks tailored for smaller datasets like UTD-MHAD
            configs = [
                # (in_c, out_c, stride)
                (in_channels, 64, 1),
                (64, 64, 1),
                (64, 128, 2),
                (128, 128, 1),
                (128, 256, 2),
                (256, 256, 1)
            ]
        else:
            # Standard 9 ST-GCN blocks
            configs = [
                (in_channels, 64, 1),
                (64, 64, 1),
                (64, 64, 1),
                (64, 64, 1),
                (64, 128, 2),
                (128, 128, 1),
                (128, 128, 1),
                (128, 256, 2),
                (256, 256, 1)
            ]
            
        self.blocks = nn.ModuleList()
        for idx, (in_c, out_c, stride) in enumerate(configs):
            self.blocks.append(
                STGCNBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    num_partitions=self.A.size(0),
                    num_nodes=num_nodes,
                    stride=stride,
                    temporal_kernel_size=9,
                    dropout=dropout
                )
            )
            
        final_channels = configs[-1][1]
        
        # Final classification head
        self.fc = nn.Linear(final_channels, num_classes)
        nn.init.normal_(self.fc.weight, 0, math_scale := (2.0 / num_classes) ** 0.5)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        """
        x: (N, C, T, V)
        returns logits: (N, num_classes)
        """
        N, C, T, V = x.size()
        
        # Normalize input features across channels and nodes
        # Reshape to (N, C*V, T) for BatchNorm1d
        x_bn = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x_bn = self.data_bn(x_bn)
        x = x_bn.view(N, C, V, T).permute(0, 1, 3, 2).contiguous() # back to (N, C, T, V)
        
        # Pass through ST-GCN blocks
        for block in self.blocks:
            x = block(x, self.A)
            
        # Global Average Pooling over time and joints: (N, C_out, T', V) -> (N, C_out)
        x = F.avg_pool2d(x, kernel_size=x.size()[2:]) # (N, C_out, 1, 1)
        x = x.view(N, -1)
        
        # Logits
        logits = self.fc(x)
        return logits

    def enable_mc_dropout(self, active=True):
        """
        Enables dropout layers during evaluation mode for MC-Dropout inference.
        Keeps batchnorm in eval mode while keeping dropout stochastic.
        """
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                if active:
                    m.train()
                else:
                    m.eval()
