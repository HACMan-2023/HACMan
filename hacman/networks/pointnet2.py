import torch
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius, global_mean_pool
from torch_geometric.nn import MLP, knn_interpolate

import numpy as np


def normalize_pcd(pos, batch, normalize_pos_param):
    pos = pos.clone()
    if normalize_pos_param is None:
        mean_pos = global_mean_pool(pos, batch)
        mean_pos_expanded = torch.index_select(mean_pos, 0, batch)
        pos -= mean_pos_expanded
        
        max_pos = global_max_pool(pos.abs(), batch).max(dim=1)[0]
        max_pos_expanded = torch.index_select(max_pos, 0, batch).reshape(-1, 1)
        pos *= (1 / max_pos_expanded) * 0.999999
    
    else:
        offset, full_scale = normalize_pos_param
        offset = torch.from_numpy(offset).float().to(pos.device)
        full_scale = torch.from_numpy(full_scale).float().to(pos.device)
        
        pos -= offset
        pos /= torch.max(full_scale/2)
    
    return pos

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn, fps_random_start=True):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)
        self.fps_random_start = fps_random_start

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio, random_start=self.fps_random_start)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointNet2Segmentation(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0, fps_random_start=True, 
                 pos_in_feature=False, normalize_pos=False, normalize_pos_param=None):
        super().__init__()
        
        self.normalize_pos = normalize_pos
        self.normalize_pos_param = normalize_pos_param
        self.pos_in_feature = pos_in_feature
        if self.pos_in_feature:
            in_channels += 3
        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.2, 0.2, MLP([in_channels + 3, 64, 64, 128]), fps_random_start=fps_random_start)
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]), fps_random_start=fps_random_start)
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + in_channels, 128, 128, 128]))

        out_channels = [out_channels] if type(out_channels) is not list else out_channels
        mlps = []        
        for out_dim in out_channels:
            mlps.append(MLP([128, 128, 128, out_dim], dropout=dropout, batch_norm=False))
        self.mlps = torch.nn.ModuleList(mlps)

    def forward(self, x, pos, batch=None):
        if self.normalize_pos:
            pos = normalize_pcd(pos, batch, self.normalize_pos_param)
            
        if self.pos_in_feature:
            x = torch.cat([x, pos], dim=1)

        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        if len(self.mlps) == 0:
            # Just use a feature extractor
            out_list = [x]
        else:            
            out_list = []
            for mlp in self.mlps:
                out_list.append(mlp(x))
        return torch.cat(out_list, dim=1) # F.log_softmax(out, dim=-1)

class PointNet2Classification(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0, fps_random_start=True, pos_in_feature=False, normalize_pos=False, 
                normalize_pos_param=None, mode=None):
        super().__init__()
        self.pos_in_feature = pos_in_feature
        if self.pos_in_feature:
            in_channels += 3
        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([in_channels + 3, 64, 64, 128]), fps_random_start=fps_random_start)
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]), fps_random_start=fps_random_start)
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))
        self.normalize_pos = normalize_pos
        self.normalize_pos_param = normalize_pos_param

        
        
        out_channels = [out_channels] if type(out_channels) is not list else out_channels
        mlps = []        
        for out_dim in out_channels:
            mlps.append(MLP([1024, 512, 256, out_dim], dropout=dropout, batch_norm=False))
        self.mlps = torch.nn.ModuleList(mlps)
        self.mode = mode

    def forward(self, x, pos, batch=None):
        if self.normalize_pos:
            pos = normalize_pcd(pos, batch, self.normalize_pos_param)
            
        if self.pos_in_feature:
            x = torch.cat([x, pos], dim=1)
        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        out_list = []
        
        if len(self.mlps) == 0:
            return x
        
        else:
            if self.mode == "binary_classification":
                for mlp in self.mlps:
                    out_list.append(torch.nn.functional.sigmoid(mlp(x)))
            else:
                for mlp in self.mlps:
                    out_list.append(mlp(x))
            return torch.cat(out_list, dim=1) # self.mlp(x).log_softmax(dim=-1)
        

if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    seg_model = PointNet2Segmentation(3, [])
    seg_count = count_parameters(seg_model)
    # 1381760 + 49152 = 1430912
    
    cls_model = PointNet2Classification(3, [])
    cls_count = count_parameters(cls_model)
    # 805120 + 524288 + 131072 = 1460480
    
    print(seg_count, cls_count)