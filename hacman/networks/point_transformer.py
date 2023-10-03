import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Identity
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_cluster import fps, knn_graph
from torch_scatter import scatter_max
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.nn.conv import PointTransformerConv
from torch_geometric.nn.pool import knn
from torch_geometric.nn.unpool import knn_interpolate

"""
Point transformer: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/point_transformer_segmentation.py#L63
"""


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], batch_norm=False)

        self.attn_nn = MLP([out_channels, 64, out_channels], batch_norm=False)

        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(torch.nn.Module):
    '''
        Samples the input point cloud by a ratio percentage to reduce
        cardinality and uses an mlp to augment features dimensionnality
    '''
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16, fps_random_start=True):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels])
        self.fps_random_start = fps_random_start

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch, random_start=self.fps_random_start)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out, _ = scatter_max(x[id_k_neighbor[1]], id_k_neighbor[0],
                               dim_size=id_clusters.size(0), dim=0)

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]),
            BN(channels[i]) if batch_norm else Identity(), ReLU())
        for i in range(1, len(channels))
    ])


class TransitionUp(torch.nn.Module):
    '''
        Reduce features dimensionnality and interpolate back to higher
        resolution and cardinality
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp_sub = MLP([in_channels, out_channels])
        self.mlp = MLP([out_channels, out_channels])

    def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
        # transform low-res features and reduce the number of features
        x_sub = self.mlp_sub(x_sub)

        # interpolate low-res feats to high-res points
        x_interpolated = knn_interpolate(x_sub, pos_sub, pos, k=3,
                                         batch_x=batch_sub, batch_y=batch)

        x = self.mlp(x) + x_interpolated

        return x


class PointTransformerSegmentation(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, k=16, fps_random_start=True, 
                 pos_in_feature=False, normalize_pos=False):
        super().__init__()
        self.k = k

        self.normalize_pos = normalize_pos
        self.pos_in_feature = pos_in_feature
        if self.pos_in_feature:
            in_channels += 3
            
        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]])

        self.transformer_input = TransformerBlock(
            in_channels=dim_model[0],
            out_channels=dim_model[0],
        )

        # backbone layers
        self.transformers_up = torch.nn.ModuleList()
        self.transformers_down = torch.nn.ModuleList()
        self.transition_up = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(0, len(dim_model) - 1):

            # Add Transition Down block followed by a Point Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i],
                               out_channels=dim_model[i + 1], k=self.k, fps_random_start=fps_random_start))

            self.transformers_down.append(
                TransformerBlock(in_channels=dim_model[i + 1],
                                 out_channels=dim_model[i + 1]))

            # Add Transition Up block followed by Point Transformer block
            self.transition_up.append(
                TransitionUp(in_channels=dim_model[i + 1],
                             out_channels=dim_model[i]))

            self.transformers_up.append(
                TransformerBlock(in_channels=dim_model[i],
                                 out_channels=dim_model[i]))

        # summit layers
        self.mlp_summit = MLP([dim_model[-1], dim_model[-1]], batch_norm=False)

        self.transformer_summit = TransformerBlock(
            in_channels=dim_model[-1],
            out_channels=dim_model[-1],
        )

        out_channels = [out_channels] if type(out_channels) is not list else out_channels
        
        mlps = []        
        for out_dim in out_channels:
            mlps.append(Seq(Lin(dim_model[0], 128), ReLU(), Lin(128, 128), ReLU(), Lin(128, 128), ReLU(), Lin(128, out_dim)))
        self.mlps = torch.nn.ModuleList(mlps)

    def forward(self, x, pos, batch=None):
        if self.normalize_pos:
            pos = pos.clone()
            mean_pos = global_mean_pool(pos, batch)
            mean_pos_expanded = torch.index_select(mean_pos, 0, batch)
            pos -= mean_pos_expanded
            
            max_pos = global_max_pool(pos.abs(), batch).max(dim=1)[0]
            max_pos_expanded = torch.index_select(max_pos, 0, batch).reshape(-1, 1)
            pos *= (1 / max_pos_expanded) * 0.999999
            
        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1)).to(pos.get_device())

        if self.pos_in_feature:
            x = torch.cat([x, pos], dim=1)

        out_x = []
        out_pos = []
        out_batch = []

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # save outputs for skipping connections
        out_x.append(x)
        out_pos.append(pos)
        out_batch.append(batch)

        # backbone down : #reduce cardinality and augment dimensionnality
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)

        # summit
        x = self.mlp_summit(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_summit(x, pos, edge_index)

        # backbone up : augment cardinality and reduce dimensionnality
        n = len(self.transformers_down)
        for i in range(n):
            x = self.transition_up[-i - 1](x=out_x[-i - 2], x_sub=x,
                                           pos=out_pos[-i - 2],
                                           pos_sub=out_pos[-i - 1],
                                           batch_sub=out_batch[-i - 1],
                                           batch=out_batch[-i - 2])

            edge_index = knn_graph(out_pos[-i - 2], k=self.k,
                                   batch=out_batch[-i - 2])
            x = self.transformers_up[-i - 1](x, out_pos[-i - 2], edge_index)

        if len(self.mlps) == 0:
            return x
        
        out_list = []
        for mlp in self.mlps:
            out_list.append(mlp(x))
        return torch.cat(out_list, dim=1) # F.log_softmax(out, dim=-1)
