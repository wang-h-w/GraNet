"""
    Author: Haowen Wang
"""
import numpy as np
import torch
import dgl
from dgl.geometry import farthest_point_sampler

class GraphGenerator:
    def __init__(self, init_k=64, vision=False):
        self.init_k = init_k
        self.v = vision

    """
        Construct knn graph based on DGL built in method.
        Params:
            pc -> torch.Tensor
            rgb -> torch.Tensor
            reverse: bi-directed graph -> bool
        Returns:
            graph -> DGLGraph
    """
    def build_knn_graph(self, pc, ks, reverse=True):
        graph = dgl.knn_graph(pc, ks)
        if reverse:
            graph = dgl.add_reverse_edges(graph)
        graph.ndata['xyz'] = pc

        return graph

    """
        Initialize knn graph.
        Params:
            data -> dict
        Returns:
            update_data: containing graph -> dict
    """
    def init_knn_graph(self, data: dict):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pc = torch.tensor(data['point_clouds'])

        graph_knn = self.build_knn_graph(pc, self.init_k, False)  # need undirected graph

        g = {'graph': graph_knn}
        data.update(g)

        return data

    """
        Down-sampling graph nodes using Farthest Point Sample(FPS) method built in DGL.
        Params:
            data -> dict
            g -> batched graph
            f: features -> torch.Tensor (B*num_points, feature_dim)
            npoints: num of sample points -> int
            k_iter: index of ks list -> int
        Returns:
            data -> after sampled
    """
    def renew_graph_fps(self, data, g, f, npoints, ks, load_label=True):

        # Get features
        batch_size = len(data['graph'])
        pc_batch = g.ndata['xyz'].view(batch_size, -1, 3)  # (B, num_points, 3)
        f_batch = f.view(batch_size, -1, f.shape[1])  # (B, num_points, feature_dim)

        # FPS according to coordinate
        pc_idx = farthest_point_sampler(pc_batch, npoints)  # (B, npoints)
        # FPS according to feature
        # pc_idx = farthest_point_sampler(f_batch, npoints)  # (B, npoints)

        graphs_new = []
        f_new = []
        xyz_new = []

        for b in range(batch_size):
            pc = pc_batch[b]  # (num_points, 3)
            feature = f_batch[b]

            # Renew graph
            pc_fps = pc[pc_idx[b]]  # (npoints, 3)
            feature_fps = feature[pc_idx[b]]  # (npoints, feature_dim)
            graph_new = self.build_knn_graph(pc_fps, ks, reverse=False)

            xyz_new.append(pc_fps)
            f_new.append(feature_fps)
            graphs_new.append(graph_new)

        if load_label:
            objectness = torch.gather(data['objectness_label'], dim=1, index=pc_idx)
            data['objectness_label'] = objectness
            score = torch.gather(data['objectness_score_gt'], dim=1, index=pc_idx)
            data['objectness_score_gt'] = score

        data['graph'] = graphs_new
        g = dgl.batch(graphs_new)
        feature = torch.stack(f_new, dim=0).view(-1, f.shape[1])  # (B*npoints, feature_dim)
        xyz = torch.stack(xyz_new, dim=0).view(-1, 3)

        return data, g, feature, xyz, pc_idx


def batch_objectness_fps(pos, mask, npoints):
    batch_size, N = mask.size()
    pos = pos.view(batch_size, -1, 3)  # (B, N, 3)
    mask = mask.view(batch_size, -1)  # (B, N)

    indexes = []
    for b in range(batch_size):
        # Select active points
        batch_pos = pos[b]  # (N, 3)
        nonzero_idx = torch.nonzero(mask[b]).squeeze(1)  # (Ns)
        nonzero_idx_ = nonzero_idx.view(-1, 1).expand(-1, 3)  # (Ns, 3)
        selected_pos = torch.gather(batch_pos, 0, nonzero_idx_)  # (Ns, 3)

        if len(nonzero_idx) == 0:
            idxs = np.random.choice(len(mask[b]), npoints, replace=True)
            fps_idx = torch.from_numpy(idxs).view(-1)
        elif npoints <= len(nonzero_idx):
            fps_idx = farthest_point_sampler(selected_pos.unsqueeze(0), npoints).squeeze(0)  # (npoints)
        else:
            idxs1 = np.arange(len(nonzero_idx))
            idxs2 = np.random.choice(len(nonzero_idx), npoints - len(nonzero_idx), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
            fps_idx = torch.from_numpy(idxs).view(-1)
        indexes.append(nonzero_idx[fps_idx])

    indexes = torch.stack(indexes, dim=0)  # (B, npoints)

    return indexes
