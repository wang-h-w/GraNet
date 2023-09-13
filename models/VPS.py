"""
    Author: Haowen Wang
"""

import os
import sys
import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch.conv import EdgeConv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from pointnet2_utils import BallQuery

class VPS(nn.Module):
    def __init__(self, k, in_dim, out_dim, neighbor_radius, nsample=32, load_label=False, labels=None, cls=True):

        super(VPS, self).__init__()

        # Networks
        self.region_score_graph = nn.ModuleList()
        self.region_score_graph_bn = nn.ModuleList()
        for i in range(1):
            self.region_score_graph.append(EdgeConv(in_dim, in_dim))
            self.region_score_graph_bn.append(nn.BatchNorm1d(in_dim, in_dim))
        self.region_score_mlp = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 32, 1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, out_dim, 1)
        )
        self.ball_query = BallQuery.apply

        # Variables
        self.top_k = k
        self.radius = neighbor_radius
        self.nsample = nsample
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.load_label = load_label
        self.labels = labels
        self.cls = cls

    def forward(self, g, x, x_origin, end_points=None):

        # Basic info
        batch_size = g.batch_size

        # Valuable points selection
        # [Grasp points graph]
        for graph, bn in zip(self.region_score_graph, self.region_score_graph_bn):
            x = graph(g, x) + x
        x_ = x.view(batch_size, -1, self.in_dim).transpose(1, 2)  # (B, in_dim, N)
        score = self.region_score_mlp(x_)  # (B, out_dim, N)
        end_points['vps_score'] = score.squeeze()
        if self.load_label:
            end_points['vps_label'] = end_points['objectness_score_gt']
        cls = score.squeeze().view(batch_size, -1)  # (B, out_dim*N)
        if self.cls:
            cls = torch.argmax(score, dim=1)  # (B, N)

        batch_region_feat_ = cls.view(-1).float()

        # Use top-K scored points
        # [Grasp points graph]
        perm, _, next_batch_num_nodes = top_k(batch_region_feat_, self.top_k,
                                              get_batch_id(g.batch_num_nodes()), g.batch_num_nodes())  # perm: (B*Ns)
        perm_label = perm.view(-1, 1)  # (B*Ns, 1)

        # Calculate objectness score
        if self.cls:
            objectness_score = score.transpose(1, 2).reshape(-1, self.out_dim)  # (B*N, out_dim)
            perm_objectness_score = perm_label.expand(-1, self.out_dim)  # (B*Ns, out_dim)
            end_points['objectness_score_pred'] = torch.gather(objectness_score, 0, perm_objectness_score).view(batch_size, -1, self.out_dim).transpose(1, 2)  # (B, out_dim, Ns)
        else:
            objectness_score = score.squeeze().view(-1)  # (B*N)
            end_points['objectness_score_pred'] = objectness_score[perm].view(batch_size, -1)  # (B, Ns)

        if self.load_label:
            for label_name in self.labels:
                end_points[label_name] = torch.gather(end_points[label_name].view(-1, 1), 0, perm_label).view(batch_size, -1)  # (B, Ns)

        # Build new graph
        g_new = []
        batch_xyz = g.ndata['xyz'].view(-1, 3)[perm].view(batch_size, -1, 3)  # (B, Ns, 3)
        for b in range(batch_size):
            g = dgl.knn_graph(batch_xyz[b], k=32)
            g.ndata['xyz'] = batch_xyz[b]
            g_new.append(g)

        g_new = dgl.batch(g_new)
        x_copy = x_origin[perm]  # (B*Ns, in_dim)
        x = x[perm]  # (B*Ns, in_dim)

        return g_new, x, x_copy, end_points

def get_batch_id(num_nodes: torch.Tensor):
    batch_size = num_nodes.size(0)
    batch_ids = []
    for i in range(batch_size):
        item = torch.full((num_nodes[i],), i, dtype=torch.long, device=num_nodes.device)
        batch_ids.append(item)
    return torch.cat(batch_ids)

def top_k(x: torch.Tensor, k: int, batch_id: torch.Tensor, num_nodes: torch.Tensor):

    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1),
         num_nodes.cumsum(dim=0)[:-1]], dim=0)  # each batch's start idx in the whole graph (B, )

    index = torch.arange(batch_id.size(0), dtype=torch.long, device=x.device)  # (B, )
    index = (index - cum_num_nodes[batch_id]) + (batch_id * max_num_nodes)

    dense_x = x.new_full((batch_size * max_num_nodes,), torch.finfo(x.dtype).min)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)
    perm_batch = perm.view(-1)
    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)

    ks = torch.ones_like(num_nodes) * k
    mask = [
        torch.arange(ks[i], dtype=torch.long, device=x.device) +
        i * max_num_nodes for i in range(batch_size)]

    mask = torch.cat(mask, dim=0)
    perm = perm[mask]
    perm_batch = perm_batch[mask]

    return perm, perm_batch, ks
