"""
    Author: Haowen Wang
"""
import os
import sys
import torch.nn as nn

from dgl.nn.pytorch.conv import EdgeConv
from GFE import GFE
from OPS import OPS
from VPS import VPS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'knn'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

class GraphBackbone(nn.Module):
    def __init__(self, is_training=True):

        super(GraphBackbone, self).__init__()

        # Networks
        self.gfe = GFE(knn=32, module_feat_dim=64, sign_feat_in_dim=64, out_feat_dim=256, sample_nodes=7000, is_training=is_training)
        self.ops = OPS(fps=2048, in_dim=256, out_dim=2, load_label=is_training,
                       labels=['objectness_label', 'objectness_score_gt'])
        self.vps = VPS(k=512, in_dim=256, out_dim=10, neighbor_radius=0.02, nsample=16,
                       load_label=is_training, labels=['objectness_label', 'objectness_score_gt'],
                       cls=True)
        self.gcn = nn.ModuleList()
        self.gcn_bn = nn.ModuleList()
        for i in range(2):
            self.gcn.append(EdgeConv(256, 256))
            self.gcn_bn.append(nn.BatchNorm1d(256))
        self.out_linear = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256)
        )
        self.dropout = nn.Dropout(0.2)
        self.act = nn.ReLU()

        # Variables
        self.load_label = is_training

    def forward(self, gs, end_points=None):

        # Basic info
        batch_size = len(gs)

        # Graph feature embedding network
        g, x, end_points = self.gfe(gs, end_points)
        x_clone = x.clone()

        # Object points selection network
        g_fps, x_fps, x_clone_fps, end_points = self.ops(g, x, x_clone, end_points)

        # Valuable points selection network
        g, x, x_clone, end_points = self.vps(g_fps, x_fps, x_clone_fps, end_points)
        x = x_clone

        # Feature propagation by EdgeConv
        for gcn, bn in zip(self.gcn, self.gcn_bn):
            x = gcn(g, x) + x
            x = bn(self.act(x))
        x = self.dropout(x)
        x = self.out_linear(x)  # (B*Ns, 256)
        out = x.view(batch_size, -1, 256).transpose(1, 2)  # (B, 256, Ns)

        seed_xyz = g.ndata['xyz'].view(batch_size, -1, 3)
        end_points['grasp_points'] = seed_xyz

        return out, seed_xyz.float(), end_points
