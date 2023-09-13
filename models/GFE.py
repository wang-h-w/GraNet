"""
    Author: Haowen Wang
"""
import torch
import torch.nn as nn
import dgl

from dgl.nn.pytorch.conv import EdgeConv
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from SIGN_attention import SIGN, preprocess_no_weight
from graph_generator import GraphGenerator

class GFE(nn.Module):
    def __init__(self, knn=32, module_feat_dim=64, sign_feat_in_dim=64, out_feat_dim=256, sample_nodes=7000, is_training=True):

        super(GFE, self).__init__()

        # Networks
        self.sign_feat_in_dim = sign_feat_in_dim
        self.generator = GraphGenerator(knn)
        self.sign = SIGN(in_feats=3, hidden=32, out_feats=self.sign_feat_in_dim, R=4, n_layers=2, dropout=0.2)
        self.gap = GlobalAttentionPooling(nn.Linear(self.sign_feat_in_dim, 1))
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, self.sign_feat_in_dim)
        )
        self.gfe_feature_learning = EdgeConv(self.sign_feat_in_dim * 3, out_feat_dim)

        # Variables
        self.sign_feat_in_dim = module_feat_dim
        self.sample = sample_nodes
        self.load_label = is_training
        self.knn = knn

    def forward(self, gs, end_points=None):

        # Basic info
        batch_size = len(gs)
        g = dgl.batch(gs)
        pos = g.ndata['xyz']  # (B*N, 3)

        # Graph feature learning module with local attention
        # [Geometric graph]
        g, pos_feat = preprocess_no_weight(g, pos, R=4)
        x_ = self.sign(pos_feat)  # (B*N, sign_feat_in_dim)
        x = x_.view(batch_size, -1, self.sign_feat_in_dim)  # (B, N, sign_feat_in_dim)
        _, N, _ = x.size()

        # Global pooling module
        # [Geometric graph]
        global_feat = self.gap(g, x_).view(batch_size, 1, self.sign_feat_in_dim).expand(batch_size, N, self.sign_feat_in_dim)  # (B, N, sign_feat_in_dim)

        # Position encoding module
        # [Geometric graph]
        pos_feat = self.pos_mlp(pos).view(batch_size, -1, self.sign_feat_in_dim)  # (B, N, sign_feat_in_dim)

        # Feature concatenation of three submodule
        x_with_scene_feat = []
        for b in range(batch_size):
            x_with_scene_feat.append(torch.cat((x[b], pos_feat[b], global_feat[b]), dim=1))  # (N, sign_feat_in_dim * 3)
        x = torch.stack(x_with_scene_feat, dim=0).view(-1, self.sign_feat_in_dim * 3)  # (B*N, sign_feat_in_dim * 3)

        # Down-sampling points and build new graph (computation efficiency)
        # [Geometric graph]
        end_points, g, x, pos, _ = self.generator.renew_graph_fps(end_points, g, x, npoints=self.sample,
                                                                  ks=self.knn, load_label=self.load_label)

        x = self.gfe_feature_learning(g, x)

        return g, x, end_points
