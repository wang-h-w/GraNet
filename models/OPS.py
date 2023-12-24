"""
    Author: Haowen Wang
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from dgl.geometry import farthest_point_sampler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

class OPS(nn.Module):
    def __init__(self, fps, in_dim, out_dim, load_label=False, labels=None):
        super(OPS, self).__init__()

        # Networks
        self.object_points_selection = nn.Sequential(
            nn.Conv1d(in_dim, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 16, 1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, out_dim, 1)
        )
        self.act = nn.ReLU()

        # Variables
        self.fps = fps
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.load_label = load_label
        self.labels = labels

    def forward(self, g, x, x_origin, end_points=None):

        # Basic info
        batch_size = len(end_points['graph'])
        pos = g.ndata['xyz']  # (B*N, 3)
        x_copy = x_origin.view(batch_size, -1, self.in_dim)  # (B, N, in_dim)
        x = x.view(batch_size, -1, self.in_dim)  # (B, N, in_dim)

        # Object points selection
        # [Objects graph]
        x_mask = self.object_points_selection(x.transpose(1, 2))  # (B, out_dim, N)

        if self.load_label:
            end_points['ops_label'] = end_points['objectness_label']  # (B, N)
        end_points['ops_score'] = x_mask  # criterion use data before softmax! (B, out_dim, N)
        x_mask = x_mask.transpose(1, 2)  # (B, N, out_dim)
        object_mask = torch.argmax(F.softmax(x_mask, dim=2), dim=2)  # (B, N)

        # FPS among object points
        # [Objects graph]
        pos_ = pos.view(batch_size, -1, 3)  # (B, N, 3)
        x_new = []
        x_copy_new = []
        pos_new = []
        g_new = []
        selected_labels = []
        if self.load_label:
            for _ in range(len(self.labels)):
                selected_labels.append([])

        for b in range(batch_size):
            pos_b = pos_[b]  # (N, 3)
            x_copy_b = x_copy[b]  # (N, in_dim)
            x_b = x[b]
            N, _ = pos_b.size()

            mask = object_mask[b]  # (N)
            idx_mask = torch.flatten(torch.nonzero(mask))  # (N_pos)
            N_pos = len(idx_mask)
            if N_pos == 0:
                N_pos = pos_b.shape[0]
                idx_mask = torch.arange(N_pos)

            # (1) Mask by softmax result
            pos_b = torch.gather(pos_b, 0, idx_mask.view(N_pos, 1).expand(-1, 3))  # (N_pos, 3)
            x_copy_b = torch.gather(x_copy_b, 0, idx_mask.view(N_pos, 1).expand(-1, self.in_dim))  # (N_pos, in_dim)
            x_b = torch.gather(x_b, 0, idx_mask.view(N_pos, 1).expand(-1, self.in_dim))  # (N_pos, in_dim)
            label_temp = []
            if self.load_label:
                for label_name in self.labels:
                    label_temp.append(end_points[label_name][b][idx_mask])  # (N_pos)

            # (2) Mask by FPS
            if N_pos == 0:
                idx = torch.as_tensor(np.random.choice(N, self.fps, replace=True))
                print("Warning: N_pos = 0!")
            else:
                if N_pos >= self.fps:
                    idx = farthest_point_sampler(torch.unsqueeze(pos_b, dim=0), self.fps)[0]  # (Ns)
                else:
                    idx1 = torch.arange(N_pos)
                    idx2 = torch.as_tensor(np.random.choice(N_pos, self.fps - N_pos, replace=True))
                    idx = torch.cat((idx1, idx2), dim=0)  # (Ns)

            end_points['att1_to_att2_idx'] = idx_mask[idx]

            pos_fps = pos_b[idx]  # (Ns, 3)
            x_copy_fps = x_copy_b[idx]  # (Ns, in_dim)
            x_fps = x_b[idx]

            if self.load_label:
                for i, label in enumerate(label_temp):
                    selected_labels[i].append(label[idx])

            g_fps = dgl.knn_graph(pos_fps, k=32)
            g_fps.ndata['xyz'] = pos_fps

            pos_new.append(pos_fps)
            x_copy_new.append(x_copy_fps)
            x_new.append(x_fps)
            g_new.append(g_fps)

        x_copy_new = torch.stack(x_copy_new, dim=0).view(-1, self.in_dim)  # (B*Ns, in_dim)
        x_new = torch.stack(x_new, dim=0).view(-1, self.in_dim)
        g_new = dgl.batch(g_new)
        if self.load_label:
            for i, label_name in enumerate(self.labels):
                end_points[label_name] = torch.stack(selected_labels[i], dim=0).view(batch_size, -1)  # (B, Ns)

        return g_new, x_new, x_copy_new, end_points
