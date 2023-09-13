import torch
import torch.nn as nn
import dgl.function as fn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(x))
        return x


class SIGN(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, R, n_layers, dropout, reduction=2):
        super(SIGN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()
        for hop in range(R + 1):
            self.inception_ffs.append(FeedForwardNet(in_feats, hidden, hidden, n_layers, dropout))
        self.attention = nn.Sequential(
            nn.Linear(R + 1, R + 1),
            nn.ReLU(),
            nn.BatchNorm1d(R + 1),
            nn.Linear(R + 1, R + 1)
        )
        self.softmax = nn.Softmax(dim=1)
        self.residual = nn.Linear(in_feats, hidden)
        self.project = FeedForwardNet(hidden, hidden, out_feats, n_layers, dropout)

    def forward(self, feats):
        hidden = []
        for feat, ff in zip(feats, self.inception_ffs):
            hidden.append(ff(feat))

        # Local attention mechanism (spatial)
        hidden_cat = torch.stack(hidden, dim=0).transpose(0, 1)  # (B*N, R+1, hidden_feat)
        hidden_feat_dim = hidden_cat.size(2)
        hidden_max, _ = torch.max(hidden_cat, dim=2)  # (B*N, R+1)
        hidden_avg = torch.mean(hidden_cat, dim=2)  # (B*N, R+1)
        max_map = self.attention(hidden_max)  # (B*N, R+1)
        avg_map = self.attention(hidden_avg)  # (B*N, R+1)
        att_weight = self.softmax(max_map + avg_map)  # (B*N, R+1)

        # Calculate weighted feature and use residential connection
        att_weight_ = att_weight.unsqueeze(-1).expand(-1, -1, hidden_feat_dim)  # (B*N, R+1, hidden_feat)
        weighted_feat = att_weight_ * hidden_cat  # (B*N, R+1, hidden_feat)
        hop_feat = torch.sum(weighted_feat, dim=1)  # (B*N, hidden_dim)
        # out = self.project(hop_feat + self.residual(feats[0]))  # (B*N, out_feat)
        out = self.project(hop_feat)  # (B*N, out_feat)

        return out


def calc_weight(g):
    """
    Compute row_normalized(D^(-1/2)AD^(-1/2))
    """
    with g.local_scope():
        # compute D^(-0.5)*D(-1/2), assuming A is Identity
        g.ndata["in_deg"] = g.in_degrees().float().pow(-0.5)
        g.ndata["out_deg"] = g.out_degrees().float().pow(-0.5)
        g.apply_edges(fn.u_mul_v("out_deg", "in_deg", "weight"))
        # row-normalize weight
        g.update_all(fn.copy_e("weight", "msg"), fn.sum("msg", "norm"))
        g.apply_edges(fn.e_div_v("weight", "norm", "weight"))
        return g.edata["weight"]


def preprocess_degree(g, features, R):
    """
    Pre-compute the average of n-th hop neighbors
    """
    with torch.no_grad():
        g.edata["weight"] = calc_weight(g)
        g.ndata["feat_0"] = features  # (B*N, 3)
        for hop in range(1, R + 1):
            g.update_all(fn.u_mul_e(f"feat_{hop-1}", "weight", "msg"), fn.sum("msg", f"feat_{hop}"))
        res = []
        for hop in range(R + 1):
            res.append(g.ndata.pop(f"feat_{hop}"))  # (R+1, B*N, 3)
        return g, res

def calc_pos_weight(g):
    """
    Compute row_normalized(D^(-1/2)AD^(-1/2))
    """
    with g.local_scope():
        g.apply_edges(fn.v_sub_u('xyz', 'xyz', 'relative_pos'))
        g.apply_edges(lambda edges: {'relative_pos': edges.data['relative_pos'].pow(2).sum(1).sqrt()})
        g.update_all(fn.copy_e("relative_pos", "pos_sum"), fn.sum("pos_sum", "pos_norm"))
        g.apply_edges(fn.e_div_v("relative_pos", "pos_norm", "pos_weight"))
        return g.edata["pos_weight"]

def preprocess_pos_weight(g, features, R):
    """
    Pre-compute the average of n-th hop neighbors
    """
    with torch.no_grad():
        g.edata["pos_weight"] = calc_pos_weight(g)
        g.ndata["feat_0"] = features  # (B*N, 3)
        for hop in range(1, R + 1):
            g.update_all(fn.u_mul_e(f"feat_{hop - 1}", "pos_weight", "msg"), fn.sum("msg", f"feat_{hop}"))
        res = []
        for hop in range(R + 1):
            res.append(g.ndata.pop(f"feat_{hop}"))  # (R+1, B*N, 3)
        return g, res

def preprocess_no_weight(g, features, R):
    with torch.no_grad():
        g.ndata["feat_0"] = features
        for hop in range(1, R + 1):
            g.update_all(fn.copy_u(f"feat_{hop - 1}", "msg"), fn.mean("msg", f"feat_{hop}"))
        res = []
        for hop in range(R + 1):
            res.append(g.ndata.pop(f"feat_{hop}"))
        return g, res

