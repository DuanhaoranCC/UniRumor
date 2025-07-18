# -*- coding: utf-8 -*-
# @Time    :
# @Author  :
# @Email   :
# @File    : model.py
# @Software: PyCharm
# @Note    :
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import global_mean_pool, global_add_pool, TemporalEncoding, GINConv, GCNConv, BatchNorm, \
    global_max_pool, LayerNorm, SAGEConv, GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected, dropout_edge, mask_feature, \
    to_networkx
from torch_geometric.data import Data, Batch
import numpy as np
import networkx as nx
import copy
from load_data import text_to_vector
from functools import partial


class textprompt(nn.Module):
    def __init__(self, in_dim):
        super(textprompt, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(in_features=in_dim, out_features=128),
                                 nn.Tanh(),
                                 nn.Linear(in_features=128, out_features=in_dim))
        self.dropout = nn.Dropout(p=0.5)
        self.prompttype = 'add'
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, h):

        h = self.dropout(h)
        if self.prompttype == 'add':
            weight = self.weight.repeat(h.shape[0], 1)
            h = self.mlp(h) + h
        if self.prompttype == 'mul':
            h = self.mlp(h) * h

        return h


class TDrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(TDrumorGCN, self).__init__()

        self.conv1 = GCNConv(in_feats, 128)
        self.conv2 = GCNConv(128, out_feats)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)

        h = self.conv2(h, edge_index)
        # h = F.relu(h)
        # h = F.dropout(h, training=self.training)

        hs = global_add_pool(h, batch)

        return hs, h


class BUrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(BUrumorGCN, self).__init__()

        self.conv1 = GCNConv(in_feats, 128)
        self.conv2 = GCNConv(128, out_feats)

    def forward(self, x, edge_index, batch):
        edge_index = torch.flip(edge_index, dims=[0])

        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)

        h = self.conv2(h, edge_index)
        # h = F.relu(h)
        # h = F.dropout(h, training=self.training)

        hs = global_add_pool(h, batch)

        return hs, h


def compute_one_level_ratios(data):
    edge_index = data.edge_index
    batch = data.batch
    num_graphs = batch.max().item() + 1

    total_non_source_nodes = 0
    s = []
    for i in range(num_graphs):

        node_mask = (batch == i)
        nodes = node_mask.nonzero(as_tuple=True)[0]

        source_node = nodes[0].item()
        current_nodes = set(nodes.tolist())

        mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        sub_edge_index = edge_index[:, mask]

        non_source_nodes = len(current_nodes)
        total_non_source_nodes += non_source_nodes

        one_level_count = (sub_edge_index[0] == source_node).sum().item()
        # total_one_level_nodes += one_level_count
        s.append(one_level_count / non_source_nodes)
    return s


def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss = - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


class BiGCN_graphcl(torch.nn.Module):
    def __init__(self, in_feats, out_feats, t, u):
        super(BiGCN_graphcl, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, out_feats)
        self.proj_head = nn.Sequential(nn.Linear(out_feats * 2, 256), nn.ReLU(inplace=True),
                                       nn.Linear(256, 128))

        self.t = t
        self.b = 0.1
        self.u = u
        self.dim = 128
        self.prompt = nn.Sequential(nn.Dropout(p=0.5),
                                    nn.Linear(in_features=in_feats, out_features=self.dim),
                                    LayerNorm(self.dim),
                                    nn.Tanh(),
                                    nn.Linear(in_features=self.dim, out_features=in_feats))

        self.prompt2 = nn.Sequential(nn.Dropout(p=0.5),
                                     nn.Linear(in_features=in_feats, out_features=self.dim),
                                     LayerNorm(self.dim),
                                     nn.Tanh(),
                                     nn.Linear(in_features=self.dim, out_features=in_feats))
        # self.decoder = nn.Sequential(nn.Linear(out_feats * 2, 256), nn.ReLU(inplace=True),
        #                                nn.Linear(256, in_feats))
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.prompt:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.prompt2:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def process_data(self, data):
        """
        Process a single dataset with the corresponding prompt.
        :param data: Input graph data.
        :return: Combined TD and BU features.
        """
        # edge_index = to_undirected(data.edge_index)
        x = data.x
        ################################################
        # g = global_max_pool(x, data.batch)
        # prompt_n = self.prompt(g)[data.batch]
        ################################################
        root_indices = []
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            root_indices.append(torch.nonzero(data.batch == num_batch, as_tuple=False)[0].item())

        ###############################################
        # (1-a)*p*x+a*(p)
        # one_level = torch.FloatTensor(compute_one_level_ratios(data)).to(x.device)
        # alpha = torch.sigmoid((one_level - self.u) / self.b).unsqueeze(-1)
        #
        # prompt1 = (1 - alpha) * self.prompt(x[root_indices])
        # prompt2 = alpha * (self.prompt2(x[root_indices]))
        #
        # z = x * prompt1[data.batch] + prompt2[data.batch]
        ################################################
        # p*root
        # prompt_n = self.prompt(x[root_indices])[data.batch]
        # z = x + prompt_n

        ################################################
        # (1-a)*p*x+a*(p+x)
        one_level = torch.FloatTensor(compute_one_level_ratios(data)).to(x.device)
        alpha = torch.sigmoid((one_level - self.u) / self.b).unsqueeze(-1)

        prompt1 = (1 - alpha) * self.prompt(x[root_indices])
        prompt2 = alpha[data.batch] * (self.prompt2(x[root_indices])[data.batch] + x)
        # # # Ablation alpha
        # prompt1 = self.prompt(x[root_indices])
        # prompt2 = self.prompt2(x[root_indices])[data.batch]
        z = x * prompt1[data.batch] + prompt2
        ############################################################
        # prompt1 = (1 - alpha[data.batch]) * self.prompt(x)
        # prompt2 = alpha[data.batch] * (self.prompt2(x) + x)
        # z = x * prompt1 + prompt2

        ################################################
        TD_x, q1 = self.TDrumorGCN(z, data.edge_index, data.batch)
        BU_x, q2 = self.BUrumorGCN(z, data.edge_index, data.batch)
        h = torch.cat((BU_x, TD_x), 1)
        qs = torch.cat((q1, q2), 1)

        return h, qs

    def forward(self, *data_list):
        """
        Forward pass for multiple datasets.
        :param data_list: List of input graph data objects.
        :return: Projected feature representation.
        """

        hs = []
        # loss = 0
        for data in data_list:
            h, _ = self.process_data(data)
            # h = self.decoder(h)
            # loss += sce_loss(data.x, h)
            hs.append(h)
        # return loss
        h = torch.cat(hs, dim=0)

        h = self.proj_head(h)

        return h

    def loss_graphcl(self, x1, x2, mean=True):

        batch_size, _ = x1.size()

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / self.t)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)
        if mean:
            loss = loss.mean()
        return loss

    def loss_cca(self, h1, h2):

        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)

        c = torch.mm(z1.T, z2)
        c1 = torch.mm(z1.T, z1)
        c2 = torch.mm(z2.T, z2)

        N = z1.size(0)

        c = c / N
        c1 = c1 / N
        c2 = c2 / N

        loss_inv = -torch.diagonal(c).sum()
        iden = torch.tensor(np.eye(c.shape[0])).to(z1.device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()

        loss = loss_inv + 1e-3 * (loss_dec1 + loss_dec2)
        return loss

    def get_embeds(self, data):

        h, _ = self.process_data(data)

        return h


class GNN_graphpred(torch.nn.Module):
    def __init__(self, out_feats):
        super(GNN_graphpred, self).__init__()

        # Initialize GNN module
        self.gnn = BiGCN_graphcl(768, out_feats, 0.5)

        # Initialize prompt parameter
        self.prompt = nn.Parameter(torch.FloatTensor(1, 768))
        self.reset_parameters()

        # Freeze GNN parameters
        self.freeze_gnn_parameters()

    def reset_parameters(self):
        # Xavier initialization for prompt parameter
        torch.nn.init.xavier_uniform_(self.prompt)

    def freeze_gnn_parameters(self):
        # Freeze all parameters in GNN module (no gradients)
        for param in self.gnn.parameters():
            param.requires_grad = False

    def from_pretrained(self, model_file):
        # Load pretrained GNN weights
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, data):
        data = self.gnn.add_virtual_node(data.x, data)
        # Forward pass: multiply prompt with input
        x = self.prompt * data.x

        # Apply GNN modules for different graph parts
        TD_x1, _ = self.gnn.TDrumorGCN(x, data.edge_index, data.batch)
        BU_x1, _ = self.gnn.BUrumorGCN(x, data.edge_index, data.batch)

        # Concatenate outputs from both parts
        h = torch.cat((BU_x1, TD_x1), 1)

        return h


class PLAN(nn.Module):
    def __init__(self, embed_dim=300, n_heads=8, n_layers=12, num_classes=2, dropout=0.3, use_time_embed=False,
                 time_bins=100):
        super(PLAN, self).__init__()
        self.embed_dim = embed_dim
        self.use_time_embed = use_time_embed

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=2 * embed_dim,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Time delay embedding（可选）
        if use_time_embed:
            self.time_embed = nn.Embedding(time_bins, embed_dim)
        else:
            self.time_embed = None

        # Attention参数
        self.gamma = nn.Parameter(torch.randn(embed_dim))
        # 分类层
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, tweet_reps, time_bins=None):
        """
        tweet_reps: (B, N, D)  # B: batch, N: posts, D: embed_dim
        time_bins: (B, N)      # 可选，每条tweet的时间延迟分桶

        1. 每条推文（如max-pool过词向量） -> tweet_reps输入
        2. 可选加时间延迟embedding
        3. transformer做self-attention，编码所有tweet间交互
        4. attention池化得整体表示
        5. 分类预测
        """

        # 加时间延迟embedding（可选）
        if self.use_time_embed and time_bins is not None:
            tweet_reps = tweet_reps + self.time_embed(time_bins)

        # transformer编码推文级关系
        post_out = self.transformer_encoder(tweet_reps)  # (B, N, D)

        # Attention池化（论文Eq.3-4）
        # 计算每个post的权重（α），对输出做加权求和，得整体表示v
        attn_logits = torch.matmul(post_out, self.gamma)  # (B, N)
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, N)
        v = torch.sum(attn_weights.unsqueeze(-1) * post_out, dim=1)  # (B, D)

        # 分类
        logits = self.fc(v)  # (B, num_classes)
        return logits, attn_weights


class Encoder1(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(Encoder1, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, out_feats)

    def forward(self, data, x):
        TD_x1, h1 = self.TDrumorGCN(x, data.edge_index, data.batch)
        BU_x1, h2 = self.BUrumorGCN(x, data.edge_index, data.batch)
        hs = torch.cat((BU_x1, TD_x1), 1)
        h = torch.cat((h1, h2), 1)

        return h, hs


def mask1(x, batch, mask_rate=0.5):
    """
    Mask a subset of nodes in a batched graph, ensuring the 0th node in each graph is not masked.

    Args:
        x (torch.Tensor): Node feature matrix.
        batch (torch.Tensor): Batch vector indicating the graph each node belongs to.
        mask_rate (float): The rate of nodes to be masked.

    Returns:
        torch.Tensor: Indices of the masked nodes.
    """
    mask_nodes = []  # Store indices of masked nodes
    num_graphs = batch.max().item() + 1  # Number of graphs in the batch

    for graph_idx in range(num_graphs):
        # Get the nodes belonging to the current graph
        graph_nodes = (batch == graph_idx).nonzero(as_tuple=True)[0]
        # Exclude the 0th node of the graph
        non_zero_nodes = graph_nodes[1:] if len(graph_nodes) > 1 else graph_nodes
        # Compute the number of nodes to mask
        num_mask_nodes = int(mask_rate * len(non_zero_nodes))
        # Randomly select nodes to mask
        perm = torch.randperm(len(non_zero_nodes), device=x.device)
        masked_nodes = non_zero_nodes[perm[:num_mask_nodes]]
        mask_nodes.append(masked_nodes)

    # Concatenate masked node indices from all graphs
    return torch.cat(mask_nodes)

class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))



class CFOP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rate, alpha, hidden=64):
        super().__init__()

        # self.online_encoder = Encoder(in_dim, out_dim, hidden, 2)
        # self.target_encoder = Encoder(in_dim, out_dim, hidden, 2)
        self.online_encoder = Encoder1(in_dim, out_dim)
        self.target_encoder = Encoder1(in_dim, out_dim)
        self.criterion = self.setup_loss_fn("sce")
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))

        self.rate = rate
        self.alpha = alpha

        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def setup_loss_fn(self, loss_fn):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=1)
        else:
            raise NotImplementedError
        return criterion

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.
        Args:
            mm (float): Momentum used in moving average update.
        """
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, *data_list):
        loss = 0
        for data in data_list:
            # data = self.add_virtual_node(data.x, data)
            x, edge_index, batch = data.x, data.edge_index, data.batch
            mask_nodes = mask1(x, batch, mask_rate=self.rate)
            x1 = x.clone()
            x1[mask_nodes] = 0.0
            x1[mask_nodes] += self.enc_mask_token

            h1, gh1 = self.online_encoder(data, x1)
            with torch.no_grad():
                h2, gh2 = self.target_encoder(data, x)

            loss += self.criterion(h1[mask_nodes], h2[mask_nodes].detach()) + \
                    self.criterion(gh1, gh2.detach())

        return loss

    def get_embeds(self, data):
        # data = self.add_virtual_node(data.x, data)
        h, gh = self.online_encoder(data, data.x)

        return gh
