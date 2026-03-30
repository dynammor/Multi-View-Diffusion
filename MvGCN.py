import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.sparse as sparse
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from args import get_arguments


args =get_arguments()

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GCN(nn.Module):
    """GCN Network Structure"""

    def __init__(self, nfeat, nhid, nclass, dropout=args.dropout, layers=args.layer_num):
        super(GCN, self).__init__()

        self.layer_list = nn.ModuleList()

        self.gc_in = GraphConvolution(nfeat, nhid)
        for _ in range(layers - 2):
            self.layer_list.append(GraphConvolution(nhid, nhid))
        self.gc_out = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        self.gc_in.reset_parameters()
        for layer in self.layer_list:
            layer.reset_parameters()
        self.gc_out.reset_parameters()

    def forward(self, x, adj):
        x = F.relu(self.gc_in(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        for layer in self.layer_list:
            x = layer(x, adj)

        x = self.gc_out(x, adj)
        return F.log_softmax(x, dim=1)


class MvGCN(nn.Module):
    def __init__(self, nfeats, num_class, nhid=args.nhid, dropout=args.dropout, batch_norm=1):
        super(MvGCN, self).__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm  # 和F.normalize配套用，加不加差不多
        self.GCNs = torch.nn.ModuleList()

        for nfeat in nfeats:
            self.GCNs.append(GCN(nfeat, nhid, num_class, dropout=0.5))

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(num_class)
        num_of_view = len(nfeats)
        self.W = nn.Parameter(torch.randn(num_of_view, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.GCNs)):
            self.GCNs[i].reset_parameters()
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))  # 初始化 self.W 的权重，加不加差不多

    def forward(self, X, adj):
        GCN_outputs = []
        for idx, model in enumerate(self.GCNs):
            tmp_output = model(X[idx], adj[idx])  # 每个视角GCN
            GCN_outputs.append(tmp_output)

        output = torch.stack(GCN_outputs, dim=1)  # (n, len(ks), nfeat)
        output = F.normalize(output, dim=-1)  # 加了会好
        # W = F.softmax(self.W)
        output = self.W * output

        output = output.sum(1)
        if self.batch_norm:
            output = self.bn1(output)
        output = F.dropout(output, self.dropout, training=self.training)  # 加了会好
        return output
