
import scipy.sparse as sp

import numpy as np
import torch
from sklearn.metrics import f1_score
import torch_geometric.transforms as T
from os import path as path

# from load_data import load_twitch, load_fb100, load_twitch_gamer, DATAPATH
# from data_utils import rand_splits, rand_train_test_idx, even_quantile_labels, set_random_seed, dataset_drive_url

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Reddit2, Actor, WebKB


DATAPATH = path.dirname(path.abspath(__file__)) + '/data/'

import torch


def reconstruct_blockwise(temp_re_A, Lambda, block_size=1024):
    N = temp_re_A.shape[0]
    device = temp_re_A.device
    A_reconstructed = []

    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        temp_block = temp_re_A[i:i_end]               # (B, D)
        block = temp_block @ Lambda @ temp_re_A.T     # (B, N)
        A_reconstructed.append(block.cpu())           # Offload to CPU if needed

    return torch.cat(A_reconstructed, dim=0)


def f1_test(output, labels):
    preds = output.max(1)[1].type_as(labels)
    # correct = preds.eq(labels).double()
    f1 = f1_score(preds.detach().cpu().numpy(), labels.detach().cpu().numpy(), average='macro')
    return f1




def load_graph_dataset(dataname):
    # dataname = args.dataset
    transform = T.NormalizeFeatures()
    if dataname in ('cora', 'citeseer', 'pubmed'):
        torch_dataset = Planetoid(root=f'{DATAPATH}Planetoid', split='public',
                                  name=dataname, transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'amazon-photo':
        torch_dataset = Amazon(root=f'{DATAPATH}Amazon',
                               name='Photo', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'amazon-computer':
        torch_dataset = Amazon(root=f'{DATAPATH}Amazon',
                               name='Computers', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'coauthor-cs':
        torch_dataset = Coauthor(root=f'{DATAPATH}Coauthor',
                                 name='CS', transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'reddit2':
        torch_dataset = Reddit2(root=f'{DATAPATH}Reddit2',
                                transform=transform)
        dataset = torch_dataset[0]
    elif dataname == 'coauthor-physics':
        torch_dataset = Coauthor(root=f'{DATAPATH}Coauthor',
                                 name='Physics', transform=transform)
        dataset = torch_dataset[0]
    else:
        raise NotImplementedError

    return dataset




def get_node_pairs_from_adj(adj, negative_sampling=False):
    """
    将邻接矩阵转换为节点对。
    参数：
      - adj (torch.Tensor): 邻接矩阵，大小为 N x N，图的边信息。
      - negative_sampling (bool): 是否生成负例节点对，默认 False。
    返回：
      - 正例节点对列表。
      - 如果 negative_sampling=True，返回负例节点对列表。
    """
    # 如果 adj 是稀疏张量，则先转换为密集张量
    if adj.is_sparse:
        adj = adj.to_dense()

    indices = torch.triu_indices(adj.size(0), adj.size(1), offset=1)
    values = adj[indices[0], indices[1]]
    positive_mask = (values == 1)
    positive_pairs_tensor = torch.stack([indices[0][positive_mask], indices[1][positive_mask]], dim=1)
    positive_pairs = positive_pairs_tensor.cpu().tolist()

    if negative_sampling:
        negative_mask = (values == 0)
        negative_pairs_tensor = torch.stack([indices[0][negative_mask], indices[1][negative_mask]], dim=1)
        negative_pairs = negative_pairs_tensor.cpu().tolist()
        return positive_pairs, negative_pairs
    else:
        return positive_pairs


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = aug_normalized_adjacency(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def aug_normalized_adjacency(adj, need_orig=False):
   if not need_orig:
       adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def normalize_matrix(matrix):

    min_val = torch.min(matrix)
    max_val = torch.max(matrix)
    normalized = (matrix - min_val) / (max_val - min_val + 1e-8)  # 避免除以0
    return normalized, (min_val, max_val)


def denormalize_matrix(normalized, params):

    min_val, max_val = params
    return normalized * (max_val - min_val + 1e-8) + min_val

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float)

