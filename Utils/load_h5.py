import os.path
import torch
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from sklearn.neighbors import kneighbors_graph
from collections import Counter
import hdf5storage
import math
import random

def load_mat(mat_path, topk=10, train_ratio=0.1, generative_adjs=True):
    """
    load dataset
    :param mat_path: str file path
    :param topk: int default 10
    :param train_ratio: float [0,1]
    :return:
        adjs, inputs, labels, train_bool, val_bool, n_view, n_node, n_feats, n_class
    """
    print('=' * 15, 'Load data ', os.path.basename(mat_path), '=' * 15)
    try:
        mat = sio.loadmat(mat_path)
    except:
        mat = hdf5storage.loadmat(mat_path)
    X = mat['X']
    if X.shape[0] == 1:
        X = X[0]  # [n_view,n_nodes,n_feature]
    else:
        X = X[:, 0]
    Y = np.squeeze(mat['Y']) - np.min(mat['Y'])

    labels = Y
    n_class = len(np.unique(labels))
    n_view = X.shape[0]
    n_node = X[0].shape[0]
    inputs = []
    n_feats = []
    if generative_adjs:
        adjs = []
    else:
        adjs = torch.tensor(0.0)

    for i in range(n_view):
        tempX = X[i]
        if sp.isspmatrix(tempX):
            tempX = tempX.toarray()
        inputs.append(torch.from_numpy(tempX.astype(np.float32)).float())
        if generative_adjs:
            adjs.append(get_adj_matrix(tempX, topk).to_dense().float())
        n_feats.append(len(tempX[0]))
    train_bool, val_bool = generate_permutation(labels, train_ratio)
    labels = torch.from_numpy(labels).long()
    train_bool = torch.from_numpy(train_bool)
    val_bool = torch.from_numpy(val_bool)
    print(f'n_view: {n_view} n_node: {n_node} n_feats: {n_feats} n_class: {n_class}')
    print(f'labels: {labels.shape} train_bool: {train_bool.sum()} val_bool: {val_bool.sum()}')
    return adjs, inputs, labels, train_bool, val_bool, n_view, n_node, n_feats, n_class