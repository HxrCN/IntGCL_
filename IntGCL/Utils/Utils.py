# -*- coding: utf-8 -*-
import torch as t
import torch.nn.functional as F
import torch
from Parameters import args


torch.cuda.set_device(args.gpu)
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix




def innerProduct(usrEmbeds, itmEmbeds):

    return t.sum(usrEmbeds * itmEmbeds,
                 dim=-1)

def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
    return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)


def calcRegLoss(model):
    ret = 0

    for W in model.parameters():
        ret += W.norm(2).square()
    return ret


def contrastLoss(embeds1, embeds2, nodes, temp):
    embeds1 = F.normalize(embeds1, p=2)
    embeds2 = F.normalize(embeds2, p=2)
    pckEmbeds1 = embeds1[nodes]
    pckEmbeds2 = embeds2[nodes]
    nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
    deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1)
    return -t.log(nume / deno)


def SparseTorchAdd(sparse1, sparse2):

    result_sparse = sparse1 * sparse2 + sparse2
    return result_sparse


def makeTorchAttention(mat):
    a = sp.csr_matrix((args.user, args.user))
    b = sp.csr_matrix((args.item, args.item))

    mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
    idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
    vals = torch.from_numpy(mat.data.astype(np.float32))
    shape = torch.Size(mat.shape)
    return torch.sparse.FloatTensor(idxs, vals, shape).cuda()


def expand_sparse_matrix(ret):
    coo_mat = coo_matrix(ret)
    max_shape = max(coo_mat.shape)
    indices = np.vstack((coo_mat.row, coo_mat.col)).astype(np.int32)

    # 创建稀疏矩阵
    expanded_mat = coo_matrix(
        (coo_mat.data, (indices[0], indices[1])),
        shape=(max_shape, max_shape)
    )

    return expanded_mat


def trim_coo_matrix(matrix, num_rows, num_cols):

    rows = matrix.row
    cols = matrix.col
    data = matrix.data


    mask = (rows < num_rows) & (cols < num_cols)


    trimmed_rows = torch.tensor(rows[mask])
    trimmed_cols = torch.tensor(cols[mask])
    trimmed_data = torch.tensor(data[mask])


    trimmed_matrix = coo_matrix((trimmed_data, (trimmed_rows, trimmed_cols)), shape=(num_rows, num_cols))

    return trimmed_matrix


def normalize_sparse_matrix(matrix):

    min_value = matrix.min()
    max_value = matrix.max()
    normalized_matrix = (matrix - min_value) / (max_value - min_value)

    return normalized_matrix






def normalize_2d_array(arr):

    global_min = np.min(arr)
    global_max = np.max(arr)
    normalized_arr = (arr - global_min) / (global_max - global_min)

    return normalized_arr
