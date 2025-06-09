# -*- coding: utf-8 -*-

from torch import nn

import torch.nn.functional as F

import torch

from Parameters import args

torch.cuda.set_device(args.gpu)

import numpy as np
from Utils.Utils import calcRegLoss, pairPredict

import torch_sparse

init = nn.init.xavier_uniform_


class Model(nn.Module):

    def __init__(self):

        super(Model, self).__init__()

        self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))

        self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))

        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

    def forward_gcn(self, adj):

        iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

        embedsLst = [iniEmbeds]

        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)

        mainEmbeds = sum(embedsLst)

        return mainEmbeds[:args.user], mainEmbeds[args.user:]

    def forward_graphcl(self, adj):
        iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst)

        return mainEmbeds

    def forward_graphcl_(self, generator):

        iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

        embedsLst = [iniEmbeds]
        count = 0

        for gcn in self.gcnLayers:
            with torch.no_grad():
                adj = generator.generate(x=embedsLst[-1], layer=count)

            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
            count += 1

        mainEmbeds = sum(embedsLst)

        return mainEmbeds

    def loss_graphcl(self, x1, x2, users, items):

        T = args.temp

        user_embeddings1, item_embeddings1 = torch.split(x1, [args.user, args.item], dim=0)

        user_embeddings2, item_embeddings2 = torch.split(x2, [args.user, args.item], dim=0)

        user_embeddings1 = F.normalize(user_embeddings1, dim=1)

        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)

        user_embs1 = F.embedding(users, user_embeddings1)

        item_embs1 = F.embedding(items, item_embeddings1)
        user_embs2 = F.embedding(users, user_embeddings2)
        item_embs2 = F.embedding(items, item_embeddings2)

        all_embs1 = torch.cat([user_embs1, item_embs1], dim=0)
        all_embs2 = torch.cat([user_embs2, item_embs2], dim=0)

        all_embs1_abs = all_embs1.norm(dim=1)
        all_embs2_abs = all_embs2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', all_embs1, all_embs2) / torch.einsum('i,j->ij', all_embs1_abs,
                                                                                    all_embs2_abs)

        sim_matrix = torch.exp(sim_matrix / T)

        pos_sim = sim_matrix[np.arange(all_embs1.shape[0]), np.arange(all_embs1.shape[0])]

        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

        loss = - torch.log(loss)

        return loss

    def getEmbeds(self):

        self.unfreeze(self.gcnLayers)

        return torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

    def unfreeze(self, layer):

        for child in layer.children():

            for param in child.parameters():
                param.requires_grad = True

    def getGCN(self):
        return self.gcnLayers


class GCNLayer(nn.Module):
    def __init__(self):

        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds, flag=True):

        if (flag):
            return torch.spmm(adj, embeds)
        else:

            return torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)


class vgae_encoder(Model):
    def __init__(self):
        super(vgae_encoder, self).__init__()

        hidden = args.latdim

        self.encoder_mean = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))

        self.encoder_std = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
                                         nn.Softplus())

    def forward(self, adj):
        x = self.forward_graphcl(adj)

        x_mean = self.encoder_mean(x)

        x_std = self.encoder_std(x)

        gaussian_noise = torch.randn(x_mean.shape).cuda()

        x = gaussian_noise * x_std + x_mean

        return x, x_mean, x_std


class vgae_decoder(nn.Module):

    def __init__(self, hidden=args.latdim):
        super(vgae_decoder, self).__init__()

        self.decoder = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                                     nn.Linear(hidden, 1))

        self.sigmoid = nn.Sigmoid()

        self.bceloss = nn.BCELoss(reduction='none')

    def forward(self, x, x_mean, x_std, users, items, neg_items, encoder):
        x_user, x_item = torch.split(x, [args.user, args.item], dim=0)

        edge_pos_pred = self.sigmoid(self.decoder(x_user[users] * x_item[items]))

        edge_neg_pred = self.sigmoid(self.decoder(x_user[users] * x_item[neg_items]))

        loss_edge_pos = self.bceloss(edge_pos_pred, torch.ones(edge_pos_pred.shape).cuda())

        loss_edge_neg = self.bceloss(edge_neg_pred, torch.zeros(edge_neg_pred.shape).cuda())

        loss_rec = loss_edge_pos + loss_edge_neg

        kl_divergence = - 0.5 * (1 + 2 * torch.log(x_std) - x_mean ** 2 - x_std ** 2).sum(dim=1)

        ancEmbeds = x_user[users]
        posEmbeds = x_item[items]
        negEmbeds = x_item[neg_items]

        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)

        bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch

        regLoss = calcRegLoss(encoder) * args.reg

        beta = 0.1

        loss = (
                loss_rec + beta * kl_divergence.mean() + bprLoss + regLoss).mean()

        return loss


class vgae(nn.Module):
    def __init__(self, encoder, decoder):
        super(vgae, self).__init__()

        self.encoder = encoder

        self.decoder = decoder

    def forward(self, data, users, items, neg_items):
        x, x_mean, x_std = self.encoder(data)

        loss = self.decoder(x, x_mean, x_std, users, items, neg_items, self.encoder)

        return loss

    def generate(self, data, edge_index, adj):
        x, _, _ = self.encoder(data)

        edge_pred = self.decoder.sigmoid(self.decoder.decoder(x[edge_index[0]] * x[edge_index[1]]))

        vals = adj._values()

        idxs = adj._indices()

        edgeNum = vals.size()

        edge_pred = edge_pred[:, 0]

        mask = ((edge_pred + 0.5).floor()).type(torch.bool)

        newVals = vals[mask]

        newVals = newVals / (newVals.shape[0] / edgeNum[0])

        newIdxs = idxs[:, mask]

        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)
