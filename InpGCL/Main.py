# -*- coding: utf-8 -*-
import torch
from Parameters import args

torch.cuda.set_device(args.gpu)

import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Parameters import args
from Model import Model, vgae_encoder, vgae_decoder, vgae
from Data_Processing import DataHandler
import numpy as np
from Utils.Utils import calcRegLoss, pairPredict, makeTorchAttention, expand_sparse_matrix, \
    trim_coo_matrix
import os
from copy import deepcopy
import random
import networkx as nx
from scipy.sparse import coo_matrix
import random
from tqdm import tqdm


def random_walk_with_restart(graph, l, R):
    all_nodes = list(graph.nodes())
    node_visits = {}
    restart_prob = 0
    with tqdm(total=len(all_nodes), bar_format='{l_bar}{bar}|') as pbar:
        for node in all_nodes:
            visits = []

            for _ in range(R):
                current_node = node
                previous_node = node

                for _ in range(l):
                    visits.append(current_node)
                    neighbors = list(graph.neighbors(current_node))

                    if random.random() < restart_prob:
                        current_node = previous_node
                    else:
                        if len(neighbors) > 0:
                            previous_node = current_node
                            current_node = random.choice(neighbors)
                        else:
                            break

                restart_prob += 0.2

            restart_prob = 0
            node_visits[node] = visits
            pbar.update(1)
    return node_visits


def calculate_attention_matrix(graph, dictionary, file_path):
    num_nodes = len(graph.nodes)
    attention_matrix = torch.zeros(num_nodes, num_nodes).to(torch.device("cuda"))
    edges_list = list(graph.edges())

    with tqdm(total=len(edges_list), bar_format='{l_bar}{bar}|') as pbar:
        for edge in edges_list:
            key1 = edge[0]
            key2 = edge[1]

            list1 = dictionary[key1]
            list2 = dictionary[key2]
            set1 = set(list1)
            set2 = set(list2)
            a = len(set1.intersection(set2))
            b = float(len(list(graph.neighbors(key1))) + len(list(graph.neighbors(key2))))

            if b == 0:
                attention_matrix[key1][key2] = 0
            else:
                attention_matrix[key1][key2] = 2 * a / b
                attention_matrix[key2][key1] = attention_matrix[key1][key2]
                pbar.update(1)

    print('The Attention-ware Matrix has been Calculated.')
    attention_matrix = attention_matrix.cpu().numpy()
    if args.no_save_mtx_flag:
        print('Saving Attention-aware Matrix...')
        np.savetxt(file_path, attention_matrix, fmt="%f")
        print('Attention-aware Matrix Saved')

    return attention_matrix


def random_edge_update(graph_coo, attention_coo, control_factor):
    num_rows = graph_coo.shape[0]

    total_rows = int(control_factor * num_rows)
    rows_to_update = np.random.choice(num_rows, total_rows, replace=False)
    for row_index in rows_to_update:
        row_indices = graph_coo.row == row_index
        cols_to_update = graph_coo.col[row_indices]
        num_edges_to_delete = np.random.choice([1, 2])
        for col_index in cols_to_update:
            graph_index = np.where((graph_coo.row == row_index) & (graph_coo.col == col_index))[0]
            attention_index = np.where((attention_coo.row == row_index) & (attention_coo.col == col_index))[0]
            if len(graph_index) > 0 and len(attention_index) > 0:
                graph_value = graph_coo.data[graph_index[0]]
                attention_value = attention_coo.data[attention_index[0]]
                if graph_value == 1 and attention_value < np.mean(attention_coo.data):
                    graph_coo.data[graph_index[0]] = 0
                    num_edges_to_delete -= 1
                if num_edges_to_delete == 0:
                    break
    return graph_coo


class Coach:

    def __init__(self, handler):

        self.handler = handler

        print('Users:', args.user, 'Items:', args.item)
        print('Number of Interactions:', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):

        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:

            val = reses[metric]

            ret += '%s = %.4f, ' % (metric, val)

            tem = name + metric

            if save and tem in self.metrics:
                self.metrics[tem].append(val)

        ret = ret[:-2] + '  '

        return ret


    def cal_attention(self):

        adj = deepcopy(self.handler.torchBiAdj)

        trnMat = deepcopy(self.handler.trnMat)

        ret2 = expand_sparse_matrix(trnMat)

        graph = nx.from_scipy_sparse_array(ret2)

        current_path = os.getcwd()

        if not os.path.exists(os.path.join(current_path, "Attention")):
            os.mkdir(os.path.join(current_path, "Attention"))

        file_path_attention = './Attention/' + args.data + '_attention' + '_M' + str(args.M) + '_R' + str(
            args.R) + '.txt'

        if args.no_cal_mtx_flag:
            M = args.M
            R = args.R
            print('Random Walk with Restart in Progress...')
            result = random_walk_with_restart(graph, M, R)
            print('Calculating Attention-aware Matrix...')
            attention = calculate_attention_matrix(graph, result, file_path_attention)

        else:
            print('Attention-aware Matrix has been Calculated in Advance.')
            print('Loading Attention-aware Matrix...')

            attention = np.loadtxt(file_path_attention)
            print('The Attention-ware Matrix has been Successfully Loaded.')

        attention = coo_matrix(attention)
        num_rows = trnMat.shape[0]
        num_cols = trnMat.shape[1]
        attention = trim_coo_matrix(attention, num_rows, num_cols)
        self.attention = attention
        torch_attention = makeTorchAttention(attention)

        self.torch_attention = torch_attention
        add_result = torch_attention * (args.epsilon * adj) + adj
        self.torch_addAttentionAdj = add_result

    def run(self):
        self.cal_attention()
        log('Initializing Model...')
        self.prepareModel()
        log('Model Initialization Completed')
        recallMax = 0
        stloc = 0
        Count_Recall = []
        Count_Ndcg = []

        for ep in range(stloc, args.epoch):

            temperature = max(0.05, args.init_temperature * pow(args.temperature_decay, ep))
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch(temperature)
            log(self.makePrint('Train', ep + 1, reses, tstFlag))

            if tstFlag:

                reses = self.testEpoch()

                if (reses['Recall'] > recallMax):
                    recallMax = reses['Recall']

                if (ep >= args.epoch - args.cal_epochs):
                    Count_Recall.append(reses['Recall'])
                    Count_Ndcg.append(reses['NDCG'])

                log(self.makePrint('Test', ep + 1, reses, tstFlag))

            print()

        avg_recall = np.array(Count_Recall).mean()
        avg_ndcg = np.array(Count_Ndcg).mean()
        print('Avg Recall : %.3f , Avg NDCG : %.3f' % (avg_recall, avg_ndcg))

    def prepareModel(self):

        self.model = Model().cuda()

        encoder = vgae_encoder().cuda()

        decoder = vgae_decoder().cuda()

        self.generator_1 = vgae(encoder, decoder).cuda()

        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        self.opt_gen_1 = torch.optim.Adam(self.generator_1.parameters(), lr=args.lr, weight_decay=0)

        self.opt_gen_2 = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator_1.parameters()), lr=args.lr,
                                          weight_decay=0, eps=args.eps)

    def trainEpoch(self, temperature):

        trnLoader = self.handler.trnLoader

        trnLoader.dataset.negSampling()

        generate_loss_1, generate_loss_2, bpr_loss, infonce_loss, ib_loss, reg_loss = 0, 0, 0, 0, 0, 0

        steps = trnLoader.dataset.__len__() // args.batch

        data = deepcopy(self.torch_addAttentionAdj).cuda()
        data1 = self.generator_generate(self.generator_1, data)
        trnMat = deepcopy(self.handler.trnMat)
        attention = deepcopy(self.attention)
        random_update_adj = random_edge_update(trnMat, attention, control_factor=args.edge_ratio)
        torch_update_adj = self.handler.makeTorchAdj(random_update_adj)

        data1_2 = self.generator_generate(self.generator_1, torch_update_adj)

        for i, tem in enumerate(trnLoader):
            out1 = self.model.forward_graphcl(data1)

            self.opt.zero_grad()
            self.opt_gen_1.zero_grad()
            self.opt_gen_2.zero_grad()

            ancs, poss, negs = tem

            ancs = ancs.long().cuda()
            poss = poss.long().cuda()
            negs = negs.long().cuda()

            out2 = self.model.forward_graphcl(data1_2)

            loss = self.model.loss_graphcl(out1, out2, ancs, poss).mean() * args.ssl_reg

            infonce_loss += float(loss)

            loss.backward()

            self.opt.step()
            self.opt.zero_grad()

            _out1 = self.model.forward_graphcl(data1)

            _out2 = self.model.forward_graphcl(data1_2)

            loss_ib = self.model.loss_graphcl(_out1, out1.detach(), ancs, poss) + self.model.loss_graphcl(_out2,
                                                                                                          out2.detach(),
                                                                                                          ancs, poss)

            loss = loss_ib.mean() * args.ib_reg

            ib_loss += float(loss)

            loss.backward()

            self.opt.step()
            self.opt.zero_grad()

            addAttentionAdj = deepcopy(self.torch_addAttentionAdj).cuda()

            usrEmbeds, itmEmbeds = self.model.forward_gcn(addAttentionAdj)

            ancEmbeds = usrEmbeds[ancs]

            posEmbeds = itmEmbeds[poss]

            negEmbeds = itmEmbeds[negs]

            scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)

            bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch

            regLoss = calcRegLoss(self.model) * args.reg

            loss = bprLoss + regLoss

            bpr_loss += float(bprLoss)

            reg_loss += float(regLoss)

            loss.backward()

            self.opt.step()
            self.opt.zero_grad()

            loss_1 = self.generator_1(deepcopy(self.handler.torchBiAdj).cuda(), ancs, poss, negs)

            loss_2 = self.generator_1(torch_update_adj, ancs, poss, negs)

            loss = loss_1 + loss_2

            generate_loss_1 += float(loss_1)

            generate_loss_2 += float(loss_2)

            loss.backward()

            self.opt.step()
            self.opt.zero_grad()

            self.opt_gen_1.step()

            self.opt_gen_2.step()

            log('Step %d/%d: gen1_loss : %.3f ; gen2_loss : %.3f ; bpr_loss: %.3f ; infonce_loss : %.3f : reg_loss : %.3f  ' % (
                i + 1,
                steps,
                generate_loss_1,
                generate_loss_2,
                bpr_loss,
                infonce_loss,
                reg_loss,
            ), save=False, oneline=True)

        ret = dict()

        ret['Gen_1 Loss'] = generate_loss_1 / steps

        ret['Gen_2 Loss'] = generate_loss_2 / steps

        ret['BPR Loss'] = bpr_loss / steps

        ret['InfoNCE Loss'] = infonce_loss / steps

        ret['IB Loss'] = ib_loss / steps

        ret['Reg Loss'] = reg_loss / steps

        return ret

    def testEpoch(self):

        tstLoader = self.handler.tstLoader

        epRecall, epNdcg = [0] * 2

        i = 0

        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat

        for usr, trnMask in tstLoader:
            i += 1

            usr = usr.long().cuda()

            trnMask = trnMask.cuda()

            usrEmbeds, itmEmbeds = self.model.forward_gcn(self.torch_addAttentionAdj)

            allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8

            _, topLocs = torch.topk(allPreds, args.topk)

            recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)

            epRecall += recall

            epNdcg += ndcg

            log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False,
                oneline=True)

        ret = dict()

        ret['Recall'] = epRecall / num

        ret['NDCG'] = epNdcg / num

        return ret

    def calcRes(self, topLocs, tstLocs, batIds):

        assert topLocs.shape[0] == len(batIds)

        allRecall = allNdcg = 0

        for i in range(len(batIds)):

            temTopLocs = list(topLocs[i])

            temTstLocs = tstLocs[batIds[i]]

            tstNum = len(temTstLocs)

            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])

            recall = dcg = 0

            for val in temTstLocs:

                if val in temTopLocs:
                    recall += 1

                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))

            recall = recall / tstNum

            ndcg = dcg / maxDcg

            allRecall += recall

            allNdcg += ndcg

        return allRecall, allNdcg

    def generator_generate(self, generator, adj):

        edge_index = []

        edge_index.append([])

        edge_index.append([])

        adj_copy = deepcopy(adj)

        idxs = adj._indices()

        with torch.no_grad():
            view = generator.generate(adj, idxs, adj_copy)

        return view


def seed_it(seed):
    random.seed(seed)

    os.environ["PYTHONSEED"] = str(seed)

    np.random.seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True

    torch.backends.cudnn.enabled = True

    torch.manual_seed(seed)


if __name__ == '__main__':
    with torch.cuda.device(args.gpu):
        logger.saveDefault = True

        seed_it(args.seed)

        print('Model: InpGCL')
        print('Loading Data...')

        handler = DataHandler()

        handler.LoadData()

        print('Data Loading Completed')

        coach = Coach(handler)
        coach.run()
