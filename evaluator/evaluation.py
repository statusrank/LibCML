# -*- coding: utf-8 -*-
'''
CopyRight: Shilong Bao
Email: baoshilong@iie.ac.cn
'''

import os 
import numpy as np 
from collections import defaultdict
import math 
import torch
from scipy.sparse import lil_matrix
import toolz 
class Evaluator(object):

    def __init__(self, 
                num_users, 
                num_items, 
                train_user_item_matrix=None,
                test_user_item_matrix=None,
                on_train = False):
        
        self.on_train = on_train
        self.num_users = num_users
        self.num_items = num_items
        
        if train_user_item_matrix is not None:
            self.train_user_item_matrix = lil_matrix(train_user_item_matrix)
        else:
            self.train_user_item_matrix = None
        # print(self.train_user_matrix)

        self.test_user_item_matrix = lil_matrix(test_user_item_matrix)

        self.user_test_set = {u : set(self.test_user_item_matrix.rows[u])
                             for u in range(self.num_users) if self.test_user_item_matrix.rows[u]}
        if train_user_item_matrix is not None:
            self.user_train_set = {u: set(self.train_user_item_matrix.rows[u])
                            for u in range(self.num_users) if self.train_user_item_matrix.rows[u]}
            
            self.max_counts = max(len(row) for row in self.train_user_item_matrix.rows)

        else:
            self.user_train_set = defaultdict(dict)
            self.max_counts = 0
    def precision_recall_ndcg_k(self, model, users, k=10):
        
        recall_k, precision_k, ndcg_k = [], [], []
        for user_chunks in toolz.partition_all(100, users):
            k = int(k)
            _, preds = torch.topk(model.predict(np.asarray(user_chunks)), min(self.max_counts + int(k), self.num_items), dim = -1)
            
            preds = preds.tolist()
            for u, tops in zip(user_chunks, preds):
                
                test_set = self.user_test_set.get(u, set())
                train_set = self.user_train_set.get(u, set())

                if len(test_set) < k:
                    continue
             
                _idcg, _dcg = 0, 0
                for pos in range(k):
                    _idcg += 1.0 / math.log(pos + 2, 2)
                    
                new_set = []
                top_k = 0
                for val in tops:
                    if val in train_set and not self.on_train:
                        continue
                    top_k += 1
                    new_set.append(val)
                    if top_k >= k:
                        break
                hits = [(idx, val) for idx, val in enumerate(new_set) if val in test_set]
                cnt = len(hits)

                for idx in range(cnt):
                    _dcg += 1.0 / math.log(hits[idx][0] + 2, 2)
                    
                precision_k.append(float(cnt /k))
                recall_k.append(float(cnt / len(test_set)))
                ndcg_k.append(float(_dcg / _idcg))
        
        return np.mean(precision_k), np.mean(recall_k), np.mean(ndcg_k)


    
    def map_mrr_auc_ndcg(self, model, users, k = 3):
        '''
        k:
            the minimum number of k to evaluate
        '''
        MAP, MRR, auc, NDCG = [], [], [], []
        for user_chunks in toolz.partition_all(100, users):
            _, preds = torch.topk(model.predict(np.asarray(user_chunks)), self.num_items, dim = -1)
            preds = preds.tolist()
            for u, tops in zip(user_chunks, preds):
                train_set = self.user_train_set.get(u, set())
                test_set = self.user_test_set.get(u, set())

                if len(test_set) < k: continue

                new_set = []
                for val in tops:
                    if val in train_set and not self.on_train:
                        continue
                    new_set.append(val)

                _idcg, _dcg, _ap = 0, 0, 0

                for pos in range(len(test_set)):
                    _idcg += 1.0 / math.log(pos + 2, 2)
                
                hits = [(idx, val) for idx, val in enumerate(new_set) if val in test_set]
                cnt = len(hits)

                for id in range(cnt):
                    _ap += float((id + 1.0) / (hits[id][0] + 1))
                    _dcg += 1.0 / math.log(hits[id][0] + 2, 2)
                if cnt:
                    MAP.append(float(_ap / cnt))
                    NDCG.append(float(_dcg / _idcg))
                    MRR.append(float(1 / (hits[0][0] + 1)))
                else:
                    MAP.append(0.)
                    NDCG.append(0.)
                    MRR.append(0.)
                
                labels = [1 if item in test_set else 0 for item in new_set]
                auc.append(self.AUC(labels, len(test_set)))
        
        return np.mean(MAP), np.mean(MRR), np.mean(auc), np.mean(NDCG)

    def AUC(self,labels,_K):
        if len(labels) <= _K:
            return 1
        auc = 0
        for i, label in enumerate(labels[::-1]):
            auc += label * (i + 1)

        return (auc - _K * (_K + 1) / 2) / (_K * (len(labels) - _K))