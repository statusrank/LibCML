'''
    Chanyoung Park, Donghyun Kim, Xing Xie, Hwanjo Yu. 
    Collaborative Translational Metric Learning. ICDM 2018: 367-376.
    Modified from their official code: https://github.com/pcy1302/TransCF
    Thanks the great contributions of the authors.
'''

import torch 
import torch.nn as nn 
import numpy as np 

class TransCF(nn.Module):
    def __init__(self,
                num_users,
                num_items,
                margin,
                dim,
                dataset=None,
                clip_max=1.0,
                mode='mean',
                dis_reg=0.0,
                nei_reg=0.0,
                **kwargs):

        super(TransCF, self).__init__()

        self.num_users = num_users
        self.num_items = num_items

        self.margin = margin
        self.dim = dim 
        self.clip_max = clip_max

        self.user_embeddings = nn.EmbeddingBag(self.num_users, self.dim, mode=mode)
        self.item_embeddings = nn.EmbeddingBag(self.num_items, self.dim, mode=mode)
        
        self.userCache = dataset.getuserCache()
        self.itemCache = dataset.getitemCache()

        # if the value greater than zero, we conduct the regularizations.
        self.dis_reg = dis_reg
        self.nei_reg = nei_reg
        if kwargs is not None:
            self.__dict__.update(**kwargs)
    
    def init_weights(self):
        nn.init.normal(self.user_embeddings.weight.data, mean=0.0, std=0.01)
        nn.init.normal(self.item_embeddings.weight.data, mean=0.0, std=0.01)

    @torch.no_grad()
    def getNeighbors(self, uids, iids):
        uid_idxvec = []
        uid_offset = []
        prev_len = 0
        
        iids = iids.cpu().data.numpy().tolist()

        for iid in iids:
            users = self.itemCache[iid]
            uid_idxvec += users
            uid_offset.append(prev_len)
            prev_len += len(users)
            
            
        iid_idxvec = []
        iid_offset = []
        prev_len = 0
        uids = uids.cpu().data.numpy().tolist()
            
        for uid in uids:
            items = self.userCache[uid]
            iid_idxvec += items
            iid_offset.append(prev_len)
            prev_len += len(items)
        
        return torch.LongTensor(iid_idxvec).cuda(), \
               torch.LongTensor(iid_offset).cuda(), \
               torch.LongTensor(uid_idxvec).cuda(), \
               torch.LongTensor(uid_offset).cuda()

    @torch.no_grad()
    def ClipNorm(self):
        pass
        # self.user_embeddings.weight.data.div_(torch.norm(self.user_embeddings.weight.data, 2, 1, True).expand_as(
        #     self.user_embeddings.weight.data)).mul_(self.clip_max)
        
        # self.item_embeddings.weight.data.div_(torch.norm(self.item_embeddings.weight.data, 2, 1, True).expand_as(
        #     self.item_embeddings.weight.data)).mul_(self.clip_max)

    def forward(self, user_ids, pos_ids, neg_ids):
        '''
        params:
            user_ids: user ids
            pos_ids: positive item ids corresponding to the user one by one
            neg_ids: negative item ids corresponding to the user one by one
        '''        
        # users
        userIdx = torch.LongTensor(range(0, len(user_ids))).cuda()
        batch_user_embeddings = self.user_embeddings(user_ids.long(), userIdx).cuda()

        # for positive items
        pos_itemIdx = torch.LongTensor(range(0, len(pos_ids))).cuda()
        batch_pos_item_embeddings = self.item_embeddings(pos_ids.long(), pos_itemIdx).cuda()

        # Get neighborhood embeddings for pos_ids
        i_viewed_u_idx, i_viewed_u_offset, u_viewed_i_idx, u_viewed_i_offset = self.getNeighbors(user_ids, pos_ids)
        pos_user_neighbor_embeddings = self.item_embeddings(i_viewed_u_idx, i_viewed_u_offset).cuda()
        pos_item_neighbor_embeddings = self.user_embeddings(u_viewed_i_idx, u_viewed_i_offset).cuda()

        # Get r_{ui} between user and pos items
        rel = pos_user_neighbor_embeddings * pos_item_neighbor_embeddings

        pos_distances = (batch_user_embeddings + rel - batch_pos_item_embeddings)**2
        
        # for negative items
        neg_itemIdx = torch.LongTensor(range(0,len(neg_ids))).cuda()
        batch_neg_item_embeddings = self.item_embeddings(neg_ids.long(), neg_itemIdx).cuda()

        # Get neighborhood embeddings for neg_ids
        neg_viewed_u_idx, neg_viewed_u_offset, u_viewed_neg_idx, u_viewed_neg_offset = self.getNeighbors(user_ids, neg_ids)
        neg_user_neighbor_embeddings = self.item_embeddings(neg_viewed_u_idx, neg_viewed_u_offset).cuda()
        neg_item_neighbor_embeddings = self.user_embeddings(u_viewed_neg_idx, u_viewed_neg_offset).cuda()

        # Get r_{ui} between user and neg items
        rel = neg_user_neighbor_embeddings * neg_item_neighbor_embeddings

        neg_distances = (batch_user_embeddings + rel - batch_neg_item_embeddings)**2
        
        loss = self.margin + pos_distances - neg_distances

        # hinge-loss
        loss = nn.ReLU()(loss).sum()

        reg = 0.0
        # Distance Regularizer
        if self.dis_reg > 0:
            reg += self.dis_reg * pos_distances.sum()

        if self.nei_reg > 0:
            reg += self.nei_reg * ((batch_user_embeddings - pos_user_neighbor_embeddings).square().sum() + (
                batch_pos_item_embeddings - pos_item_neighbor_embeddings).square().sum())
        
        loss += reg
        return loss

    @torch.no_grad()
    def convert_to_tensor(self, arr):
        if not torch.is_tensor(arr):
            arr = torch.from_numpy(arr).long().cuda()
        return arr
    @torch.no_grad()
    def predict(self, users, items=None):
        '''
        This function is extremely time-consuming. 
        Use with caution!!!
        '''   
        if items is None:
            items = np.arange(self.num_items)
        
        user_ids = self.convert_to_tensor(users)
        item_ids = self.convert_to_tensor(items)

        items_viewed_u_idx, items_viewed_u_offset, u_viewed_item_idx, u_viewed_item_offset = self.getNeighbors(user_ids, item_ids)

        userIdx = torch.LongTensor(range(0,len(user_ids))).cuda()
        user_embeddings = self.user_embeddings(user_ids, userIdx).cuda() # (batch, dim)
        user_embeddings = user_embeddings.unsqueeze(1) # (batch, 1, dim)

        itemIdx = torch.LongTensor(range(0,len(item_ids))).cuda()
        item_embeddings = self.item_embeddings(item_ids, itemIdx).cuda() # (N, dim)
        item_embeddings = item_embeddings.unsqueeze(0) # (1, N, dim)

        user_neighbor_embeddings = self.item_embeddings(items_viewed_u_idx, items_viewed_u_offset).cuda() # (batch, dim)
        item_neighbor_embeddings = self.user_embeddings(u_viewed_item_idx, u_viewed_item_offset).cuda() # (N, dim)

        user_neighbor_embeddings = user_neighbor_embeddings.unsqueeze(1) # (batch, 1, dim)
        item_neighbor_embeddings = item_neighbor_embeddings.unsqueeze(0) # (1, N, dim)

        rel = user_neighbor_embeddings * item_neighbor_embeddings # (bacth, N, dim) 

        # print(rel.shape)

        distances = (user_embeddings + rel - item_embeddings).square().sum(-1) # (bacth, N)
        return -distances