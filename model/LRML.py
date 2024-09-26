'''
    Implements "Latent Relational Metric Learning" model from
    "Latent Relational Metric Learning via Memory-based Attention for Collaborative Ranking"
    Authors - Yi Tay, Luu Anh Tuan, Siu Cheung Hui
'''

import torch 
import torch.nn as nn 
import numpy as np 
import torch.nn.functional as F

class LRML(nn.Module):
    def __init__(self,
                num_users,
                num_items,
                num_mems=20,
                margin=1.0,
                dim=256,
                clip_max=1.0,
                **kwargs):
        super(LRML, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_mems = num_mems
        self.dim = dim
        self.margin = margin
        self.clip_max = clip_max

        self.user_embeddings = nn.Embedding(num_users, dim)# 
        self.item_embeddings = nn.Embedding(num_items, dim) # max_norm=clip_max
        self.user_item_key = nn.Embedding(dim, num_mems)
        self.memories = nn.Embedding(num_mems, dim)

        if kwargs is not None:
            self.__dict__.update(**kwargs)

    @torch.no_grad()
    def ClipNorm(self):
        pass
        self.user_embeddings.weight.data.div_(torch.norm(self.user_embeddings.weight.data, 2, 1, True).expand_as(
            self.user_embeddings.weight.data)).mul_(self.clip_max)
        
        self.item_embeddings.weight.data.div_(torch.norm(self.item_embeddings.weight.data, 2, 1, True).expand_as(
            self.item_embeddings.weight.data)).mul_(self.clip_max)

    def forward(self, user_ids, pos_ids, neg_ids):
        
        user_embeddings = self.user_embeddings(user_ids).cuda() # (batch, dim)
        pos_item_embeddings = self.item_embeddings(pos_ids).cuda() # (batch, dim)
        neg_item_embeddings = self.item_embeddings(neg_ids).cuda() # (batch, dim)

        user_pos_s = torch.mul(user_embeddings, pos_item_embeddings)# (batch, dim)
        attention_weight = F.softmax(user_pos_s.mm(self.user_item_key.weight), dim=-1) # (batch, N)

        latent_rel_vec = attention_weight.mm(self.memories.weight) # (batch, dim)

        pos_distances = (user_embeddings + latent_rel_vec - pos_item_embeddings).square().sum(-1) # (batch,)
        neg_distances = (user_embeddings + latent_rel_vec - neg_item_embeddings).square().sum(-1) # (batch,)

        loss = self.margin + pos_distances - neg_distances
        return nn.ReLU()(loss).sum()
    
    @torch.no_grad()
    def convert_to_tensor(self, arr):
        if not torch.is_tensor(arr):
            arr = torch.from_numpy(arr).long().cuda()
        return arr
    @torch.no_grad()
    def predict(self, users, items=None):
        '''
        This function is extremely time-consuming if the number of item is large.
        Use with caution!!!
        '''   
        if items is None:
            items = np.arange(self.num_items)
        
        user_ids = self.convert_to_tensor(users)
        item_ids = self.convert_to_tensor(items)

        user_embeddings = self.user_embeddings(user_ids).cuda() # (batch, dim)
        item_embeddings = self.item_embeddings(item_ids).cuda() # (num_items, dim)

        user_embeddings = user_embeddings.unsqueeze(1) # (batch, 1, dim)
        item_embeddings = item_embeddings.unsqueeze(0) # (1, num_items, dim)

        user_item_s = user_embeddings * item_embeddings # (batch, num_items, dim)
        user_item_key = self.user_item_key.weight.data.expand(user_ids.shape[0], self.dim, self.num_mems)
        
        attention_weight = torch.bmm(user_item_s, user_item_key) # (batch, num_items, num_mems)
        attention_weight = F.softmax(attention_weight, dim = -1) # (batch, num_items, num_mems)

        memories = self.memories.weight.data.expand(user_ids.shape[0], self.num_mems, self.dim)
        latent_rel_vec = torch.bmm(attention_weight, memories) # (batch, num_items, dim)

        distances = (user_embeddings + latent_rel_vec - item_embeddings).square().sum(-1) # (batch, num_items)
        return -distances
