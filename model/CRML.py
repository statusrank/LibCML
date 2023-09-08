from typing import Type
import torch
from torch._C import dtype 
import torch.nn as nn 
import numpy as np
from torch.nn.modules import distance 
from scipy.sparse import lil_matrix

class CRML(nn.Module):
    def __init__(self,
                user_item_interactions,
                num_users,
                num_items,
                margin=2.0,
                dim=256,
                alpha=0.01,
                beta=0.01,
                C_min=1,
                C_max=100,
                lam=3/4,
                clip_max=1.0):
        super(CRML, self).__init__()

        self.user_item_interactions = lil_matrix(user_item_interactions, dtype=np.float32)
        self.num_users = num_users
        self.num_items = num_items
        self.dim = dim
        self.margin = margin
        self.C_min = C_min
        self.C_max = C_max
        self.lam = lam
        self.clip_max = clip_max
        self.alpha = alpha
        self.beta = beta

        self.user_embeddings = nn.Embedding(self.num_users, self.dim, max_norm=self.clip_max)
        self.item_embeddings = nn.Embedding(self.num_items, self.dim, max_norm=self.clip_max)

        self.co_user_embeddings = nn.Embedding(self.num_users, self.dim, max_norm=self.clip_max)
        self.co_user_bias = nn.Parameter(torch.randn([self.num_users, 1]))

        self.co_item_embeddings = nn.Embedding(self.num_items, self.dim, max_norm=self.clip_max)
        self.co_item_bias = nn.Parameter(torch.randn([self.num_items, 1]))

        self.user_interactions_counts = {u: set(row) for u, row in enumerate(self.user_item_interactions.rows)}

        if user_item_interactions is not None:
            self.item_user_matrix = lil_matrix(self.user_item_interactions.toarray().T)
            self.item_ineraction_counts = {i: set(row) for i, row in enumerate(self.item_user_matrix.rows)}
        else:
            self.item_user_matrix = None
            self.item_ineraction_counts = None

    @torch.no_grad()
    def ClipNorm(self):
        self.user_embeddings.weight.data.div_(torch.norm(self.user_embeddings.weight.data, 2, 1, True).expand_as(
            self.user_embeddings.weight.data)).mul_(self.clip_max)
        self.co_user_embeddings.weight.data.div_(torch.norm(self.co_user_embeddings.weight.data, 2, 1, True).expand_as(
            self.co_user_embeddings.weight.data)).mul_(self.clip_max)
        
        self.item_embeddings.weight.data.div_(torch.norm(self.item_embeddings.weight.data, 2, 1, True).expand_as(
            self.item_embeddings.weight.data)).mul_(self.clip_max)
        self.co_item_embeddings.weight.data.div_(torch.norm(self.co_item_embeddings.weight.data, 2, 1, True).expand_as(
            self.co_item_embeddings.weight.data)).mul_(self.clip_max)

    @torch.no_grad()
    def get_co_occurrence(self, arrs, interaction_counts):
        
        if not torch.is_tensor(arrs):
            raise TypeError

        N = arrs.shape[0]
        co_matirx = np.ones((N, N))
        np_arrs = arrs.cpu().numpy()
        for i in range(N):
            for j in range(i + 1, N):
                # print(len(interaction_counts[np_arrs[i]] & interaction_counts[np_arrs[j]]))
                insert_count = len(interaction_counts[np_arrs[i]] & interaction_counts[np_arrs[j]])
                # print(insert_count)
                co_matirx[i][j] = co_matirx[j][i] = insert_count if insert_count else 1.

        return torch.from_numpy(co_matirx).cuda()
    def forward(self, user_ids, pos_ids, neg_ids):
        '''
        param:
            user_ids: (batch, )
            pos_ids: (batch, )
            neg_ids: (batch, num_negs)
        '''
        user_embeddings = self.user_embeddings(user_ids).cuda() # (batch, dim)
        pos_item_embeddings = self.item_embeddings(pos_ids).cuda() # (batch, dim)

        user_pos_distances = (user_embeddings - pos_item_embeddings).square().sum(-1) # (batch, )

        neg_item_embeddings = self.item_embeddings(neg_ids).cuda() # (batch, k, dim); each (user, pos_id) sample k negative items

        user_neg_distances = (user_embeddings.unsqueeze(1) - neg_item_embeddings).square().sum(-1) # (batch, k)

        min_user_neg_distances, _ = torch.min(user_neg_distances, dim = -1) # (batch, )
        
        loss_0 = nn.ReLU()(self.margin + user_pos_distances - min_user_neg_distances).mean()

        users = user_ids.unique(sorted=True).long().cuda() # (N, )
        items = pos_ids.unique(sorted=True).long().cuda() # (M, )

        # print("users: ", users)
        # print('items: ', items)
        co_users_x = self.co_user_embeddings(users.long()).cuda() # (N, dim)
        co_users_bias = self.co_user_bias[users.long()].cuda() # (N, 1)

        users_products = co_users_x.mm(co_users_x.T) # (N, N)

        # print("users_product: ", users_products)

        user_sum_bias = co_users_bias.unsqueeze(1) + co_users_bias.unsqueeze(0) # (N, 1) + (1, N) -> (N, N)

        # print("user_sum_bias: ", user_sum_bias)

        user_co_matrix = self.get_co_occurrence(users, self.user_interactions_counts) # (N, N)

        # print("user_co_matrix: ", user_co_matrix)

        users_weight = user_co_matrix.div(self.C_max).pow(self.lam) # (N, N)
        users_weight = torch.where(user_co_matrix.le(self.C_min), 0., users_weight)
        users_weight = torch.where(user_co_matrix.ge(self.C_max), 1., users_weight)

        # print("users_weight: ", users_weight)

        loss_1 = (users_weight * (users_products + user_sum_bias - user_co_matrix.log()).square()).mean()

        loss_1 += self.alpha * (self.user_embeddings(users) - co_users_x).square().mean()

        co_items_y = self.co_item_embeddings(items).cuda() # (M, dim)
        co_items_bias = self.co_item_bias[items].cuda() # (M, 1)

        items_prodducts = co_items_y.mm(co_items_y.T).cuda() # (M, dim)
        item_sum_bias = co_items_bias.unsqueeze(1) + co_items_bias.unsqueeze(0) # (M, 1) + (1, M) -> (M, M)
        item_co_matrix = self.get_co_occurrence(items, self.item_ineraction_counts)
        items_weight = item_co_matrix.div(self.C_max).pow(self.lam)
        items_weight = torch.where(item_co_matrix.le(self.C_min), 0., items_weight)
        items_weight = torch.where(item_co_matrix.ge(self.C_max), 1., items_weight)

        loss_2 = (items_weight * (items_prodducts + item_sum_bias - item_co_matrix.log()).square()).mean()
        loss_2 += self.beta * (self.item_embeddings(items) - co_items_y).square().mean()

        # print("loss_0: ", loss_0.item())
        # print("loss_1: ", loss_1.item())
        # print("loss_2: ", loss_2.item())

        return loss_0 + loss_1 + loss_2

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

        distance = (user_embeddings.unsqueeze(1) - item_embeddings.unsqueeze(0)).square().sum(-1)
        return -distance

