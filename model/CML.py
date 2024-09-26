'''
Code implementation of paper:

The Minority Matters: A Diversity-Promoting Collaborative Metric Learning Algorithm. 
Shilong Bao, Qianqian Xu, Zhiyong Yang , Yuan He, Xiaochun Cao, Qingming Huang. 
Advances in Neural Information Processing Systems (NeurIPS), 2022. (Oral, 1.7%) 

'''
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class CML(nn.Module):
    '''
    Note that, if one set per_user_k=1, then COCML is degraded tp the conventional CML.
    '''
    def __init__(self,
                 num_users, 
                 num_items,
                 margin=2.0,
                 DCRS_reg=10,
                 m1=0.05,
                 m2=0.25,
                 dim=100,
                 per_user_k=3,
                 max_norm=1.0):
        super(CML, self).__init__()

        self.num_users = num_users
        self.num_items = num_items 

        assert per_user_k != 0, 'per_user_k should be greater than zero!'

        self.per_user_embed_k = per_user_k

        self.margin = margin
        self.dim = dim
        self.max_norm = max_norm
        
        self.reg = DCRS_reg
        self.m1 = m1 
        self.m2 = m2

        # user embeddings
        self.user_embeddings = nn.Embedding(num_users, self.per_user_embed_k * self.dim)
        nn.init.normal_(self.user_embeddings.weight, mean=0, std=0.01)

        # item embeddings
        self.item_embeddings = nn.Embedding(num_items, self.dim)
        nn.init.normal_(self.item_embeddings.weight, mean=0, std= 1.0 / (self.dim ** 0.5))

    def ClipNorm(self):
        def normalize_embeddings(embedding):
            import torch.nn.functional as F
            with torch.no_grad():  
                weight = embedding.weight
                embedding.weight.copy_(F.normalize(weight, p=2, dim=1).clamp(max=1.0))
        
        normalize_embeddings(self.item_embeddings)
        with torch.no_grad():
            
            user_embeddings_weight = self.user_embeddings.weight.data
            user_embeddings_weight = user_embeddings_weight.view(self.num_users, 
                                                                 self.per_user_embed_k, 
                                                                 self.dim)

            user_embeddings_weight *= self.max_norm / torch.norm(user_embeddings_weight, 
                                                                p=2, 
                                                                dim=-1,
                                                                keepdim=True)

            self.user_embeddings.weight.data = user_embeddings_weight.view(self.num_users, -1)


    def preference_loss(self, user_ids, pos_ids, neg_ids):
        pass

    def forward(self, user_ids, pos_ids, neg_ids):
        
        loss = self.preference_loss(user_ids, pos_ids, neg_ids)
        
        if self.reg and self.m1 < self.m2 :
            loss += self.reg * self.DCRS()

        return loss
    
    def DCRS(self):
        user_embeddings = self.user_embeddings.weight.cuda() # (batch, k * dim)
        user_embeddings = user_embeddings.view(self.num_users, self.per_user_embed_k, self.dim)

        user_embeddings_1 = user_embeddings.unsqueeze(-2) # (M, k, 1, d)
        user_embeddings_2 = user_embeddings.unsqueeze(1)  # (M, 1, k, d)

        diversity = torch.square(user_embeddings_1 - user_embeddings_2).sum(-1) # (M, k, k)
        
        mask = 1.0 - torch.eye(self.per_user_embed_k).unsqueeze(0) # (1, k, k)

        return F.relu((self.m1 - diversity) * mask.cuda()).mean() + F.relu((diversity - self.m2) * mask.cuda()).mean()
    
    def predict(self, user_ids):
        
        # return (batch, num_items)
        if not torch.is_tensor(user_ids):
            user_ids = torch.from_numpy(user_ids).cuda()

        # (batch, k*dim) -> (batch, k, dim) ->  (batch, k, 1, dim)
        user_embeddings = self.user_embeddings(user_ids).cuda()
        user_embeddings = user_embeddings.view(user_ids.shape[0], self.per_user_embed_k, self.dim).unsqueeze(-2) 

        item_embeddings = self.item_embeddings.weight # (N, dim)
        item_embeddings = item_embeddings.cuda()
        item_embeddings = item_embeddings.view(1, 1, self.num_items, self.dim) # (1, 1, N, dim)

        scores = torch.square(user_embeddings - item_embeddings).sum(-1) # (batch, k, N)
        scores, _ = torch.min(scores, dim=1) # (batch, N)

        return -scores

# DPCML1 in the paper
class COCML(CML):
    def __init__(self,
                 num_users, 
                 num_items,
                 margin=2.0,
                 DCRS_reg=10,
                 m1=0.05,
                 m2=0.25,
                 dim=100,
                 per_user_k=3,
                 max_norm=1.0):
        super(COCML, self).__init__(num_users, num_items, margin, DCRS_reg, m1, m2, dim, per_user_k, max_norm)

    def preference_loss(self, user_ids, pos_ids, neg_ids):
        
        batch_size = user_ids.shape[0]

        user_embeddings = self.user_embeddings(user_ids).cuda() # (batch, k * dim)
        pos_embeddings = self.item_embeddings(pos_ids).cuda()
        neg_embeddings = self.item_embeddings(neg_ids).cuda()

        user_embeddings = user_embeddings.view(batch_size, self.per_user_embed_k, self.dim) # (batch, k, dim)
        pos_embeddings = pos_embeddings.unsqueeze(1).expand_as(user_embeddings) # (bacth, k, dim)
        neg_embeddings = neg_embeddings.unsqueeze(1).expand_as(user_embeddings) # (bacth, k, dim)

        pos_distances = torch.square(user_embeddings - pos_embeddings).sum(-1) # (batch, k)
        neg_distances = torch.square(user_embeddings - neg_embeddings).sum(-1) # (batch, k)
        
        min_pos_distances, _ = torch.min(pos_distances, -1) # return min value and indices
        min_neg_distances, _ = torch.min(neg_distances, -1)
        
        embedding_loss = self.margin + min_pos_distances - min_neg_distances  # (batch, )
        
        loss = nn.ReLU()(embedding_loss).sum()

        return loss

# DPCML2 in the paper
class HarCML(CML):
    def __init__(self,
                 num_users, 
                 num_items,
                 margin=2.0,
                 DCRS_reg=10,
                 m1=0.05,
                 m2=0.25,
                 dim=100,
                 per_user_k=3,
                 max_norm=1.0):

        super(HarCML, self).__init__(num_users, num_items, margin, DCRS_reg, m1, m2, dim, per_user_k, max_norm)
    
    def preference_loss(self, user_ids, pos_ids, neg_ids):
        
        batch_size = user_ids.shape[0]

        user_embeddings = self.user_embeddings(user_ids).cuda() # (batch, k * dim)
        pos_embeddings = self.item_embeddings(pos_ids).cuda() # (batch, dim) 

        user_embeddings = user_embeddings.view(batch_size, self.per_user_embed_k, self.dim) # (batch, k, dim)
        pos_embeddings = pos_embeddings.unsqueeze(1).expand_as(user_embeddings) # (bacth, k, dim)
        pos_distances = torch.square(user_embeddings - pos_embeddings).sum(-1) # (batch, k)
        
        neg_embeddings = self.item_embeddings(neg_ids).cuda() # (batch, n_negatives, dim)
        neg_embeddings = neg_embeddings.unsqueeze(1) # (batch, 1, n_negatives, dim)
        user_embeddings_with_neg = user_embeddings.unsqueeze(2) # (batch, k, 1, dim)

        neg_distances = torch.square(user_embeddings_with_neg - neg_embeddings).sum(-1) # (batch, k, n_negatives)
        
        # return min value and indices
        min_pos_distances, _ = torch.min(pos_distances, dim = -1) # (batch, )
        min_neg_distances, _ = torch.min(neg_distances, dim = -2) # (batch, n_negatives) 

        min_neg_distances, _ = torch.min(min_neg_distances, dim=-1) # (batch, )
        
        embedding_loss = self.margin + min_pos_distances - min_neg_distances  # (batch, )
        
        loss = nn.ReLU()(embedding_loss).sum()

        return loss