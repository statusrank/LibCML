'''
Code implementation of paper:

Collaborative Preference Embedding against Sparse Labels. Shilong Bao, Qianqian Xu, Ke Ma, Zhiyong Yang, Xiaochun Cao and Qingming Huang. ACM International Conference on Multimedia (ACM-MM), 2019.

'''

import torch 
import torch.nn as nn 

class CPE(nn.Module):
    def __init__(self,
                 num_users, 
                 num_items,
                 margin=0.5,
                 dim=256,
                 max_norm=1.0,
                 cov_loss_reg = 0.): # Using Cov loss will slow down the training process heavily. Be careful to Enable it.
        super(CPE, self).__init__()

        self.num_users = num_users
        self.num_items = num_items 
        self.margin = margin
        self.dim = dim
        self.max_norm = max_norm

        # user embeddings
        self.user_embeddings = nn.Embedding(num_users, self.dim)
        nn.init.normal_(self.user_embeddings.weight, mean=0, std=0.01)

        # item embeddings
        self.item_embeddings = nn.Embedding(num_items, self.dim)
        nn.init.normal_(self.item_embeddings.weight, mean=0, std= 1.0 / (self.dim ** 0.5))

        self.cov_loss_reg = cov_loss_reg
        self.hinge_loss = nn.ReLU()

    def preference_loss(self, user_ids, pos_ids, neg_ids):
        # batch_size = user_ids.shape[0]

        user_embeddings = self.user_embeddings(user_ids).cuda() # (batch, dim)
        pos_embeddings = self.item_embeddings(pos_ids).cuda() # (batch, dim) 

        pos_distances = torch.square(user_embeddings - pos_embeddings).sum(-1) # (batch, )
        
        neg_embeddings = self.item_embeddings(neg_ids).cuda() # (batch, n_negatives, dim)
        user_embeddings_with_neg = user_embeddings.unsqueeze(1) # (batch, 1, dim)

        neg_distances = torch.square(user_embeddings_with_neg - neg_embeddings).sum(-1) # (batch, n_negatives)
        
        min_neg_distances, _ = torch.min(neg_distances, dim = -1) # (batch, ) 
        
        delta_p_n = min_neg_distances - pos_distances # (batch, )
        
        loss = self.hinge_loss(delta_p_n - self.margin).sum() + self.hinge_loss(self.margin - delta_p_n).sum()

        return loss
    
    @torch.no_grad()
    def ClipNorm(self):
        def normalize_embeddings(embedding):
            import torch.nn.functional as F
            with torch.no_grad():  
                weight = embedding.weight
                embedding.weight.copy_(F.normalize(weight, p=2, dim=1).clamp(max=1.0))
        normalize_embeddings(self.user_embeddings)
        normalize_embeddings(self.item_embeddings)

    def forward(self, user_ids, pos_ids, neg_ids):
        
        loss = self.preference_loss(user_ids, pos_ids, neg_ids)
        if self.cov_loss_reg > 0:
            loss += self.cov_loss_reg * self.Cov_loss()
        return loss
    
    def Cov_loss(self):
        user_embeddings = self.user_embeddings.weight.cuda()
        item_embeddings = self.item_embeddings.weight.cuda()

        concat_embeddings = torch.cat((user_embeddings, item_embeddings), 0).cuda() # (num_user+num_items, dim)
        
        # print("===> concat_embeddings: ", concat_embeddings.shape)
        
        concat_embeddings = concat_embeddings.cuda() - torch.mean(concat_embeddings.cuda(), 0, keepdim=True)
        cov_matrix = torch.matmul(concat_embeddings.transpose(0, 1), concat_embeddings) # (dim, dim)
        
        masks = torch.eye(cov_matrix.shape[0]).cuda()
        target_matrixs = cov_matrix * masks

        eig_values, _ = torch.linalg.eig(cov_matrix.cuda())

        log_determinant_of_C = torch.sum((eig_values + 1e-8).float().log()).cuda()

        return torch.trace(cov_matrix) - log_determinant_of_C - self.dim
        # return (cov_matrix - target_matrixs).square().sum()
    
    def predict(self, user_ids):
        
        if not torch.is_tensor(user_ids):
            user_ids = torch.from_numpy(user_ids).cuda()

        # (batch, dim) -> (batch, 1, dim)
        user_embeddings = self.user_embeddings(user_ids).cuda()
        user_embeddings = user_embeddings.unsqueeze(-2) 

        item_embeddings = self.item_embeddings.weight # (N, dim)
        item_embeddings = item_embeddings.cuda()
        item_embeddings = item_embeddings.view(1, self.num_items, self.dim) # (1, N, dim)

        scores = torch.square(user_embeddings - item_embeddings).sum(-1) # (batch, N, dim) -> (batch, N)
        
        return -scores