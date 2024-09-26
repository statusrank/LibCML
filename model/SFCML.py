'''
Code implementation of paper:

Rethinking Collaborative Metric Learning: Toward an Efficient Alternative without Negative Sampling. Shilong Bao, Qianqian Xu, Zhiyong Yang, Xiaochun Cao and Qingming Huang. IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), 2022.

'''

import torch 
import torch.nn as nn 
import numpy as np 

class SFCML(nn.Module):

    def __init__(self,
                 args, 
                 num_users, 
                 num_items,
                 user_item_matrix,
                 margin,
                 user_max_norm = 1.0,
                 item_const_norm = 1.0,
                 device = torch.device('cpu')
                 ):
        super(SFCML, self).__init__()

        self.args = args
        self.dim = args.dim   
        self.num_users = num_users
        self.num_items = num_items
        self.margin = margin
        self.device = device
        self.item_const_norm = item_const_norm
        self.user_max_norm = user_max_norm

        # user_embedding = r1
        self.user_embeddings = nn.Embedding(num_users, self.dim)
        # self.user_embeddings.weight.data.normal_(0, 0.1)

        self.ClipUserNorm()

        # item embedding = r2
        self.item_embeddings = nn.Embedding(num_items, self.dim)
        # self.item_embeddings.weight.data.normal_(0, 0.1)
        self.ClipItemNorm()

        # target -> (num_users, num_items)
        # self.target = torch.from_numpy(user_item_matrix.todense().astype(np.float))

        self.user_item_matrix = user_item_matrix # np.array()

    def forward(self, users):
        
        pred = self.predict(users).to(self.device) # (batch_users, num_items)
        # target = self.target[users].to(self.device) # (batch_users, num_items)

        # # (batch_users, num_items)
        target = torch.from_numpy(self.user_item_matrix[users.cpu().numpy()].todense().astype(np.float32)).to(self.device)
        
        # print("shape of target: ", target.shape)

        numPos = target.sum(1, keepdim=True) # (batch_users, 1)
        numNeg = (1 - target).sum(1, keepdim=True) # (batch_users, 1)
        
        # print(numPos.shape)
        # print(numNeg.shape)

        Dp, Dn = 1.0 / numPos, 1.0 / numNeg
        Yp, Yn = Dp.mul(target).to(self.device), Dn.mul(1 - target).to(self.device) # (batch_users, num_items)

        return self.FastSqPairwiseLoss(pred, target, Yp, Yn)

    def FastSqPairwiseLoss(self, pred, target, Yp, Yn):
        '''
        pred: 
            score = u^T \dot v
            [batch_users, num_items]
            loss: 
                l(pred, gt) = (1 - t)^2 
        target:  must be two classes
            [batch_users, num_items]
        
        Yp, Yn:
            [batch_users, num_items]
        '''
        pred = pred.cuda()
        target = target.cuda()
        Yp  = Yp.cuda()
        Yn = Yn.cuda()
        # batch_size = pred.shape[0]
        diff = pred - self.margin * target # (batch_users, num_items)
        weight = Yp + Yn # (batch_users, num_items)
        A = diff.mul(weight).mul(diff).sum(1, keepdim=True) # (batch_users, 1)
        B = (diff.mul(Yn).sum(1, keepdim=True)) * (Yp.mul(diff).sum(1, keepdim=True)) # (batch_users, 1)
        return torch.mean(A - 2 * B)

    @torch.no_grad()
    def ClipUserNorm(self):

        self.user_embeddings.weight.data = self.user_embeddings.weight.data * self.user_max_norm / torch.norm(self.user_embeddings.weight.data, 
                                                                                       p=2, 
                                                                                       dim=-1, 
                                                                                       keepdim=True) 
        
        assert self.user_embeddings.weight.requires_grad == True, 'Need grad to update!'
    @torch.no_grad()
    def ClipItemNorm(self):

        self.item_embeddings.weight.data = self.item_embeddings.weight.data * self.item_const_norm / torch.norm(self.item_embeddings.weight.data, 
                                                                                       p=2, 
                                                                                       dim=-1, 
                                                                                       keepdim=True) 
        
        assert self.item_embeddings.weight.requires_grad == True, 'Need grad to update!'
    def predict(self, users):
        if not torch.is_tensor(users):
            users = torch.from_numpy(users).long().to(self.device)
        user_weight = self.user_embeddings(users).cuda()
        item_weight = self.item_embeddings.weight.cuda()
        return 2 * user_weight.mm(item_weight.T)