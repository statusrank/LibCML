import torch.utils.data as data 
from sampler import SamplerFactory
from scipy.sparse import lil_matrix
from torch.utils.data import Dataset

class SampleDataset(data.Dataset):

    def __init__(self, 
                 user_item_interactions,
                 num_negatives=10,
                 sample_method='uniform',
                 random_seed=1234):
        
        super(SampleDataset, self).__init__()

        """
        This class is adopted to generate (user, PosItem, NegItem) for training 
            according to the given sample method.
        """

        self.user_item_interactions = lil_matrix(user_item_interactions)
        
        self.sampler = SamplerFactory.generate_sampler(sample_method,
                                                       user_item_interactions,
                                                       num_negatives,
                                                       random_seed)
        self.user_pos_neg = self.sampler.sampling()
    
    def getuserCache(self):
        return {u: list(items) for u, items in enumerate(self.user_item_interactions.rows)}

    def getitemCache(self):
        item_user_interactions = lil_matrix(self.user_item_interactions.toarray().T)
        
        assert self.user_item_interactions.shape[1] == item_user_interactions.shape[0], 'error of transpose'

        return {i: list(users) for i, users in enumerate(item_user_interactions.rows)}

    def __len__(self):
        return len(self.user_pos_neg) 
    
    def __getitem__(self, idx):
        user = self.user_pos_neg[idx][0]
        pos_id = self.user_pos_neg[idx][1] 
        neg_id = self.user_pos_neg[idx][2]

        return (user, pos_id, neg_id)

    def generate_triplets_by_sampling(self):
        self.user_pos_neg = self.sampler.sampling()

'''The following dataset is merely applied to the Sampling-Free CML (SFCML) algorithm.
'''
class UserDataset(Dataset):
    def __init__(self, num_whole_users):
        self.num_whole_users = num_whole_users
    def __getitem__(self, index):
        return index
    def __len__(self):
        return self.num_whole_users

