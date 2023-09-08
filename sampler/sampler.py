import numpy as np 
from collections import defaultdict
import torch as tc 
import torch.nn as nn 
from abc import abstractmethod
from scipy.sparse import lil_matrix

class Sampler:
    """
    A sampler is responsible for triplet sampling within a specific strategy
    :param name: sampler name
    :param model: current training model
    :param interactions: input user interactions in
           scipy.sparse.lil_matrix format
    :param n_workers: number of workers
    :param n_negatives: number of negatives
    :param batch_size: batch size
    :param kwargs: optional keyword arguments
    """

    @classmethod 
    def _get_popularity(cls, interactions):
        popularity_dict = defaultdict(set)
        for uid, iids in enumerate(interactions.rows):
            for iid in iids:
                popularity_dict[iid].add(uid)

        popularity_dict = {
            key: len(val) for key, val in popularity_dict.items()
        }
        return popularity_dict
    
    def __init__(self, 
                 sampling_strategy, 
                 interactions, 
                 n_negatives=10, 
                 random_seed=1234,
                 **kwargs):

        if sampling_strategy not in ['uniform', 'pop', '2st', 'hard']:
            raise ValueError('only support [uniform, pop, 2st] now!')

        self.sampling_strategy = sampling_strategy
        self.interactions = lil_matrix(interactions)
        self.n_negatives = n_negatives
        self.random_seed = random_seed
        self.neg_alpha = 1.0

        if kwargs is not None:
            self.__dict__.update(kwargs) 

        # user positive item dictionary, use to check
        # if an item is a positive one
        self.user_items = {uid: set(iids) for uid, iids in enumerate(
            self.interactions.rows)}

        # get item popularities
        if self.sampling_strategy in ['pop', '2st']:
            self.item_counts = self._get_popularity(self.interactions)

            if self.neg_alpha != 1.0:
                self.item_counts = {iid: np.power(freq, self.neg_alpha)
                                    for iid, freq in self.item_counts.items()}

            total_count = np.sum(list(self.item_counts.values()))

            self.item_popularities = np.zeros(self.interactions.shape[1],
                                              dtype=np.float32)
            for iid in range(self.interactions.shape[1]):
                if iid in self.item_counts:
                    self.item_popularities[iid] = float(
                        self.item_counts[iid]) / total_count
                else:
                    self.item_popularities[iid] = 0.0
        
        # self.rng = np.random.RandomState(random_seed)

        # self.sampling()

    def sampling(self):
        """
        Sampling a batch of training samples!

        :return: batch (user, pos_item, neg_items)
        """

        # positive user item pairs
        user_positive_item_pairs = np.asarray(self.interactions.nonzero()).T
        
        return self._negative_sampling(user_positive_item_pairs)
        
    @abstractmethod
    def _candidate_neg_ids(self):
        """
        Candidate for negative ids
        :param pos_ids: batch positive ids
        :return:
        """
        return np.arange(self.interactions.shape[1])

    @abstractmethod    
    def _negative_sampling(self, user_ids, pos_ids, neg_ids):
        """
        Negative sampling
        :param user_ids:
        :param pos_ids:
        :param neg_ids:
        :return:
        """
        raise NotImplementedError(
            '_negative_sampling method should be implemented in child class')
    
