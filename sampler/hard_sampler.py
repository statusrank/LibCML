from operator import ne
import numpy as np 
from .sampler import Sampler
from tqdm import tqdm

class HardSampler(Sampler):
    """
    generate negative samples using uniform distribution.
    """
    def _negative_sampling(self, user_item_pairs):

        sampling_triplets = []
        for user_id, pos in tqdm(user_item_pairs, desc='===> generate negative samplings...'):

            candidate_neg_ids = self._candidate_neg_ids()
            neg_samples = np.random.choice(candidate_neg_ids,
                                            size=self.n_negatives)
            for j, neg in enumerate(neg_samples):
                while neg in self.user_items[user_id]:
                    neg_samples[j] = neg = np.random.choice(candidate_neg_ids)
                
            sampling_triplets.append((user_id, pos, neg_samples))
        
        return sampling_triplets