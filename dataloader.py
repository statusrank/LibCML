import numpy as np 
import torch

class _DataLoader(object):
    
    def __init__(self, data, batch_size = 1, shuffle = False):
        super(_DataLoader, self).__init__()

        self.data = np.asarray(data)   
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.result_queue = []
        np.random.shuffle(self.data)

    def split_batch(self):
        if self.shuffle == True:
            np.random.shuffle(self.data)
        
        left = 1 if len(self.data) % self.batch_size else 0
        
        for i in range(len(self.data) // self.batch_size + left):
            r_min = min((i + 1) * self.batch_size, len(self.data))
            batch_data = self.data[i * self.batch_size: r_min]

            self.result_queue.append(batch_data)

    def _start(self):
        self.split_batch()
    def next_batch(self):
        if len(self.result_queue) == 0:
            raise ValueError("result_queue is empty!!!")

        return torch.from_numpy(self.result_queue.pop())

    def __len__(self):
        return len(self.result_queue)
