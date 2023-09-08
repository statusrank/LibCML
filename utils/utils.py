import numpy as np 
import torch 
from collections import defaultdict 
from tqdm import tqdm
import argparse
from Log import MyLog
import os 
from sklearn.model_selection import train_test_split
import json 
from scipy.sparse import dok_matrix,lil_matrix

def load_data(args, mylog, data_name = 'users.dat', threholds=5):
    data_path = os.path.join(args.data_path, data_name)

    user_dict = defaultdict(set)
    # user_prob = []
    num_users, num_items = 0, 0
    for u, u_liked_items in enumerate(open(data_path).readlines()):
        
        items = u_liked_items.strip().split()
        if int(items[0]) < threholds:
            continue

        user_dict.setdefault(num_users, set())

        for item in items[1:]:
            user_dict[num_users].add(int(item))
            num_items = max(num_items, int(item) + 1)
        
        num_users += 1

    mylog.info('number of users: {}'.format(num_users))
    mylog.info('number of items: {}'.format(num_items))
    
    _num_users = 0
    user_item_matrix = dok_matrix((num_users, num_items), dtype=np.int32)
    for u, u_liked_items in enumerate(open(data_path).readlines()):
        items = u_liked_items.strip().split()

        if int(items[0]) < threholds:
            continue
        for item in items[1: ]:
            user_item_matrix[_num_users, int(item)] = 1
        _num_users += 1
    
    assert num_users == _num_users

    return user_item_matrix, num_users, num_items

def split_train_val_test(user_item_matrix, args, cur_log, threholds = 5):

    # set seed to have deterministic results.
    np.random.seed(args.random_seed)

    train_matrix = dok_matrix(user_item_matrix.shape)
    val_matrix = dok_matrix(user_item_matrix.shape)
    test_matrix = dok_matrix(user_item_matrix.shape)
    user_prob = []
    user_item_matrix = lil_matrix(user_item_matrix)
    num_users = user_item_matrix.shape[0]
    num_items = user_item_matrix.shape[1]

    for u in tqdm(range(num_users), desc = "Split data into train/valid/test"):
        items = list(user_item_matrix.rows[u])

        if len(items) < threholds: continue

        np.random.shuffle(items)

        train_count = int(len(items) * args.split_ratio[0] / sum(args.split_ratio))
        valid_count = int(len(items) * args.split_ratio[1] / sum(args.split_ratio))
        
        for i in items[0:train_count]:
            train_matrix[u, i] = 1
        for i in items[train_count:train_count + valid_count]:
            val_matrix[u, i] = 1
        for i in items[train_count + valid_count:]:
            test_matrix[u ,i] = 1

    cur_log.info("total interactions: {}".format(len(train_matrix.nonzero()[0]) + len(val_matrix.nonzero()[0]) + len(test_matrix.nonzero()[0])))
    cur_log.info("split the data into trian/validatin/test {}/{}/{} ".format(
        len(train_matrix.nonzero()[0]),
        len(val_matrix.nonzero()[0]),
        len(test_matrix.nonzero()[0])))
    return train_matrix, val_matrix, test_matrix

def set_seeds(random_seed = 1234):
    
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    torch.manual_seed(random_seed)