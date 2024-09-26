# -*- coding: utf-8 -*-
'''
CopyRight: Shilong Bao
Email: baoshilong@iie.ac.cn
'''
import torch 
import numpy as np 
import os 
from Log import MyLog
import gc 
from dataset import SampleDataset, UserDataset
from evaluator import * 
from torch.utils.data import DataLoader
from model import TransCF, LRML, SFCML, CPE, COCML, HarCML, CRML
from utils import *
from torch.optim import Adam, Adagrad,lr_scheduler
from scipy import sparse
from scipy.sparse import coo_matrix, dok_matrix
from tqdm import tqdm
torch.cuda.empty_cache()
gc.collect()

device = torch.device('cuda')
SUPPORT_MODEL = {
    'CPE': CPE,
    'COCML': COCML,
    'HarCML': HarCML,
    'TransCF': TransCF,
    'LRML': LRML,
    'SFCML': SFCML,
    'CRML': CRML
}

OPT_DICTS = {
        'CPE': Adagrad,
        'COCML': Adam,
        'HarCML': Adam,
        'TransCF': Adagrad,
        'LRML': Adagrad,
        'SFCML': Adagrad,
        'CRML': Adam
    }

ALL_SAMS = {
    'CPE': 'hard',
    'COCML': 'uniform',
    'HarCML': 'hard',
    'TransCF': 'uniform',
    'LRML': 'uniform',
    'SFCML': None,
    'CRML': 'uniform'
}
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run Recommender")
    parser.add_argument('--model', nargs='?', help='Choose a recommender.', required=True) # 'TransCF', 'CRML', 'LRML', 'SFCML', 'CPE', 'COCML', 'HarCML'
    parser.add_argument('--data_path', nargs='?', help='Choose a dataset.', required=True)        
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate.', required=True)
    parser.add_argument('--topks', nargs='?', default=[3, 5, 10, 20, 30, 50], help="topK")
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--num_negs', type=int, default=10, help='Number of negative samples.')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--dim', type=int, default=100, help='Number of embedding dimensions.')
    parser.add_argument('--random_seed', type=int, default=1234, help='Random seed.')
    # parser.add_argument('--sampling_strategy', type=str, default='uniform')
    parser.add_argument('--split_ratio', type=tuple, default=(3,1,1), help='fraction to split the data')
    parser.add_argument('--max_norm', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--test', action='store_true', default=False, help='if true run the test program')
    parser.add_argument('--eval_user_nums', type=int, default=100000, help='number of user for eval.')
    
    # params for CPE
    parser.add_argument('--cov_loss_reg', type=float, default=5e-3)

    # params for TransCF
    parser.add_argument('--dis_reg', type=float, default=1e-2, help='Distance Regularization.')
    parser.add_argument('--nei_reg', type=float, default=1e-2, help='Neighborhood Regularization.')

    # param for DPCML (COCML, HarCML)
    parser.add_argument('--per_user_k', type=int, default=5, help='the number of embeddings of users')
    parser.add_argument('--DCRS_reg', type=float, default=0.0, help='whether regularization')
    parser.add_argument('--m1', type=float, default=0.05, help='the minimum distance between two vectors')
    parser.add_argument('--m2', type=float, default=0.25, help='the maximum distance between two vectors')

    # params for LRML
    parser.add_argument('--num_mems', type=int, default=20, help='number of memory')
    
    # params for CRML
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
    return parser.parse_args()

def test(model, logger, val_users, evaluators, top_rec_ks, epoch=0):
    
    if not isinstance(top_rec_ks, list):
        top_rec_ks = list(top_rec_ks)
    with torch.no_grad():
        model.eval()

        for k in top_rec_ks:
            p_k, r_k, n_k = evaluators.precision_recall_ndcg_k(model, val_users, k)
            logger.info("Epoch: {}, precision@{}: {}, recall@{}: {}, ndcg@{}: {}".format(epoch, 
                                                                                         k, 
                                                                                         p_k, 
                                                                                         k, 
                                                                                         r_k, 
                                                                                         k, 
                                                                                         n_k))

        _map, _mrr, _auc, _ndcg = evaluators.map_mrr_auc_ndcg(model, val_users)
        logger.info("Epoch: {}, MAP: {}, MRR: {}, AUC: {}, NDCG: {}".format(epoch, _map, _mrr, 
                                                                            _auc, _ndcg))

    return _auc

def train(args, model, logger, metric_evaluator, train_loader):
    
    val_users = np.asarray([i for i in range(model.num_users)])
    
    opt = OPT_DICTS[args.model](model.parameters(), lr = args.lr, weight_decay=args.weight_decay)

    _, val_evaluator, test_evaluator = metric_evaluator['train_evaluator'], \
                metric_evaluator['val_evaluator'], \
                metric_evaluator['test_evaluator']

    best_val_auc = 0
    for epoch in range(args.epoch):
        torch.cuda.empty_cache()
        gc.collect()

        logger.info('\n========> Epoch %3d: '% epoch) 
        model.train()
        if args.model == 'SFCML':
            scheduler_opt = lr_scheduler.ReduceLROnPlateau(opt, 
                                                   mode="max", 
                                                   factor=0.99,
                                                   patience=5,
                                                   threshold=0.0001, 
                                                   verbose=True
                                                   ) # Optional
            train_loader._start()

            for it in tqdm(range(len(train_loader)), desc="Opt. with FastCML loss"):
                user_ids = train_loader.next_batch()
                user_ids = user_ids.long().cuda()

                loss = model(user_ids)

                opt.zero_grad()
                loss.backward()
                
                opt.step()
                
                logger.info('======> Iter %4d/%4d:  loss: %.4f'%(it, len(train_loader), loss.mean().item()))
                
                model.ClipItemNorm()
                model.ClipUserNorm()
        else:

            train_loader.dataset.generate_triplets_by_sampling() # conduct negative sampling before training

            for it, (user_ids, pos_ids, neg_ids) in enumerate(train_loader):

                user_ids = user_ids.cuda()
                pos_ids = pos_ids.cuda()
                neg_ids = neg_ids.cuda()

                loss = model(user_ids, pos_ids, neg_ids)

                opt.zero_grad()
                loss.backward() 
                opt.step()

                if it % 1000 == 0 or it == len(train_loader) - 1:
                    logger.info('Iter %4d/%4d:  loss: %.4f'%(it, len(train_loader), loss.mean().item()))
        
            # Clip user/item embeddings for regularizations.
            # model.ClipNorm()

        logger.info('\n========> Evaluating validation set...')
        _auc = test(model, logger, val_users, val_evaluator, args.topks, epoch)

        if _auc > best_val_auc:
            best_val_auc = _auc
            logger.info('\n========> Evaluating test set...')

            test(model, logger, val_users, test_evaluator, args.topks, epoch)
            
            logger.save_model(model)

    logger.info('\nFinal results (val set) ====> best_val_auc:  %.6f' % (best_val_auc))
    return best_val_auc

if __name__ == '__main__':

    args = parse_args()
    set_seeds(args.random_seed)
    
    save_path = os.path.join(args.data_path, 
                             args.model,
                             'margin_{}'.format(args.margin),
                             'num_negs_{}'.format(args.num_negs))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
 
    log_path = '_'.join([
        'lr_{}'.format(args.lr),
        'margin_{}'.format(args.margin),
        'cov_loss_reg_{}'.format(args.cov_loss_reg),
        'dim_{}'.format(args.dim)
    ]) 

    cur_log = MyLog(os.path.join(save_path, log_path + '.log'), log_path + '.pth')

    cur_log.info(args)

    # load data
    if os.path.exists(os.path.join(args.data_path, "np_data")):

        print("load saved data....")
        user_train_matrix = sparse.load_npz(os.path.join(args.data_path, "np_data", 'user_train_matrix.npz'))
        user_val_matrix = sparse.load_npz(os.path.join(args.data_path, "np_data", 'user_val_matrix.npz'))
        user_test_matrix = sparse.load_npz(os.path.join(args.data_path, "np_data", 'user_test_matrix.npz'))

        user_train_matrix = dok_matrix(user_train_matrix)
        user_val_matrix = dok_matrix(user_val_matrix)
        user_test_matrix = dok_matrix(user_test_matrix)

        num_users, num_items = user_train_matrix.shape
        cur_log.info("number of users: {}".format(num_users))
        cur_log.info("number of items: {}".format(num_items))

    else:
        os.makedirs(os.path.join(args.data_path, "np_data"))
        # load data
        user_item_matrix, num_users, num_items = load_data(args, cur_log, data_name='users.dat', threholds=5)

        # split train/val/test and calculate prob 
        user_train_matrix, user_val_matrix, user_test_matrix = split_train_val_test(user_item_matrix, args, cur_log)

        # save data 
        print('saving splited data')
        sparse.save_npz(os.path.join(args.data_path, "np_data", 'user_train_matrix.npz'), coo_matrix(user_train_matrix))
        sparse.save_npz(os.path.join(args.data_path, "np_data", 'user_val_matrix.npz'), coo_matrix(user_val_matrix))
        sparse.save_npz(os.path.join(args.data_path, "np_data", 'user_test_matrix.npz'), coo_matrix(user_test_matrix))

    retain_train = False

    train_evaluator = Evaluator(num_users, num_items, user_train_matrix, user_train_matrix, on_train=True)
    val_evaluator = Evaluator(num_users, num_items, user_train_matrix, user_val_matrix, on_train=retain_train)
    test_evaluator = Evaluator(num_users, num_items, user_train_matrix, user_test_matrix, on_train=retain_train)

    if args.model == 'SFCML':
        from dataloader import _DataLoader
        train_loader =  _DataLoader([i for i in range(num_users)], batch_size = args.batch_size, shuffle = True)
        model = SUPPORT_MODEL[args.model](args, 
                    num_users,
                    num_items,
                    user_train_matrix,
                    args.margin, 
                    device=device).to(device)
    else:
        train_set = SampleDataset(user_train_matrix, 
                                args.num_negs, 
                                ALL_SAMS[args.model], 
                                args.random_seed)

        if args.model == 'CPE':
            model = SUPPORT_MODEL[args.model](
                        num_users,
                        num_items,
                        args.margin,
                        args.dim,
                        max_norm=args.max_norm,
                        cov_loss_reg = args.cov_loss_reg).to(device)
        elif args.model in ['COCML', 'HarCML']:
            model = SUPPORT_MODEL[args.model](
                 num_users, 
                 num_items,
                 margin=args.margin,
                 DCRS_reg=args.DCRS_reg,
                 m1=args.m1,
                 m2=args.m2,
                 dim=args.dim,
                 per_user_k=args.per_user_k,
                 max_norm=args.max_norm).to(device)
        elif args.model == 'TransCF':
            model = SUPPORT_MODEL[args.model](
                        num_users,
                        num_items,
                        args.margin,
                        args.dim,
                        dataset=train_set,
                        clip_max=args.max_norm,
                        dis_reg=args.dis_reg,
                        nei_reg=args.nei_reg).to(device)
        elif args.model == 'LRML':
            model = SUPPORT_MODEL[args.model](
                    num_users,
                    num_items,
                    num_mems=args.num_mems,
                    margin=args.margin,
                    dim=args.dim,
                    clip_max=args.max_norm).to(device)
        elif args.model == 'CRML':
            model = SUPPORT_MODEL[args.model](user_train_matrix,
                        num_users,
                        num_items,
                        args.margin,
                        args.dim,
                        alpha = args.alpha,
                        beta = args.beta).to(device)
        else:
            raise NotImplementedError
        
        train_loader =  DataLoader(train_set,
                                    batch_size = args.batch_size, 
                                    shuffle = True,
                                    pin_memory=True)
    metric_evaluator = {
            'train_evaluator': train_evaluator,
            'val_evaluator': val_evaluator,
            'test_evaluator': test_evaluator
        }
    train(args, model, cur_log, metric_evaluator, train_loader)
