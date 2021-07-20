import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
from torch.nn.init import xavier_normal_
from torch import optim

from torch.optim.lr_scheduler import StepLR

import pickle
import time
from ipypb import track
import argparse

import torch
from sklearn.preprocessing import normalize

from t_alg import mttcrp, mttcrp1, get_elem_deriv_tensor, factors_to_tensor, gcp_grad, multi_ind_to_indices, indices_to_multi_ind

from samplings import give_ns, generate_data

from elementwise_grads import bernoulli_logit_loss, bernoulli_logit_loss_grad, bernoulli_loss, bernoulli_loss_grad

from general_functions1 import sqrt_err_relative, check_coo_tensor, gen_coo_tensor
from general_functions1 import create_filter, hr

from decimal import Decimal
from timeit import default_timer as timer

from experiments import data_storage, Trainer, run_epoch

from model import FoxIE

#import CP_ALS3.CP_ALS3 as cp

#with open('test_filter.pkl', 'rb') as f:
    #test_filter = pickle.load(f)
    
with open('/notebook/Relations_Learning/test_filter.pkl', 'rb') as f:
    test_filter = pickle.load(f)
    
with open('/notebook/Relations_Learning/valid_filter.pkl', 'rb') as f:
    valid_filter = pickle.load(f)
    
import numpy as np

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(fail_count=0)
def check_early_stop(target_score, previous_best, margin=0, max_attempts=1000):
    if (previous_best > target_score):
        previous_best = target_score
    if (margin >= 0) and (target_score > previous_best + margin):
        check_early_stop.fail_count += 1
    else:
        check_early_stop.fail_count = 0
    if check_early_stop.fail_count >= max_attempts:
        print('Interrupted due to early stopping condition.', check_early_stop.fail_count, flush = True)
        raise StopIteration

@static_vars(fail_count_score=0)        
def check_early_stop_score(target_score, previous_best, margin=0, max_attempts=3000):
    if (previous_best > target_score):
        previous_best = target_score
    if (margin >= 0) and (target_score < previous_best + margin):
        check_early_stop_score.fail_count_score += 1
    else:
        check_early_stop_score.fail_count_score = 0
    if check_early_stop_score.fail_count_score >= max_attempts:
        print('Interrupted due to early stopping scoring condition.', check_early_stop_score.fail_count_score, flush = True)
        raise StopIteration


def main():
    print ("loaded 0", flush = True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=200, nargs="?",
                    help="set desored emebdding dimention")

    args = parser.parse_args()
    dim = args.dim

    path_data = "/notebook/Relations_Learning/Link_Prediction_Data/FB15K237/"
    entity_list = pickle.load(open(path_data + 'entity_list', 'rb'))
    relation_list = pickle.load(open(path_data + 'relation_list', 'rb'))

    train_triples = pickle.load(open(path_data + 'train_triples', 'rb'))
    valid_triples = pickle.load(open(path_data + 'valid_triples', 'rb'))
    test_triples = pickle.load(open(path_data + 'test_triples', 'rb'))
    train_valid_triples = pickle.load(open(path_data + 'train_valid_triples', 'rb'))

    entity_map = pickle.load(open(path_data + 'entity_map', 'rb'))
    relation_map = pickle.load(open(path_data + 'relation_map', 'rb'))

    all_triples = train_valid_triples + test_triples


    print ("loaded1_", flush = True)
    num_epoch = 50
    rank = dim 
    lr = 1e-2
    seed = 13 
    hm = 1000
    how_many = 2
    l2 = 0
    
    values = [1] * len(train_triples)
    values = np.array(values, dtype=np.int64)

    coords = np.array(train_triples, dtype=np.int64)
    nnz = len(train_triples)
    data_shape = (len(entity_list), len(relation_list), len(entity_list))
    
    print (data_shape, flush = True)
    
    coo_tensor = coords
    vals = values
    shape = data_shape

    device=torch.device("cuda:4")

    num_epoch = 200

    random_state = np.random.seed(seed)

    # specify property of data
    batch_size = 30
    init_mind_set = set(indices_to_multi_ind(coo_tensor, shape))
    coo_ns = np.empty((how_many * len(init_mind_set) + vals.size, 3), dtype=np.int64)
    vals_ns = np.empty((how_many * len(init_mind_set) + vals.size,), dtype=np.float64)
    
    data_s = data_storage(sparse_coords = coords, sparse_vals =values, mind_set = init_mind_set, shape=data_shape, how_many=2, valid_filters = valid_filter, valid_triples = valid_triples)

    # specify property of training
    err_arr = np.empty((num_epoch*vals_ns.shape[0]//batch_size + 1, ), dtype=np.float64)
    error = 0.0
    it = 0
    previous_best_loss = 100000.0
    best_hit_10 = 0.0
    # specify training class
    trainer = Trainer(best_hit_10, previous_best_loss, err_arr, it)
    
    start = timer()

    #num_ent = 14541
    #dim_emb = 200
    #num_rel = 237

    model = FoxIE(rank=rank, shape=data_shape, given_loss=bernoulli_logit_loss, given_loss_grad=bernoulli_logit_loss_grad, device=device)
    model.init()
    optimizer = optim.SGD([model.a_torch, model.b_torch], lr=1e-4)
    #optimizer = optim.Adam([model.a_torch, model.b_torch], lr=5e-4)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

    show_iter = True
    start = timer()
    
    for epoch in range(num_epoch):
        try:
            run_epoch(data_s, epoch, device, model, optimizer, scheduler, batch_size, trainer, show_iter = True)
        except StopIteration: # early stopping condition met
            break
            print ("early_stoping loss", flush = True)
            raise StopIteration


        hit3, hit5, hit10, mrr = model.evaluate(data_s)
        print (hit3, hit5, hit10, mrr, flush = True)
        
    # early stopping by hit@10
        try:
            check_early_stop_score(hit10, best_hit_10, margin=0.01, max_attempts=1000)
        except StopIteration: # early stopping condition met
            break
            print ("early_stoping score", flush = True)
        

        
        # if hit@10 grows update checkpoint
        if (hit10 > best_hit_10):
            best_hit_10 = hit10
            np.save('/notebook/Relations_Learning/gpu/gpu_a.npz', a_torch.cpu().data.numpy())
            np.save('/notebook/Relations_Learning/gpu/gpu_b.npz', b_torch.cpu().data.numpy())
            np.save('/notebook/Relations_Learning/gpu/gpu_c.npz', a_torch.cpu().data.numpy())
    
        end = timer()
        print (end - start)
        
if __name__ == "__main__":
    main()

