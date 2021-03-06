import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

import pickle
import time
from ipypb import track
import argparse

import torch

from t_alg import mttcrp, mttcrp1, get_elem_deriv_tensor, factors_to_tensor, gcp_grad, multi_ind_to_indices, indices_to_multi_ind

from samplings import give_ns, generate_data

from elementwise_grads import bernoulli_logit_loss, bernoulli_logit_loss_grad

from general_functions1 import sqrt_err_relative, check_coo_tensor, gen_coo_tensor
from general_functions1 import create_filter, hr

from decimal import Decimal
from timeit import default_timer as timer

from experiments import data_storage, Trainer, evaluate_epoch

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
def check_early_stop(target_score, previous_best, margin=0, max_attempts=10):
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
def check_early_stop_score(target_score, previous_best, margin=0, max_attempts=3):
    if (previous_best > target_score):
        previous_best = target_score
    if (margin >= 0) and (target_score < previous_best + margin):
        check_early_stop_score.fail_count_score += 1
    else:
        check_early_stop_score.fail_count_score = 0
    if check_early_stop_score.fail_count_score >= max_attempts:
        print('Interrupted due to early stopping scoring condition.', check_early_stop_score.fail_count_score, flush = True)
        raise StopIteration

def gcp_grad(coo, val, shape, a, b, l2, loss_function, loss_function_grad, device):
    """
        GCP loss function and gradient calculation.
        All the tensors have the same coordinate set: coo_tensor.
    """

    # Construct sparse kruskal tensor
    kruskal_val = torch.sum((a[coo[:,0], :] * b[coo[:,1], :] * a[coo[:,2], :]),1)
    #factors_to_tensor(coo_tensor, vals, a, b, c)
    
    # Calculate mean loss on known entries
    loss = loss_function(val, kruskal_val)
    # Compute the elementwise derivative tensor
    deriv_tensor_val = loss_function_grad(val, kruskal_val)
    
    #print ("in qcp_grad in deriv_tensor_val ", deriv_tensor_val)
    # Calculate gradients w.r.t. a, b, c factor matrices
    g_a = mttcrp1(coo, deriv_tensor_val, shape, 0, b, a, device)
    g_b = mttcrp1(coo, deriv_tensor_val, shape, 1, a, a, device)
    g_c = mttcrp1(coo, deriv_tensor_val, shape, 2, a, b, device)
    
    #print ("\n\n")
    
    
    # Add L2 regularization
    if l2 != 0:
        g_a += l2 * a[coo[0], :]
        g_b += l2 * b[coo[1], :]
        g_c += l2 * c[coo[2], :]
    
    return loss, g_a, g_b, g_c



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


    print ("loaded1", flush = True)
    num_epoch = 20
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
    
    
    
    loss_bse = torch.nn.BCELoss()

    coo_tensor = coords
    vals = values
    shape = data_shape
    loss_function = bernoulli_logit_loss
    loss_function_grad = bernoulli_logit_loss_grad

    from torch.nn.init import xavier_normal_
    from torch import optim

    device=torch.device("cuda:4")

    num_epoch = 600

    random_state = np.random.seed(seed)

    # specify property of data
    batch_size = 56
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
    trainer = Trainer(best_hit_10, previous_best_loss, err_arr, it, l2, loss_function = bernoulli_logit_loss, loss_function_grad = bernoulli_logit_loss_grad)
    
    start = timer()

    num_ent = 14541
    dim_emb = 200
    num_rel = 237

    a_torch = torch.empty((num_ent, dim_emb), requires_grad = True, device = device)
    xavier_normal_(a_torch)
    a_torch.grad = torch.zeros(a_torch.shape, device = device)

    b_torch = torch.empty((num_rel, dim_emb), requires_grad = True, device = device)
    xavier_normal_(b_torch)
    b_torch.grad = torch.zeros(b_torch.shape, device = device)

    optimizer = optim.SGD([a_torch, b_torch], lr=1e-2, momentum=0.1, nesterov = True)
    #optimizer = optim.Adam([a_torch, b_torch], lr=1e-3)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

    show_iter = True
    start = timer()
    for epoch in range(num_epoch):
        try:
            a_torch, b_torch = evaluate_epoch(data_s, epoch, device, a_torch, b_torch, optimizer, scheduler, batch_size, trainer, show_iter = True)
        except StopIteration: # early stopping condition met
            break
            print ("early_stoping loss", flush = True)
            raise StopIteration
            

        a = a_torch.cpu().data.numpy()
        b = b_torch.cpu().data.numpy()
        c = a_torch.cpu().data.numpy()
        print ("count hr")
        hit3, hit5, hit10, mrr = hr(valid_filter[:10000], valid_triples[:10000], a, b, c, [1, 3, 10])
        print (hit3, hit5, hit10, mrr, flush = True)
        
        # early stopping by hit@10
        try:
            check_early_stop_score(hit10, best_hit_10, margin=0.01, max_attempts=10)
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
        #np.save('/notebook/Relations_Learning/gpu/gpu_a.npz', a_torch.cpu().data.numpy())
        #np.save('/notebook/Relations_Learning/gpu/gpu_b.npz', b_torch.cpu().data.numpy())
        #np.save('/notebook/Relations_Learning/gpu/gpu_c.npz', a_torch.cpu().data.numpy())

    
    
if __name__ == "__main__":
    main()

