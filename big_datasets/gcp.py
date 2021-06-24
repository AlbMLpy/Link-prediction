import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt

import pickle
import time
from ipypb import track
import argparse

from general_functions1 import sqrt_err_relative, check_coo_tensor, gen_coo_tensor
from general_functions1 import create_filter, hr

import CP_ALS3.CP_ALS3 as cp

#with open('test_filter.pkl', 'rb') as f:
    #test_filter = pickle.load(f)
    
#with open('valid_filter.pkl', 'rb') as f:
    #valid_filter = pickle.load(f)
    
def mttcrp(coo, val, shape, mode, a, b):
    """
        Calculate matricized-tensor times Khatri-Rao product. 
    """
    temp = np.zeros((a.shape[1],))
    
    if mode == 0:
        mode_a = 1 
        mode_b = 2
        
    elif mode == 1:
        mode_a = 0
        mode_b = 2
        
    else:
        mode_a = 0
        mode_b = 1
        
    temp += a[coo[mode_a], :] * b[coo[mode_b], :] * val 
    
    return temp

def get_elem_deriv_tensor(vals, kruskal_vals, loss_function_grad):
    """
        Calculate the elementwise derivative tensor Y.
    """
    #deriv_tensor_vals = loss_function_grad(vals, kruskal_vals) / vals.size
    deriv_tensor_vals = loss_function_grad(vals, kruskal_vals)
    return deriv_tensor_vals    

def factors_to_tensor(coo_tensor, vals, a, b, c):
    """
        Calculate Kruskal tensor values with
        the same coordinates as initial tensor has.
    """
    
    krus_vals = np.zeros_like(vals)
    for item in range(coo_tensor.shape[0]):
        coord = coo_tensor[item]
        krus_vals[item] = np.sum(
            a[coord[0], :] * b[coord[1], :] * c[coord[2], :]
        )
    return krus_vals    

def gcp_grad(coo, val, shape, a, b, c, l2, loss_function, loss_function_grad):
    """
        GCP loss function and gradient calculation.
        All the tensors have the same coordinate set: coo_tensor.
    """
    
    # Construct sparse kruskal tensor
    kruskal_val = np.sum(
        a[coo[0], :] * b[coo[1], :] * c[coo[2], :]
    )#factors_to_tensor(coo_tensor, vals, a, b, c)
    
    #if (val >=0):
        #loss_warp, grad_warp = count_warp(triplets)
    
    # Calculate mean loss on known entries
    loss = loss_function(val, kruskal_val)
    
    # Compute the elementwise derivative tensor
    deriv_tensor_val = loss_function_grad(val, kruskal_val)
    
    # Calculate gradients w.r.t. a, b, c factor matrices
    g_a = mttcrp(coo, deriv_tensor_val, shape, 0, b, c)
    g_b = mttcrp(coo, deriv_tensor_val, shape, 1, a, c)
    g_c = mttcrp(coo, deriv_tensor_val, shape, 2, a, b)
    
    # Add L2 regularization
    if l2 != 0:
        g_a += l2 * a[coo[0], :]
        g_b += l2 * b[coo[1], :]
        g_c += l2 * c[coo[2], :]
    
    return loss, g_a, g_b, g_c

def multi_ind_to_indices(multi_indices, shape):
    coords = np.zeros(shape=(multi_indices.shape[0], 3), dtype=np.int64)
    coords[:, 2] = multi_indices % shape[2]
    i1 = multi_indices // shape[2]
    coords[:, 0] = i1 // shape[1]
    coords[:, 1] = i1 % shape[1]
    return coords

def indices_to_multi_ind(coords, shape):
    multi_indices = (coords[:, 2] + (shape[2] * coords[:, 1])
        + (shape[1] * shape[2] * coords[:, 0]))
    return multi_indices

def give_ns(multi_inx_set, tensor_shape, how_many=1, seed=13, show_iter=False):
    random_state = np.random.seed(seed)
    ns_size = how_many * len(multi_inx_set)
    mixs = multi_inx_set.copy()
    ns = np.zeros(ns_size, dtype=np.int64)
    all_ind = tensor_shape[0] * tensor_shape[1] * tensor_shape[2]
    for i in range(ns_size):
        check = True
        while check:
            cand = np.random.choice(all_ind)
            if cand not in mixs: 
                mixs.add(cand)
                ns[i] = cand
                check = False
        if show_iter:        
            if i % 10000 == 0:
                print("Iter: ", i)
    return  multi_ind_to_indices(ns, tensor_shape)   

#@jit(nopython=True)
def gaussian_loss(x_vals, m_vals):
    return (x_vals - m_vals)**2

#@jit(nopython=True)
def gaussian_loss_grad(x_vals, m_vals):
    return -2 * (x_vals - m_vals)

#@jit(nopython=True)
def bernoulli_loss(x_vals, m_vals):
    eps = 1e-8
    return np.log(1 + m_vals) - x_vals*np.log(eps + m_vals)

#@jit(nopython=True)
def bernoulli_loss_grad(x_vals, m_vals):
    eps = 1e-8
    return (1 / (1 + m_vals)) - (x_vals / (eps + m_vals))

#@jit(nopython=True)
def bernoulli_logit_loss(x_vals, m_vals):
    return np.log(1 + np.exp(m_vals)) - (x_vals * m_vals)

#@jit(nopython=True)
def bernoulli_logit_loss_grad(x_vals, m_vals):
    exp_vals = np.exp(m_vals)
    return (exp_vals / (1 + exp_vals)) - x_vals

#@jit(nopython=True)
def poisson_loss(x_vals, m_vals):
    eps = 1e-8
    return m_vals - x_vals * np.log(m_vals + eps)

#@jit(nopython=True)
def poisson_loss_grad(x_vals, m_vals):
    eps = 1e-8
    return 1 - (x_vals / (m_vals + eps))

#@jit(nopython=True)
def poisson_log_loss(x_vals, m_vals):
    return np.exp(m_vals) - (x_vals * m_vals)

#@jit(nopython=True)
def poisson_log_loss_grad(x_vals, m_vals):
    return np.exp(m_vals) - x_vals

def generate_data(coo_tensor, vals, multi_inx_set, shape, how_many, seed):
    ns = give_ns(multi_inx_set, shape, how_many, seed, show_iter=False)
    all_coords = np.concatenate((coo_tensor, ns), axis=0)
    all_vals = np.zeros((how_many * len(multi_inx_set) + vals.size, ))
    all_vals[:vals.size] = vals
    return all_coords, all_vals

from decimal import Decimal
from timeit import default_timer as timer

def gcp_sgd_ns(coo_tensor,
               vals,
               shape,
               loss_function,
               loss_function_grad,
               rank=5,
               lr=0.1,
               num_epoch=20,
               how_many=1,
               seed=13,
               show_iter=False,
               it_over=True):
    """
        Factorize initial sparse tensor by generalized CP decomposition.
    """
    
    random_state = np.random.seed(seed)
    # a = np.random.normal(0.0, 0.1, size=(shape[0], rank))
    # b = np.random.normal(0.0, 0.1, size=(shape[1], rank))
    # c = np.random.normal(0.0, 0.1, size=(shape[2], rank))
    a =  np.load(open("/notebook/data/results/a100.npz.npy", 'rb'))
    b =  np.load(open("/notebook/data/results/b100.npz.npy", 'rb'))
    c =  np.load(open("/notebook/data/results/c100.npz.npy", 'rb'))
    init_mind_set = set(indices_to_multi_ind(coo_tensor, shape))
    coo_ns = np.empty((how_many * len(init_mind_set) + vals.size, 3), dtype=np.int64)
    vals_ns = np.empty((how_many * len(init_mind_set) + vals.size,), dtype=np.float64)
    
    err_arr = np.empty((num_epoch, ), dtype=np.float64)
    error = 0.0
    it = 0
    err_start = 0.0
    start = timer()
    for epoch in range(num_epoch):
        if (epoch < 20):
            lr = 6e-2
        else:
            lr = 1e-2
        print ("lr", lr)
        coo_ns, vals_ns = generate_data(coo_tensor, vals, init_mind_set, shape, how_many, epoch)
        shuffler = np.random.permutation(vals_ns.shape[0])
        coo_ns = coo_ns[shuffler]
        vals_ns = vals_ns[shuffler]
        err = 0.0
        err_list = []
        for i in range(vals_ns.shape[0]):
            # Get loss and gradients per sample
            error, g_a, g_b, g_c = gcp_grad(
                coo_ns[i], vals_ns[i], shape, a, b, c,
                lr, loss_function, loss_function_grad,
            )
            err_list.append(error)
            # Update factor matrices
            a[coo_ns[i][0], :] = a[coo_ns[i][0], :] - (lr * g_a)
            b[coo_ns[i][1], :] = b[coo_ns[i][1], :] - (lr * g_b)
            c[coo_ns[i][2], :] = c[coo_ns[i][2], :] - (lr * g_c)
        err = np.mean(np.array(err_list))
        if (err > err_start):
            err_start = err
            print ("new_err_start", err_start)
        if (err_start > err*100 ):
            lr = lr /10
            err_start = err
            print ("new_lr", lr)
        err_arr[it] = err
        if show_iter:
            print("Iter: ", it, "; Error: ", error, flush = True)
        it += 1 
        np.save('/notebook/data/results/a'+str(rank)+'.npz', a)
        np.save('/notebook/data/results/b'+str(rank)+'.npz', b)
        np.save('/notebook/data/results/c'+str(rank)+'.npz', c)
    end = timer()
    print ("time", end - start)
    
    return a, b, c, err_arr, it

def main():
    print ("loaded 0", flush = True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=200, nargs="?",
                    help="set desored emebdding dimention")

    args = parser.parse_args()
    dim = args.dim

    path_data = "Link_Prediction_Data/RuBIQ8M/"
    entity_dict = pickle.load(open(path_data + 'ent_to_ind', 'rb'))
    relation_dict = pickle.load(open(path_data + 'rel_to_ind', 'rb'))
    print ("loaded 0")
    train_triples = pickle.load(open(path_data + 'train', 'rb'))
    valid_triples = pickle.load(open(path_data + 'valid', 'rb'))
    test_triples = pickle.load(open(path_data + 'test', 'rb'))
    #train_valid_triples = pickle.load(open(path_data + 'train_valid_triples', 'rb'))

    entity_map = pickle.load(open(path_data + 'ents_map', 'rb'))
    relation_map = pickle.load(open(path_data + 'relation_map', 'rb'))

    all_triples = train_triples + valid_triples + test_triples
    print ("loaded 1", flush = True)
    num_epoch = 100
    rank = dim 
    lr = 1e-2
    seed = 13 
    hm = 1000
    how_many = 2
    
    values = [1] * len(train_triples)
    values = np.array(values, dtype=np.int64)

    coords = np.array(train_triples, dtype=np.int64)
    nnz = len(train_triples)
    data_shape = (len(entity_dict), len(relation_dict), len(entity_dict))
    
    print (data_shape, flush = True)
    
    a, b, c, err_arr, it = gcp_sgd_ns(
    coords, values, data_shape,
    bernoulli_logit_loss, 
    bernoulli_logit_loss_grad, 
    rank=rank,
    lr=lr,
    num_epoch=num_epoch,
    how_many=how_many,
    seed=seed,
    show_iter=True,
    it_over=True)
    
if __name__ == "__main__":
    main()

