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

import CP_ALS3.CP_ALS3 as cp

#with open('test_filter.pkl', 'rb') as f:
    #test_filter = pickle.load(f)
    
with open('/notebook/data/Link_Prediction_Data/RuBIQ8M/valid_filter.pkl', 'rb') as f:
    valid_filter = pickle.load(f)
    
import numpy as np

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
    data_shape = (len(entity_dict), len(relation_dict), len(entity_dict))
    
    print (data_shape, flush = True)
    
    
    coo_tensor = coords
    vals = values
    shape = data_shape
    loss_function = bernoulli_logit_loss
    loss_function_grad = bernoulli_logit_loss_grad

    from torch.nn.init import xavier_normal_
    from torch import optim

    device=torch.device("cpu")


    random_state = np.random.seed(seed)

    batch_size = 512
    init_mind_set = set(indices_to_multi_ind(coo_tensor, shape))
    coo_ns = np.empty((how_many * len(init_mind_set) + vals.size, 3), dtype=np.int64)
    vals_ns = np.empty((how_many * len(init_mind_set) + vals.size,), dtype=np.float64)

    err_arr = np.empty((num_epoch*vals_ns.shape[0]//batch_size + 1, ), dtype=np.float64)
    error = 0.0
    it = 0


    a_torch = torch.empty((shape[0], rank), requires_grad = True, device = device)
    xavier_normal_(a_torch)
    a_torch.grad = torch.zeros(a_torch.shape, device = device)

    b_torch = torch.empty((shape[1], rank), requires_grad = True, device = device)
    xavier_normal_(b_torch)
    b_torch.grad = torch.zeros(b_torch.shape, device = device)

    optimizer = optim.Adam([a_torch, b_torch], lr=6e-3)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
    print ("loaded 2", flush = True)
    show_iter = True

    start = timer()
    for epoch in range(num_epoch):
        coo_ns, vals_ns = generate_data(coo_tensor, vals, init_mind_set, shape, how_many, epoch)
        coo_ns = torch.tensor(coo_ns, device = device)
        vals_ns = torch.tensor(vals_ns, device = device)
        shuffler = np.random.permutation(vals_ns.shape[0])
        coo_ns = coo_ns[shuffler]
        vals_ns = vals_ns[shuffler]
        print ("loaded 3", flush = True)
        #idxs = np.random.permutation(vals_ns.shape[0])
        print (vals_ns.shape[0], batch_size, vals_ns.shape[0]//batch_size)
        err_list = []
        for i in range(vals_ns.shape[0]//batch_size):
            # Get loss and gradients per sample
            # print ("coo_ns[i], vals_ns[i]", coo_ns[i], vals_ns[i])
            end = min(vals_ns.shape[0] - 1, (i+1)*batch_size)
            loss, g_a, g_b, g_c = gcp_grad(
                coo_ns[i*batch_size : end], vals_ns[i*batch_size : end], shape, a_torch, b_torch,
                l2, loss_function, loss_function_grad, device
            )
            err_list.append(loss.cpu().detach().numpy().mean())

            a_elems = coo_ns[i*batch_size : end, 0]
            b_elems = coo_ns[i*batch_size : end, 1]
            c_elems = coo_ns[i*batch_size : end, 2]

            a_torch.grad[a_elems, :] = g_a
            b_torch.grad[b_elems, :] = g_b
            a_torch.grad[c_elems, :] = g_c
            optimizer.step()
            a_torch.grad = torch.zeros(a_torch.shape, device = device)
            b_torch.grad = torch.zeros(b_torch.shape, device = device)

            err_arr[it] = np.mean(err_list)
            if show_iter and i%100 == 0:
                print("Iter: ", it, "; Error: ", np.mean(np.array(err_list)))
            it += 1  
        scheduler.step()
        a = a_torch.cpu().data.numpy()
        b = b_torch.cpu().data.numpy()
        c = a_torch.cpu().data.numpy()
        print ("count hr", flush = True)
        print (hr(valid_filter[:10000], valid_triples[:10000], a, b, c, [1, 3, 10]), flush = True)
        np.save('/notebook/data/results/gpu_a.npz', a_torch.cpu().data().numpy())
        np.save('/notebook/data/results/gpu_b.npz', b_torch.cpu().data().numpy())
        np.save('/notebook/data/results/gpu_c.npz', a_torch.cpu().data().numpy())
    end = timer()
    print ("time", end - start, flush = True)
    
    
if __name__ == "__main__":
    main()

