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

from samplings import give_ns, generate_data

from elementwise_grads import bernoulli_logit_loss, bernoulli_logit_loss_grad

from general_functions1 import sqrt_err_relative, check_coo_tensor, gen_coo_tensor
from general_functions1 import create_filter, hr
from cp_with_grads import gcp_grad

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
        
class data_storage():
    def __init__(self, sparse_coords, sparse_vals, mind_set, shape, how_many, valid_triples, valid_filters):
        self.coo_tensor = sparse_coords
        self.vals = sparse_vals
        self.mind_set = mind_set
        self.shape = shape
        self.how_many = how_many
        self.valid_triples = valid_triples
        self.valid_filters = valid_filters
        
class Trainer():
    def __init__(self, best_hit_10, previous_best_loss, err_arr, it, l2, loss_function, loss_function_grad, path_for_save = "/notebook/Relations_Learning/gpu/"):
        self.best_hit_10 = best_hit_10
        self.previous_best_loss = previous_best_loss
        self.err_arr = err_arr
        self.it = it
        self.l2 = l2
        self.loss_function_grad = loss_function_grad
        self.loss_function = loss_function
        self.path_for_save = path_for_save

def evaluate_epoch(datas, epoch, device, a_torch, b_torch, optimizer, sheduler, batch_size, trainer, show_iter = True):
    coo_ns, vals_ns = generate_data(datas.coo_tensor, datas.vals, datas.mind_set, datas.shape, datas.how_many, epoch)
    coo_ns = torch.tensor(coo_ns, device = device)
    vals_ns = torch.tensor(vals_ns, device = device)
    shuffler = np.random.permutation(vals_ns.shape[0])
    coo_ns = coo_ns[shuffler]
    vals_ns = vals_ns[shuffler]
    #idxs = np.random.permutation(vals_ns.shape[0])
    print (vals_ns.shape[0], batch_size, vals_ns.shape[0]//batch_size)
    err_list = []

    a = a_torch.cpu().data.numpy()
    b = b_torch.cpu().data.numpy()
    c = a_torch.cpu().data.numpy()
    print ("count hr", flush = True)
    hit3, hit5, hit10, mrr = hr(datas.valid_filters[:10000], datas.valid_triples[:10000], a, b, c, [1, 3, 10])
    if (hit10 > trainer.best_hit_10):
        trainer.best_hit_10 = hit10
    print (hit3, hit5, hit10, mrr, flush = True)

        
    for i in range(vals_ns.shape[0]//batch_size):
        # Get loss and gradients per sample
        # print ("coo_ns[i], vals_ns[i]", coo_ns[i], vals_ns[i])
        end = min(vals_ns.shape[0] - 1, (i+1)*batch_size)
        loss, g_a, g_b, g_c = gcp_grad(
            coo_ns[i*batch_size : end], vals_ns[i*batch_size : end], datas.shape, a_torch, b_torch,
            trainer.l2, trainer.loss_function, trainer.loss_function_grad, device
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
        trainer.err_arr[trainer.it] = np.mean(err_list)
        if show_iter and i%500 == 0:
            print("Iter: ", trainer.it, "; Error: ", np.mean(np.array(err_list)), flush = True)
        try:
            check_early_stop(trainer.err_arr[trainer.it], trainer.previous_best_loss, margin=trainer.err_arr[trainer.it]%20, max_attempts=10)
            if (trainer.previous_best_loss > trainer.err_arr[trainer.it]):
                trainer.previous_best_loss = trainer.err_arr[trainer.it]
        except StopIteration: # early stopping condition met
            break
            print ("early_stoping loss", flush = True)
            raise StopIteration
        trainer.it += 1
    
    sheduler.step()
    a = a_torch.cpu().data.numpy()
    b = b_torch.cpu().data.numpy()
    c = a_torch.cpu().data.numpy()
    print ("count hr")
    hit3, hit5, hit10, mrr = hr(datas.valid_filters[:10000], datas.valid_triples[:10000], a, b, c, [1, 3, 10])
    print (hit3, hit5, hit10, mrr, flush = True)
        
        # early stopping by hit@10
    try:
        check_early_stop_score(hit10, trainer.best_hit_10, margin=0.01, max_attempts=10)
    except StopIteration: # early stopping condition met
        print ("early_stoping score", flush = True)
        raise StopIteration
        
        
        # if hit@10 grows update checkpoint
    if (hit10 > trainer.best_hit_10):
        trainer.best_hit_10 = hit10
        np.save(trainer.path_for_save + 'gpu_a.npz', a_torch.cpu().data.numpy())
        np.save(trainer.path_for_save + '/notebook/Relations_Learning/gpu/gpu_b.npz', b_torch.cpu().data.numpy())
        np.save(trainer.path_for_save + '/notebook/Relations_Learning/gpu/gpu_c.npz', a_torch.cpu().data.numpy())
            
    return a_torch, b_torch
