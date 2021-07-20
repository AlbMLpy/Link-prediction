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
from sklearn.preprocessing import normalize

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
        self.filter = valid_filters
        
class Trainer():
    def __init__(self, best_hit_10, previous_best_loss, err_arr, it, l2 = 0, path_for_save = "/notebook/Relations_Learning/gpu/"):
        self.best_hit_10 = best_hit_10
        self.previous_best_loss = previous_best_loss
        self.err_arr = err_arr
        self.it = it
        self.l2 = l2
        self.path_for_save = path_for_save

def run_epoch(datas, epoch, device, model, optimizer, sheduler, batch_size, trainer, show_iter = True):
    coo_ns, vals_ns = generate_data(datas.coo_tensor, datas.vals, datas.mind_set, datas.shape, datas.how_many, epoch)
    coo_ns = torch.tensor(coo_ns, device = device)
    vals_ns = torch.tensor(vals_ns, device = device)
    shuffler = np.random.permutation(vals_ns.shape[0])
    coo_ns = coo_ns[shuffler]
    vals_ns = vals_ns[shuffler]
    #idxs = np.random.permutation(vals_ns.shape[0])
    print (vals_ns.shape[0], batch_size, vals_ns.shape[0]//batch_size)
        
    for i in range(vals_ns.shape[0]//batch_size):
        # Get loss and gradients per sample
        # print ("coo_ns[i], vals_ns[i]", coo_ns[i], vals_ns[i])
        end = min(vals_ns.shape[0] - 1, (i+1)*batch_size)
        
        a_elems = coo_ns[i*batch_size : end, 0]
        b_elems = coo_ns[i*batch_size : end, 1]
        c_elems = coo_ns[i*batch_size : end, 2]
        
        model.forward(coo_ns[i*batch_size : end], vals_ns[i*batch_size : end], a_elems, b_elems, c_elems)
        
        optimizer.step()
        optimizer.zero_grad()
       
        trainer.err_arr[trainer.it] = np.mean(model.err_list)
        if show_iter and i%2000 == 0:
            print("Iter: ", trainer.it, "; Error: ", np.mean(np.array(model.err_list)), flush = True)
        try:
            check_early_stop(trainer.err_arr[trainer.it], trainer.previous_best_loss, margin=trainer.err_arr[trainer.it]%20, max_attempts=10000)
            if (trainer.previous_best_loss > trainer.err_arr[trainer.it]):
                trainer.previous_best_loss = trainer.err_arr[trainer.it]
        except StopIteration: # early stopping condition met
            break
            print ("early_stoping loss", flush = True)
            raise StopIteration
        trainer.it += 1
    
    sheduler.step()
    a = model.a_torch.cpu().data.numpy()
    b = model.b_torch.cpu().data.numpy()
    c = model.a_torch.cpu().data.numpy()
    print ("count hr", flush = True)
    hit_rate = model.evaluate(datas)
    hit3, hit5, hit10 = hit_rate[0], hit_rate[1], hit_rate[2]
    if (hit10 > trainer.best_hit_10):
        trainer.best_hit_10 = hit10
    print (hit3, hit5, hit10, flush = True)
        # early stopping by hit@10
    try:
        check_early_stop_score(hit10, trainer.best_hit_10, margin=0.01, max_attempts=100)
    except StopIteration: # early stopping condition met
        print ("early_stoping score", flush = True)
        raise StopIteration
        
        
        # if hit@10 grows update checkpoint
    if (hit10 > trainer.best_hit_10):
        trainer.best_hit_10 = hit10
        np.save(trainer.path_for_save + 'gpu_a.npz', a_torch.cpu().data.numpy())
        np.save(trainer.path_for_save + 'gpu_b.npz', b_torch.cpu().data.numpy())
        np.save(trainer.path_for_save + 'gpu_c.npz', a_torch.cpu().data.numpy())
            
    return 0
