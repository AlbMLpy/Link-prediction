import torch
import numpy as np
import pandas as pd
from numba import jit
from sklearn.preprocessing import normalize
from torch.optim.lr_scheduler import StepLR

import pickle
import time
from ipypb import track
import argparse
import matplotlib.pyplot as plt


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

        
class DataStorage():
    def __init__(self, sparse_coords,
                 sparse_vals, mind_set,
                 shape, valid_triples, filters):
        
        self.coo_tensor = sparse_coords
        self.vals = sparse_vals
        self.mind_set = mind_set
        self.shape = shape
        self.valid_triples = valid_triples
        self.filter = filters
        
class Trainer():
    def __init__(self, best_hit_10,
                 previous_best_loss, err_arr, it, l2 = 0,
                 path_for_save = "/notebook/Relations_Learning/gpu/"):
        
        self.best_hit_10 = best_hit_10
        self.previous_best_loss = previous_best_loss
        self.err_arr = err_arr
        self.it = it
        self.l2 = l2
        self.path_for_save = path_for_save

        
def train_epoch(datas, epoch, device,
                model, optimizer, scheduler,
                batch_size, trainer, show_iter=True):
    
    user_idx = np.arange(len(un_pair))
    np.random.shuffle(user_idx)
    batches = np.array_split(user_idx, train_data.shape[0] // bs)
    cols = torch.arange(data_shape[2])
    
    for i, batch in enumerate(batches):
        rows = torch.tensor(batch)

        # Compute prediction error
        rating = torch.zeros((len(un_pair[rows]), data_shape[2])).to(device)
        for j in range(rating.shape[0]):
            rating[j][sample_positive[tuple(un_pair[rows][j].tolist())]] = 1.0

        prediction = model(
            torch.repeat_interleave(torch.tensor(un_pair[rows][:, 0]), cols.shape[0]),
            torch.repeat_interleave(torch.tensor(un_pair[rows][:, 1]), cols.shape[0]),
            torch.tile(cols, (un_pair[rows].shape[0], )),
        ).view(un_pair[rows].shape[0], cols.shape[0]).to(torch.float64)

        loss = loss_fn(prediction, rating)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        trainer.err_arr[trainer.it] = np.mean(model.err_list)
        if show_iter and (i % 2000 == 0):
            print("Iter: ", trainer.it, "; Error: ", np.mean(np.array(model.err_list)), flush=True)
        
        try:
            check_early_stop(
                trainer.err_arr[trainer.it],
                trainer.previous_best_loss,
                margin=trainer.err_arr[trainer.it] % 20,
                max_attempts=10000,
            )
            if (trainer.previous_best_loss > trainer.err_arr[trainer.it]):
                trainer.previous_best_loss = trainer.err_arr[trainer.it]
        
        except StopIteration: # early stopping condition met
            break
            print ("early_stoping loss", flush=True)
            raise StopIteration
        
        trainer.it += 1
           
        
        if show and (i % 10 == 0):
            loss, current = loss.item(), i * len(rows)
            print(f"loss: {loss:>7f}  [{current:>5d}/{coords.shape[0]:>5d}]")
    
    scheduler.step()
    a = model.entity_factors.cpu().detach().numpy() #a = model.a_torch.cpu().data.numpy()
    b = model.relations_factors.cpu().detach().numpy() #b = model.b_torch.cpu().data.numpy()
    c = model.entity_factors.cpu().detach().numpy() #c = model.a_torch.cpu().data.numpy()

    print("Count HR:", flush=True)
    hit_rate = model.evaluate(datas)
    hit3, hit5, hit10, mrr10 = hit_rate[0], hit_rate[1], hit_rate[2], hit_rate[3]
    
    if (hit10 > trainer.best_hit_10):
        trainer.best_hit_10 = hit10
    
    print(hit3, hit5, hit10, mrr10, flush=True)
    
    # early stopping by hit@10
    try:
        check_early_stop_score(
            hit10,
            trainer.best_hit_10,
            margin=0.01,
            max_attempts=100,
        )
    
    except StopIteration: # early stopping condition met
        print ("early_stoping score", flush=True)
        raise StopIteration
        
        
    # if hit@10 grows update checkpoint
    if (hit10 > trainer.best_hit_10):
        trainer.best_hit_10 = hit10
        np.save(trainer.path_for_save + 'gpu_a.npz', a_torch.cpu().data.numpy())
        np.save(trainer.path_for_save + 'gpu_b.npz', b_torch.cpu().data.numpy())
        np.save(trainer.path_for_save + 'gpu_c.npz', a_torch.cpu().data.numpy())
            
    return 0