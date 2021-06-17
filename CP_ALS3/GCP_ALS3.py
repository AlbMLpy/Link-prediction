import numpy as np
from numba import jit
import math

from scipy import sparse

import numpy as np
import math

from scipy import sparse
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/notebook/Relations_Learning/")

from general_functions import sqrt_err_relative, check_coo_tensor, gen_coo_tensor
from general_functions import create_filter, hr

def gen_coo_tensor(shape, density=0.02):
    nnz = int(density * shape[0] * shape[1] * shape[2])
    m = np.random.choice(shape[0], nnz)
    n = np.random.choice(shape[1], nnz)
    k = np.random.choice(shape[2], nnz)
    vals = np.random.rand(nnz)
    return np.vstack((m, n, k)).T, vals

def check_coo_tensor(coo):
    count = 0
    for i in range(coo.shape[0]):
        for j in range(coo.shape[0]):
            if (coo[i]==coo[j]).sum() == 3:
                count += 1
                if count > 1:
                    return "Bad"
        count = 0  
def gen_hilbert_tensor(shape):
    coo = []
    vals = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                coo.append((i, j, k))
                vals.append(1 / (i + j + k + 3))
    
    coo = np.array(coo)
    vals = np.array(vals)
    return coo, vals     

def mttcrp(coo_tensor, vals, shape, mode, a, b):
    temp = np.zeros(shape=(shape[mode], a.shape[1]))
    
    if mode == 0:
        mode_a = 1 
        mode_b = 2
        
    elif mode == 1:
        mode_a = 0
        mode_b = 2
        
    else:
        mode_a = 0
        mode_b = 1
        
    for item in range(coo_tensor.shape[0]):
        coord = coo_tensor[item]
        temp[coord[mode], :] += a[coord[mode_a], :] * b[coord[mode_b], :] * vals[item] 
    
    return temp

def cp_als3(coo_tensor,
            vals,
            shape,
            rank=5,
            max_iter=200,
            tol=1e-8,
            pr=True):
    
    a = np.random.normal(0.0, 0.1, size=(shape[0], rank))
    b = np.random.normal(0.0, 0.1, size=(shape[1], rank))
    c = np.random.normal(0.0, 0.1, size=(shape[2], rank))
    err_arr = np.empty((max_iter, 1))
    
    it = 0
    err1 = 1.0
    err2 = 0.0
    while np.abs(err1 - err2) > tol:
        it += 1

        v1 = b.T @ b
        v2 = c.T @ c
        v = v1 * v2
        v = np.linalg.pinv(v)
        a = mttcrp(coo_tensor, vals, shape, 0, b, c) @ v
        
        v1 = a.T @ a
        v2 = c.T @ c
        v = v1 * v2
        v = np.linalg.pinv(v)
        b = mttcrp(coo_tensor, vals, shape, 1, a, c) @ v
        
        v1 = a.T @ a
        v2 = b.T @ b
        v = v1 * v2
        v = np.linalg.pinv(v)
        c = mttcrp(coo_tensor, vals, shape, 2, a, b) @ v
        
        error = sqrt_err_relative(coo_tensor, vals, shape, a, b, c)
        err_arr[it - 1] = error
        err2 = err1
        err1 = error
        if it == max_iter:
            if pr:
                print("iterations over")
            break
    
    return a, b, c, err_arr, it

def count_derivative(xi, mi, omega, mode = 'logit'):
    if (mode == 'logit'):
        return (math.exp(mi)/(1 + math.exp(mi)) - xi)/omega
    if (mode == 'regular'):
        return (1/(1 + mi) - xi/mi)/omega
 

def unfold(tensor, mode):
    
    return np.reshape(np.moveaxis(tensor, mode, 0),
                      (tensor.shape[mode], -1))


def fold(unfolded_tensor, mode, shape):
    
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(
          unfolded_tensor, full_shape), 0, mode)

def khatri_rao(matrix1, matrix2):
    n_columns = matrix1.shape[1]
    result = np.einsum('ik,jk->ijk', matrix1, matrix2)
    return result.reshape((-1, n_columns))
    
def factors_to_tensor_full(A, B, C):
    full_shape = (A.shape[0], B.shape[0], C.shape[0])
    unfolded_tensor = A.dot(khatri_rao(B, C).T)
    return fold(unfolded_tensor, 0, full_shape)

def der_to_coo(full_tensor, X_coords, X_val):
    to_ind = {}
    coo = []
    vals = []
    for i in range(full_tensor.shape[0]):
        for j in range(full_tensor.shape[1]):
            for k in range(full_tensor.shape[2]):
                if ([i , j, k] in X_coords):
                    xi = X_val[to_ind[[i , j, k]]]
                else:
                    xi = 0
                res = count_derivative(xi, full_tensor[i , j, k])
                if (res != 0 and res!= 0.0):
                    coo.append([i , j, k])
                    vals.append(res)
    return coo, vals

def kruskal_tensor(a, b, c):
    full_tensor = factors_to_tensor(a, b, c)
    print (full_tensor.shape, type(full_tensor))
    return full_tensor

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
    return coo_tensor, krus_vals


def gcp_als3(coo_tensor,
            vals,
            shape,
            rank=5,
            max_iter=200,
            tol=1e-8,
            pr=True):
    
    a = np.random.normal(0.0, 0.1, size=(shape[0], rank))
    b = np.random.normal(0.0, 0.1, size=(shape[1], rank))
    c = np.random.normal(0.0, 0.1, size=(shape[2], rank))
    
    it = 0
    err1 = 1.0
    err2 = 0.0
    lr = 0.1
    while np.abs(err1 - err2) > tol:
        it += 1
        
        # restored tensor M = [A_1, A_2, A_3] with the same size as initial tensor
        coo_restored, vals_restores = factors_to_tensor(coo_tensor, vals, a, b, c)
        print (len(coo_restored), len(coo_tensor))
        omega = len(vals)
        func_vals = []
        for ind, elem in enumerate(vals):
            func_vals.append(count_derivative(vals[ind], vals_restores[ind], omega))

        assert len(func_vals) == len(vals)
        err_arr = np.empty((max_iter, 1))
    
        v1 = b.T @ b
        v2 = c.T @ c
        v = v1 * v2
        v = np.linalg.pinv(v)
        g_a = mttcrp(coo_tensor, func_vals, shape, 0, b, c) @ v
        
        v1 = a.T @ a
        v2 = c.T @ c
        v = v1 * v2
        v = np.linalg.pinv(v)
        g_b = mttcrp(coo_tensor, func_vals, shape, 1, a, c) @ v
        
        v1 = a.T @ a
        v2 = b.T @ b
        v = v1 * v2
        v = np.linalg.pinv(v)
        g_c = mttcrp(coo_tensor, func_vals, shape, 2, a, b) @ v
        
        a = a + (lr * g_a)
        b = b + (lr * g_b)
        c = c + (lr * g_c)
        
        error = sqrt_err_relative(coo_tensor, vals, shape, a, b, c)
        err_arr[it - 1] = error
        err2 = err1
        err1 = error
        if it == max_iter:
            if pr:
                print("iterations over")
            break
    
    return a, b, c, err_arr, it

def get_elem_deriv_tensor(coo_tensor, vals, kruskal_vals, loss_function_grad):
    """
        Calculate the elementwise derivative tensor Y.
    """
    
    deriv_tensor_vals = loss_function_grad(vals, kruskal_vals) / vals.size
    return deriv_tensor_vals 

def sqrt_err(coo_tensor, vals, shape, a, b, c):
    result = 0.0
    for item in range(coo_tensor.shape[0]):
        coord = coo_tensor[item]
        result += (vals[item] - np.sum(
            a[coord[0], :] * b[coord[1], :] * c[coord[2], :]))**2        
    return np.sqrt(result)


def sqrt_err_relative(coo_tensor, vals, shape, a, b, c):
    result = sqrt_err(coo_tensor, vals, shape, a, b, c)        
    return result / np.sqrt((vals**2).sum())

def gcp_fg(coo_tensor, vals, shape, a, b, c, l2, loss_function, loss_function_grad):
    """
        GCP loss function and gradient calculation.
        All the tensors have the same coordinate set: coo_tensor.
    """
    
    # Construct sparse kruskal tensor
    kruskal_vals = factors_to_tensor(coo_tensor, vals, a, b, c)
    
    # Calculate mean loss on known entries
    loss_array = loss_function(vals, kruskal_vals)
    loss = np.mean(loss_array)
    
    # Compute the elementwise derivative tensor
    deriv_tensor_vals = get_elem_deriv_tensor(
        coo_tensor, vals, kruskal_vals, loss_function_grad
    )
    
    # Calculate gradients w.r.t. a, b, c factor matrices
    g_a = mttcrp(coo_tensor, deriv_tensor_vals, shape, 0, b, c)
    g_b = mttcrp(coo_tensor, deriv_tensor_vals, shape, 1, a, c)
    g_c = mttcrp(coo_tensor, deriv_tensor_vals, shape, 2, a, b)
    
    # Add L2 regularization
    if l2 != 0:
        g_a += l2 * a
        g_b += l2 * b
        g_c += l2 * c
    
    return loss, g_a, g_b, g_c

def gcp_gd(coo_tensor,
           vals,
           shape,
           loss_function,
           loss_function_grad,
           rank=5,
           lr=0.1,
           l2=0,
           max_iter=20,
           tol=1e-8,
           seed=13,
           show_iter=False,
           it_over=True):
    """
        Factorize initial sparse tensor by generalized CP decomposition.
    """
    
    random_state = np.random.seed(seed)
    a = np.random.normal(0.0, 0.1, size=(shape[0], rank))
    b = np.random.normal(0.0, 0.1, size=(shape[1], rank))
    c = np.random.normal(0.0, 0.1, size=(shape[2], rank))
    
    err_arr = np.empty((max_iter, 1))
    it = 0
    err1 = 1.0
    err2 = 0.0
    while np.abs(err1 - err2) > tol:
        
        # Get loss and gradients
        error, g_a, g_b, g_c = gcp_fg(
            coo_tensor, vals, shape, a, b, c,
            l2, loss_function, loss_function_grad,
        )
        
        # Update factor matrices
        a = a - (lr * g_a)
        b = b - (lr * g_b)
        c = c - (lr * g_c)
        
        it += 1
        err_arr[it - 1] = error
        err2 = err1
        err1 = error
        if show_iter:
            print("Iter: ", it, "; Error: ", error)
            
        if it == max_iter:
            if it_over:
                print("iterations over")
            break
    
    return a, b, c, err_arr, it

def bernoulli_logit_loss(x_vals, m_vals):
    return np.log(1 + np.exp(m_vals)) - (x_vals * m_vals)

def bernoulli_logit_loss_grad(x_vals, m_vals):
    exp_vals = np.exp(m_vals)
    return (exp_vals / (1 + exp_vals)) - x_vals

def main():

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
    
    
    values = [1] * len(train_triples)
    values = np.array(values, dtype=np.float64)

    coords = np.array(train_triples, dtype=np.int32)
    nnz = len(train_triples)
    data_shape = (len(entity_list), len(relation_list), len(entity_list))
        
    max_iter = 20
    rank = 200
    l2 = 4e0
    seed = 1
    
    a, b, c, err_arr, it = gcp_als3(
    coords, values, data_shape,
    rank=200,
    max_iter=max_iter
)
    print ("decomposed", a.shape, b.shape, c.shape)
    
    print ("create filters")
    
    with open("FB15K237_test.txt", "rb") as fp:   # Unpickling
        test_filter = pickle.load(fp)
    with open("FB15K237_train.txt", "rb") as fp:   # Unpickling
        valid_filter = pickle.load(fp)
    
    #test_filter = create_filter(test_triples, all_triples)  
    #valid_filter = create_filter(valid_triples, all_triples)  
    print ("created filters")
    
    hr1 = hr(test_filter, test_triples, a, b, c, [1, 3, 10])
    print ("hr 1 3 10 ", hr1)
    
    
    plt.xlabel("Iteration")
    plt.ylabel("Relative error")
    plt.title(f"WN18 / WRCP-ALS3(R={rank})")
    #plt.xticks(np.arange(it))
    plt.yscale("log")
    plt.plot(np.arange(1, it+1), err_arr[:it], '-*', c="#8b0a50")
    plt.savefig("iters_err.png")
    
if __name__ == "__main__":
    main()