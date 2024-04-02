import collections
import logging
import math
import sys
import copy

import torch
import torch.distributed as dist

def schedule_lr(t, args):
    if t >= 0.75 * args.round_num:
        return args.lr * 0.01
    if t >= 0.5 * args.round_num:
        return args.lr * 0.1
    return args.lr

def combine_reduction(reductions, new_reductions, t):
    for i, (reduction, new_reduction) in enumerate(zip(reductions, new_reductions)):
        for j, (r, n_r) in enumerate(zip(reduction, new_reduction)):
            reductions[i][j] = (r * t + n_r) / (t + 1)
    return reductions


def fedAvg_communicate(global_model, models, args, Ls,vt):
    trained_dict = global_model.state_dict()
    trained_dict_update = {}
    
    
    for i, model in enumerate(models):
        model.to("cuda")
        model_dict = model.state_dict()
        for k, v in model_dict.items():
            update_data = (model_dict[k].data - trained_dict[k].data) * Ls[i]
            if k in trained_dict_update:
                trained_dict_update[k] += update_data
            else:
                trained_dict_update[k] = update_data
        model.to("cpu")

    for k, v in trained_dict.items():
        # vt[k] = 0.2*vt[k]+args.globallr * (trained_dict_update[k] / sum(Ls))
        trained_dict[k] =  trained_dict[k] + (args.globallr * (trained_dict_update[k] / sum(Ls)))
        # trained_dict[k] =  trained_dict[k] + vt[k]
    
    global_model.load_state_dict(trained_dict)
    return global_model

def IS_communicate(global_model, models, args, P, N, Ls,vt):
    trained_dict = global_model.state_dict()
    trained_dict_update = {}
    
    for i, model in enumerate(models):
        model.to("cuda")
        model_dict = model.state_dict()
        for k, v in model_dict.items():
            update_data = (model_dict[k].data - trained_dict[k].data) *Ls[i] / (N*P[i])
            if k in trained_dict_update:
                trained_dict_update[k] += update_data
            else:
                trained_dict_update[k] = update_data
        model.to("cuda")

#     sum_P = sum([(1 / p) for p in P])
    for k, v in trained_dict.items():
        # vt[k] = 0.2*vt[k]+args.globallr * (trained_dict_update[k] / sum(Ls))
        trained_dict[k] =  trained_dict[k] + (args.globallr * (trained_dict_update[k] / sum(Ls)))
        # trained_dict[k] =  trained_dict[k] + vt[k]
    
    global_model.load_state_dict(trained_dict)
    return global_model