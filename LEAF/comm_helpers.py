import collections
import logging
import math
import sys
import copy
# from more_itertools import sample

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


def fedAvg_communicate(global_model, models, args, Ls):
    global_model.to("cuda")
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
    global_model.to("cpu")
    return global_model

def LayerIS_communicate(global_model, models, args, Ls, P, layer_id,N):
    global_model.to("cuda")
    trained_dict = list(global_model.children())[layer_id].state_dict()
    trained_dict_update = {}
    
    
    for i, model in enumerate(models):
        model.to("cuda")
        model_dict = model.state_dict()
        for k, v in model_dict.items():
            update_data = (model_dict[k].data - trained_dict[k].data) *P[i]
            # update_data = (model_dict[k].data - trained_dict[k].data) *Ls[i] / (N*P[i])
            if k in trained_dict_update:
                trained_dict_update[k] += update_data
            else:
                trained_dict_update[k] = update_data
        model.to("cpu")

    for k, v in trained_dict.items():
        trained_dict[k] =  trained_dict[k] + (args.globallr * (trained_dict_update[k] ))

    list(global_model.children())[layer_id].load_state_dict(trained_dict)
    # global_model.load_state_dict(trained_dict)
    global_model.to("cpu")
    return global_model

def IS_aggregation(global_model, models, args, Ls,P,N):
    global_model.to("cuda")
    trained_dict = global_model.state_dict()
    trained_dict_update = {}
    
    
    for i, model in enumerate(models):
        model.to("cuda")
        model_dict = model.state_dict()
        for k, v in model_dict.items():
            update_data = (model_dict[k].data - trained_dict[k].data) *P[i]
            if k in trained_dict_update:
                trained_dict_update[k] += update_data
            else:
                trained_dict_update[k] = update_data
        model.to("cpu")

    for k, v in trained_dict.items():
        # vt[k] = 0.2*vt[k]+args.globallr * (trained_dict_update[k] / sum(Ls))
        trained_dict[k] =  trained_dict[k] + (args.globallr * (trained_dict_update[k]))
        # trained_dict[k] =  trained_dict[k] + vt[k]
    
    global_model.load_state_dict(trained_dict)
    global_model.to("cpu")
    return global_model

def IS_communicate(global_model, models, args, P, N, Ls,vt, S):
    global_model.to("cuda")
    trained_dict = global_model.state_dict()
    trained_dict_update = {}
    model_temp = []
    for j in range(S):
        model_temp.append(models[j])
    for i, model in enumerate(model_temp):
        model.to("cuda")
        model_dict = model.state_dict()
        for k, v in model_dict.items():
            update_data = (model_dict[k].data - trained_dict[k].data) *Ls[i] / (N*P[i])
            if k in trained_dict_update:
                trained_dict_update[k] += update_data
            else:
                trained_dict_update[k] = update_data
        model.to("cpu")

#     sum_P = sum([(1 / p) for p in P])
    for k, v in trained_dict.items():
        # vt[k] = 0.2*vt[k]+args.globallr * (trained_dict_update[k] / sum(Ls))
        trained_dict[k] =  trained_dict[k] + (args.globallr * (trained_dict_update[k] / sum(Ls)))
        # trained_dict[k] =  trained_dict[k] + vt[k]
        
    global_model.load_state_dict(trained_dict)
    global_model.to("cpu")
    return global_model

def IS_communicate_cluster(global_model, models, args, sample_index_temp,P_cluster, N, Ls ):
    global_model_cluster = copy.deepcopy(global_model)
    global_model_cluster.to("cuda")
    trained_dict = global_model_cluster.state_dict()
    trained_dict_update = {}
    sum_LS = 0

    for i, model in enumerate(models):
        model.to("cuda")
        model_dict = model.state_dict()
        for k, v in model_dict.items():
            update_data = (model_dict[k].data - trained_dict[k].data) *Ls[i] / (len(P_cluster)*P_cluster[sample_index_temp[i]])
            if k in trained_dict_update:
                trained_dict_update[k] += update_data
            else:
                trained_dict_update[k] = update_data
        model.to("cpu")
        sum_LS += Ls[i]
#     sum_P = sum([(1 / p) for p in P])
    for k, v in trained_dict.items():
        # vt[k] = 0.2*vt[k]+args.globallr * (trained_dict_update[k] / sum(Ls))
        trained_dict[k] =  trained_dict[k] + (args.globallr * (trained_dict_update[k] / sum(Ls)))
        # trained_dict[k] =  trained_dict[k] + vt[k]
        
    global_model_cluster.load_state_dict(trained_dict)
    global_model_cluster.to("cpu")
    return global_model_cluster

def Avg_communicate(global_model, models, args, Ls,vt):
    trained_dict = global_model.state_dict()
    trained_dict_update = {}
    
    
    for i, model in enumerate(models):
        model.to("cuda")
        model_dict = model.state_dict()
        for k, v in model_dict.items():
            update_data = (model_dict[k].data - trained_dict[k].data) 
            if k in trained_dict_update:
                trained_dict_update[k] += update_data
            else:
                trained_dict_update[k] = update_data
        model.to("cpu")

    for k, v in trained_dict.items():
        # vt[k] = 0.2*vt[k]+args.globallr * (trained_dict_update[k] / len(Ls))
        trained_dict[k] =  trained_dict[k] + (args.globallr * (trained_dict_update[k] / len(Ls)))
        # trained_dict[k] =  trained_dict[k] + vt[k]
    
    global_model.load_state_dict(trained_dict)
    return global_model

def FedVARP_communicate(global_model, models, args, Ls, y_t, yi_t,  N, S, sample_index):
    trained_dict = global_model.state_dict()
    # trained_dict_update = {}
    varp_update_dict = {}
    Varp_t = {}
    
    for i, model in enumerate(models):
        model.to("cuda")
        model_dict = model.state_dict()
        for k, v in model_dict.items():
            # update_data = (model_dict[k].data - trained_dict[k].data) * Ls[i]
            varp_update_data = (trained_dict[k].data - model_dict[k].data - yi_t[sample_index[i]][k])* Ls[i]
            if k in varp_update_dict:
                # trained_dict_update[k] += update_data
                varp_update_dict[k]+= varp_update_data
            else:
                # trained_dict_update[k] = update_data
                varp_update_dict[k] = varp_update_data
        for k, v in model_dict.items():
            yi_t[sample_index[i]][k] = (trained_dict[k].data - model_dict[k].data) 
        model.to("cpu")

    for k, v in trained_dict.items():
        # vt[k] = 0.2*vt[k]+args.globallr * (trained_dict_update[k] / sum(Ls))
        Varp_t[k] = y_t[k] + varp_update_dict[k]/ sum(Ls)
        trained_dict[k] =  trained_dict[k] - (args.globallr * Varp_t[k] )
        # trained_dict[k] =  trained_dict[k] + vt[k]
        y_t[k] = y_t[k] + (varp_update_dict[k]/ sum(Ls))*(S/N)
    
    
    global_model.load_state_dict(trained_dict)
    return global_model, y_t, yi_t

def FedVARP_IS_communicate(global_model, models, args, Ls, y_t, yi_t,  N, S, sample_index, P):
    trained_dict = global_model.state_dict()
    # trained_dict_update = {}
    varp_update_dict = {}
    varp_update_dict_2={}
    Varp_t = {}
    
    for i, model in enumerate(models):
        model.to("cuda")
        model_dict = model.state_dict()
        for k, v in model_dict.items():
            # update_data = (model_dict[k].data - trained_dict[k].data) * Ls[i]
            varp_update_data_1 = (trained_dict[k].data - model_dict[k].data)* Ls[i] #/ (N*P[i])
            varp_update_data_2 = ((trained_dict[k].data - model_dict[k].data)-yi_t[sample_index[i]][k])* Ls[i] / (N*P[i])
            varp_update_data = varp_update_data_1 -yi_t[sample_index[i]][k]* Ls[i]
            if k in varp_update_dict:
                # trained_dict_update[k] += update_data
                varp_update_dict[k]+= varp_update_data
                varp_update_dict_2[k]+= varp_update_data_2
            else:
                # trained_dict_update[k] = update_data
                varp_update_dict[k] = varp_update_data
                varp_update_dict_2[k] = varp_update_data_2
        for k, v in model_dict.items():
            yi_t[sample_index[i]][k] = (trained_dict[k].data - model_dict[k].data) 
        model.to("cpu")

    for k, v in trained_dict.items():
        # vt[k] = 0.2*vt[k]+args.globallr * (trained_dict_update[k] / sum(Ls))
        Varp_t[k] = y_t[k] + varp_update_dict[k]/ sum(Ls)
        trained_dict[k] =  trained_dict[k] - (args.globallr * Varp_t[k] )
        # trained_dict[k] =  trained_dict[k] + vt[k]
        y_t[k] = y_t[k] + (varp_update_dict_2[k]/ sum(Ls))*(S/N)
    
    
    global_model.load_state_dict(trained_dict)
    return global_model, y_t, yi_t


def FedVARP_communicate_modify(global_model, models, args, Ls, y_t, yi_t,  N, S, sample_index, P,y_t_middle):
    trained_dict = global_model.state_dict()
    # trained_dict_update = {}
    varp_update_dict = {}
    Varp_t = {}
    
    for i in range(N):
        for k, v in trained_dict.items():
            y_t_middle[k] =  yi_t[i][k]+y_t_middle[k]
    for i, model in enumerate(models):
        model.to("cuda")
        model_dict = model.state_dict()
        for k, v in model_dict.items():
            # update_data = (model_dict[k].data - trained_dict[k].data) * Ls[i]
            varp_update_data = ((trained_dict[k].data - model_dict[k].data ) - yi_t[sample_index[i]][k])#/ (N*P[i])
            if k in varp_update_dict:
                # trained_dict_update[k] += update_data
                varp_update_dict[k]+= varp_update_data
            else:
                # trained_dict_update[k] = update_data
                varp_update_dict[k] = varp_update_data
        for k, v in model_dict.items():
            yi_t[sample_index[i]][k] = (trained_dict[k].data - model_dict[k].data) 
        model.to("cpu")
    

    for k, v in trained_dict.items():
        # vt[k] = 0.2*vt[k]+args.globallr * (trained_dict_update[k] / sum(Ls))
        Varp_t[k] = (y_t_middle[k] + varp_update_dict[k])/ N
        trained_dict[k] =  trained_dict[k] - (args.globallr * Varp_t[k] )
        # trained_dict[k] =  trained_dict[k] + vt[k]
        y_t[k] = y_t[k] + (varp_update_dict[k]/ sum(Ls))*(S/N)

    global_model.load_state_dict(trained_dict)
    return global_model, y_t, yi_t
    