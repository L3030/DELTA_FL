from distutils.log import Log
from dis import dis
from builtins import int
import pandas as pd
import copy
from torch.nn import parameter
from torch import optim
from torch.optim import optimizer
from cNNmodel import   MLP, Net, Logistic, CNNNet
from losses.cflLoss import CFLLoss
from numpy.core.fromnumeric import partition
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.serialization import save
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import pdb
import os
from torch.distributions.dirichlet import Dirichlet
from copy import deepcopy
import torch.nn.functional as F
from torchvision import datasets, models
from torch.autograd import Variable, grad, grad_mode
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
import SplitDataset
from SplitDataset import Partition, SplitDataset
import optimizers.fedProx
from optimizers.fedProx import FedProx
import optimizers.fedNova
from optimizers.fedNova import FedNova
from optimizers.fcl import FCLR
from comm_helpers import IS_communicate_cluster, combine_reduction, fedAvg_communicate, IS_communicate, Avg_communicate, FedVARP_communicate,FedVARP_IS_communicate,FedVARP_communicate_modify
import torch.distributions.dirichlet as dirichlet
from hessian_tools import combine_hessian, get_SI_omega, get_diag_fisher, get_diag_hessian, copy_hessian
import os
from pathlib import Path
import h5py
from torch.utils.data import TensorDataset
import heapq
from torchvision.transforms.functional import rotate
from py_func.clustering import sample_clients_cluster, get_clusters_with_alg1

def schedule_lr(t, args):
    # args.lr = args.lr/(t+1)**0.5
    # if t >= 0.5 * args.round_num:
    #     return 0.1
    # else:
    #     return 0.01
    
    return args.lr


def choose_optimizer(model, args, t):
    lr = schedule_lr(t, args)
    return FedProx(model.parameters(), lr = lr,  mu=0)

    
def get_csd_loss( model, mu):
        loss_set = []
        for name, param in model.named_parameters():
            theta = model.state_dict()[name]
            loss_set.append((0.5) * ((theta - mu[name]) ** 2).sum())
        return sum(loss_set)


def local_epoch(model, global_model, data, device, args, t , local_ep, data_set):
    model.train()
    global_model.to(device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = choose_optimizer(model, args, t)
    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=schedule_lr(t, args))
    gradients = [torch.zeros_like(p) for p in model.parameters()]
    count = 0
    # new_mu = dict()
    # global_dict = global_model.state_dict()
    # for name, param in model.named_parameters():
    #         new_mu[name] = deepcopy(global_dict[name])
    for k in range(local_ep):
        # loss_record = []
        all_x = torch.cat([x for x, y in data]).to(device)
        all_y = torch.cat([y for x, y in data]).to(device)
        # for (X, Y) in data:
            
        count += 1
        # if args.datasetname == "cookup_train_1" or "cookup_train_2" or "cookup_train_3":
        #     X = X.unsqueeze(1)
        # X, Y = X.to(device), Y.to(device)
        # X, Y = Variable(X), Variable(Y)
        optimizer.zero_grad() 
        output = model(all_x)
        # output = model(X)
        # Y = Y.long()
        # print(output)
        # print(Y)
        loss = criterion(output, all_y)
        # loss = criterion(output, Y)
        # print("ce_loss:", ce_loss)
        # csd_loss = get_csd_loss(model, new_mu) if args.csd_importance > 0 else 0
        # print("csd_loss:" ,csd_loss)
        #loss = ce_loss  + args.csd_importance * csd_loss
        # print("total loss:", loss)
        loss.backward()
        # loss_record.append(loss)
        gradients = [g + p.grad.clone().detach() if p.grad is not None else g for g, p in zip(gradients, model.parameters())]
        optimizer.step()
        # X.to("cpu"), Y.to("cpu")
    # losses = sum(loss_record)/len(loss_record)   # final loss
    gradients = [g / count for g in gradients]
    # print("model:",model.state_dict())

    return model, len(data_set), gradients

def Average_model(global_model, models):
    global_model.to("cuda")
    trained_dict = global_model.state_dict()
    trained_dict_update = {}
    
    
    for i, model in enumerate(models):
        model.to("cuda")
        model_dict = model.state_dict()
        for k, v in model_dict.items():
            update_data = model_dict[k].data
            if k in trained_dict_update:
                trained_dict_update[k] += update_data
            else:
                trained_dict_update[k] = update_data
        model.to("cpu")

    for k, v in trained_dict.items():
        # vt[k] = 0.2*vt[k]+args.globallr * (trained_dict_update[k] / sum(Ls))
        trained_dict[k] = trained_dict_update[k] / len(models)
        # trained_dict[k] =  trained_dict[k] + vt[k]
    
    global_model.load_state_dict(trained_dict)
    global_model.to("cpu")
    return global_model

def rotate_dataset(images, labels, angle):
    # rotation = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
    #         interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
    #     transforms.ToTensor()])
    
    rotation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Lambda(lambda x: rotate(x, angle, fill=(0,))),
        transforms.ToTensor()])

    x = torch.zeros(len(images), 1, 28, 28)
    for i in range(len(images)):
        x[i] = rotation(images[i])
    # y = labels.view(-1)
    labels = torch.tensor(labels)
    return TensorDataset(x, labels)

def get_noniid_class_and_labels( original_images, original_labels, N):

        M = len(original_labels) // N

        clients_images = [[] for _ in range(N)]
        clients_labels = [[] for _ in range(N)]
        classes_by_index = [[] for _ in range(10)]
        classes_by_index_len = [0 for _ in range(10)]
        for i, label in enumerate(original_labels):
            classes_by_index[label].append(i)
            classes_by_index_len[label] += 1

        
        for i in range(N):
            p = torch.tensor(classes_by_index_len) / sum(classes_by_index_len)
            q = dirichlet.Dirichlet(0.1 * p).sample()
            while(len(clients_labels[i]) < M):
                sampled_class = torch.multinomial(q, 1)
                if classes_by_index_len[sampled_class] == 0:
                    q = reweight(q, sampled_class)
                    # print(q)
                else:
                    sampled_index = random.randint(0, classes_by_index_len[sampled_class] - 1)
                    sampled_original_index = classes_by_index[sampled_class][sampled_index]
                    clients_images[i].append(original_images[sampled_original_index])
                    clients_labels[i].append(original_labels[sampled_original_index])
                    classes_by_index[sampled_class].pop(sampled_index)
                    classes_by_index_len[sampled_class] -= 1
            clients_labels[i] = torch.tensor(clients_labels[i])
            clients_images[i] = torch.tensor([image.numpy() for image in clients_images[i]])
        
        return clients_images, clients_labels
def reweight(q, empty_class):
    # sum_q = sum(q)
    q[empty_class] = 0
    q = q / sum(q)
    return q
class FedTrain():

    def __init__(self, args) -> None:
        # useful parameters
        self.N= args.client_num
        self.T = args.round_num
        self.S= args.sample_num
        self.global_lr = args.globallr
        
        # # generate data
        # self.new_partitioner = SplitDataset(args)
        # self.total_train_sets = self.new_partitioner.trainset
        # # self.train_sets = self.new_partitioner.client_sets
        # self.sampled_clients = self.new_partitioner.sampled_clients
        # self.test_loader = self.new_partitioner.test_loader

        
     
        # data_temp = dataloader.DataLoader(self.sampled_clients[0], batch_size= 3, shuffle=True)
        # for x,y in data_temp:
        #     self.sampled_clients.append(TensorDataset(x,y))
        # self.N = len(self.sampled_clients)
          

        traindata = datasets.MNIST( root=args.datapath, train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
                       )
        self.sampled_clients = [[] for _ in range(10)]
        for x, y in zip(traindata.data, traindata.targets):
            self.sampled_clients[y].append(torch.tensor(x.unsqueeze(0).unsqueeze(0), dtype=torch.float32))
        print(torch.cat(self.sampled_clients[0]).shape)
        
        clients_data_x, clients_data_y =get_noniid_class_and_labels(torch.cat(self.sampled_clients[0]), torch.tensor(torch.ones((len(self.sampled_clients[0]),)) * 0, dtype=int),10)
        
        for i in range(10):
            self.sampled_clients[i] = TensorDataset(torch.cat(self.sampled_clients[i]), torch.tensor(torch.ones((len(self.sampled_clients[i]),)) * i, dtype=int))
        # clients_data_x, clients_data_y =get_noniid_class_and_labels(torch.cat(self.sampled_clients[i]), torch.tensor(torch.ones((len(self.sampled_clients[i]),)) * i, dtype=int),100)
        del(self.sampled_clients[0])
        for _ in range(10):
            self.sampled_clients += self.sampled_clients
        for x, y in zip(clients_data_x, clients_data_y):
            # for _ in range(2):
            self.sampled_clients.append(TensorDataset(x,y))
        # data_temp = dataloader.DataLoader(self.sampled_clients[0], batch_size= 600, shuffle=True)

        # for x,y in data_temp:
        #     self.sampled_clients.append(TensorDataset(x,y))
        
        
        
        # data_temp = dataloader.DataLoader(self.sampled_clients[1], batch_size= 60, shuffle=True)
        # for x,y in data_temp:
        #     self.sampled_clients.append(TensorDataset(x,y))
        # data_temp = dataloader.DataLoader(self.sampled_clients[2], batch_size= 60, shuffle=True)
        # for x,y in data_temp:
        #     self.sampled_clients.append(TensorDataset(x,y))
        self.N = len(self.sampled_clients)
        
        # for i in range(self.N):
        #     images = []
        #     targets = []
        #     for x, y in self.sampled_clients[i]:
        #         images.append(x)
        #         targets.append(y)
        #     self.sampled_clients[i] = rotate_dataset(images,targets,float(i))
        
        # target_labels = torch.stack([traindata.targets == i for i in range(10)])
        # target_labels_split = []
        # for i in range(10):
        #     target_labels_split += torch.split(torch.where(target_labels[i].sum(0))[0], int(60000 / 10))
        # print(len(target_labels_split))
        # traindata_split = [torch.utils.data.Subset(traindata, tl) for tl in target_labels_split]
        # self.sampled_clients = traindata_split
        # print("len_traindata", len(self.sampled_clients[0]))
        # data_temp = dataloader.DataLoader(self.sampled_clients[0], batch_size= 60, shuffle=True)
        # print(data_temp)
        # print("len_datatemp", len(data_temp))
        # for x,y in data_temp:
        #     print("one x y")
        #     self.sampled_clients.append(TensorDataset(x,y))
        
        # del(self.sampled_clients[0])
        self.N = len(self.sampled_clients)
        
        # self.sampled_clients = [torch.utils.data.DataLoader(x, batch_size=args.bs, shuffle=True) for x in traindata_split]

        self.test_loader = torch.utils.data.DataLoader(
                datasets.MNIST( root=args.datapath, train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
                ), batch_size=args.bs, shuffle=True)
        
            
            
        
        
        
        
        
        # # dataset of peter team
        # f = h5py.File('./data_peter/dataverse/{}.h5'.format(args.datasetname),"r")
        # f_test = h5py.File('./data_peter/dataverse/test.h5',"r")
        # d1_group = f["examples"]
        # test_group = f_test["examples"]
        # # test_len = len(f_test)
        # # clients_set_key = []
        # # test_set_key = []
        
        # train_set_peter_loader=[]
        # test_set_peter_loader = []

        # for key in d1_group.keys():
        #     # clients_set_key.append(key)
        #     d1_group_group = d1_group[key]
        #     # train_set_peter.append(d1_group_group["pixels"])
        #     # test_set_peter.append(d1_group_group["label"])
        #     d1_list = torch.tensor(d1_group_group["pixels"][()])
        #     d1_list_iter = [d1_list[i,:,:] for i in range(d1_list.size()[0])]
        #     a_concatenate = []
        #     for i in range(10):
        #         a_concatenate = a_concatenate + d1_list_iter
        #     x_group = torch.stack(a_concatenate)

        #     d1_list_y = torch.tensor(d1_group_group["label"][()])
        #     d1_list_iter_y = [d1_list_y[i] for i in range(d1_list_y.size()[0])]
        #     a_concatenate_y = []
        #     for i in range(10):
        #         a_concatenate_y = a_concatenate_y + d1_list_iter_y
        #     y_group = torch.stack(a_concatenate_y)
        
        #     train_set_peter_loader.append(TensorDataset(x_group,y_group))
        # self.sampled_clients = train_set_peter_loader
        # x_test = []
        # y_test = []
        # for key in test_group.keys():
        #     # test_set_key.append(key)
        #     test_group_group= test_group[key]
        #     test_list_x = test_group_group["pixels"][()]
        #     test_list_y = test_group_group["label"][()]
        #     for i in range(len(test_list_y)):
        #         x_test.append(test_list_x[i])
        #         y_test.append(test_list_y[i])
        
        # test_set_peter_loader=TensorDataset(torch.tensor(x_test),torch.tensor(y_test))  

        #    # test_set_peter_loader.append(TensorDataset(torch.tensor(test_group_group["pixels"][()]), torch.tensor(test_group_group["label"][()])))
        
        # self.test_loader = test_set_peter_loader

        
        # initial global model and local models
        self.global_model, self.models = self.init_models()

    def init_models(self):
        # # load finetune models
        # local_models = [models.resnet18(pretrained=False) for i in range(self.S)]
        # global_model = models.resnet18(pretrained=False)
        # # change output layer
        # num_ftrs = global_model.fc.in_features
        # global_model.fc = nn.Linear(num_ftrs, 10)
        # for i in range(self.S):
        #     local_models[i].fc = nn.Linear(num_ftrs, 10)
        # local_models = [Net() for i in range(self.S)]
        # global_model = Net()
        local_models = [Logistic() for i in range(self.S)]
        global_model = Logistic()
        return global_model, local_models
    
    def set_weights(self):
        # transmit parameters to local models

        global_dict = self.global_model.state_dict()

        for i in range(len(self.models)):
            self.models[i].load_state_dict(deepcopy(global_dict))
    
    def combine_weights(self, args, Ls, P,vt):
        # combine local parameters for global model
        if args.optimizer == 'FedAvg':
            self.global_model = fedAvg_communicate(self.global_model, self.models, args, Ls,vt)
        elif args.optimizer == 'MD':
            self.global_model = Avg_communicate(self.global_model, self.models, args, Ls,vt)
        else:
            self.global_model = IS_communicate(self.global_model, self.models, args, P, self.N, Ls,vt)
        return 
    
    def conbine_weights_cluster(self, args, Ls, sample_index_temp, P_cluster):
        global_model_total = []
        
        # for i in range(args.cluster_num):
        #     global_model_total.append([])
            
        for i in range(args.cluster_num):
            if len(P_cluster[i]) == 0:
                pass
            else:
                model_len = []
                # Ls = [Ls[k] for k in sample_index_temp[i]]
                for k in range(int(self.S/args.cluster_num)):
                    model_len.append(int(self.S/args.cluster_num)*i+k)
                
                models = [self.models[i] for i in model_len]
                global_model_total_temp = IS_communicate_cluster(self.global_model, models, args, sample_index_temp[i],P_cluster[i],  self.N, Ls)
                global_model_total.append(global_model_total_temp)
        # aa = 0*global_model_total[0]
        # count = 0
        # for i in range(len(global_model_total)):
        #     count +=1
        #     aa += i
        # self.global_model = aa/count
        self.global_model = Average_model(self.global_model, global_model_total)
        return
        

    def combine_weights_VARP(self, args, Ls,  y_t, yi_t, sample_index):
        self.global_model, y_t, yi_t = FedVARP_communicate(self.global_model, self.models, args, Ls, y_t, yi_t,self.N,self.S,sample_index)
        return y_t, yi_t
    def combine_weights_VARP_IS(self, args, Ls, P, y_t, yi_t, sample_index):
        self.global_model, y_t, yi_t = FedVARP_IS_communicate(self.global_model, self.models, args, Ls, y_t, yi_t,self.N,self.S,sample_index,P)
        return y_t, yi_t
    def combine_weights_VARP_IS_modify(self, args, Ls, P, y_t, yi_t, sample_index,y_t_middle):
        self.global_model, y_t, yi_t = FedVARP_communicate_modify(self.global_model, self.models, args, Ls, y_t, yi_t,self.N,self.S,sample_index,P, y_t_middle)
        return y_t, yi_t

    def sample_clients(self, args, P, N_uniform_sample_index):
        # if args.optimizer == 'FedAvg' or "FedVARP":
        #     print("uniform sampling========================================")
        if args.optimizer == "FedAvg":
            index = np.random.randint(0, self.N, self.S)
            a=[]
            for i in index:
                a.append(N_uniform_sample_index[i])
            return a, index
        # elif args.optimizer == "MD":
        #     return torch.multinomial(torch.tensor(ratio_p), self.S, False)
        else:
            b_index = torch.multinomial(torch.tensor(P), self.S, False)
            b = []
            for i in b_index:
                b.append(N_uniform_sample_index[i])
            return b, b_index
        # else:
        #     b_index_temp = np.random.randint(0, self.N, 32)
        #     bb = []
        #     for i in b_index_temp:
        #         bb.append(N_uniform_sample_index[i])
        #     b_index = torch.multinomial(torch.tensor(P), self.S, False) 

    
    # def reweight_loss(self, P, sample_index, losses, Ls):
    #     alpha = 0.0
    #     sum_L = sum(Ls)
    #     weights = [ L / sum_L * sum([torch.norm(p) ** 2 for p in g]) ** 0.5 for L, g in zip(Ls, losses)]
        
    #     sum_weights = sum(weights)
    #     sum_old_weights = sum([P[i] for i in sample_index])
    #     weights = [ w / sum_weights * sum_old_weights for w in weights]
    #     for i in range(len(weights)):
    #         P[sample_index[i]] = alpha * P[sample_index[i]] + (1 - alpha) * weights[i]
    #     return P

    def reweight(self, P, sample_index, grads, Ls):
        alpha = 0.0
        sum_L = sum(Ls)
        weights = [ L / sum_L * sum([torch.norm(p) ** 2 for p in g]) ** 0.5 for L, g in zip(Ls, grads)]
        
        sum_weights = sum(weights)
        sum_old_weights = sum([P[i] for i in sample_index])
        weights = [ w / sum_weights * sum_old_weights for w in weights]
        for i in range(len(weights)):
            P[sample_index[i]] = alpha * P[sample_index[i]] + (1 - alpha) * weights[i]
        return P

    def reweight_prac(self, P, sample_index, grads, Ls,t, args):
        alpha = 0.0
        beta = args.beta

        c = [0]*len(grads[0])
        for i in range(len(grads)):
            for j in range(len(c)):
                # print("round:" , i)
                # print("gradsij.size:", grads[i][j].size())
                c[j] += grads[i][j]
        
        c = [i/len(grads) for i in c]
        sum_L = sum(Ls)
        weights1 = [ L / sum_L * sum([torch.norm(p) ** 2 for p in g]) ** 0.5 for L, g in zip(Ls, grads)]
        

        for i in range(len(grads)):
            for j in range(len(c)):
                grads[i][j] = grads[i][j] - c[j]

        weights2 = [ L / sum_L * sum([torch.norm(p) ** 2 for p in g]) ** 0.5 for L, g in zip(Ls, grads)]

        weights =[(1-beta)* a + beta*b for a,b in zip(weights1,weights2) ] 
        sum_weights = sum(weights)
        sum_old_weights = sum([P[i] for i in sample_index])
        weights = [ w / sum_weights * sum_old_weights for w in weights]
        for i in range(len(weights)):
            P[sample_index[i]] = alpha * P[sample_index[i]] + (1 - alpha) * weights[i]
        
        return P

    def reweight_theory(self, P, device, args, t , sample_index):
        client_grads = []
        # client_loss = []
        # sample_index = [i for i in range(self.N)]
        Ls = []

        for i in range(self.N):
            model = deepcopy(self.global_model)
            data_set = self.sampled_clients[sample_index[i]]
            data_loader = dataloader.DataLoader(data_set, args.bs, True, drop_last=False)
            model, L, grads = local_epoch(model.to(device),self.global_model, data_loader, device, args, t , local_ep=1, data_set = data_set)
            client_grads.append(grads)
            # client_loss.append(local_loss)
            Ls.append(L)
        sum_L = sum(Ls)
        
        weights = [ ( L / sum_L * sum([torch.norm(p) ** 2 for p in g]) ** 0.5) for L, g in zip(Ls, client_grads)]
        # print("p_before uniform:", weights)
        sum_weights = sum(weights)
        
        P = [ w / sum_weights for w in weights]
        # print("p after uniform",  P)
        return P
    def reweight_theory_cluster(self, P, device, args, t , cluster_list_index):
        client_grads = []
        # client_loss = []
        # sample_index = [i for i in range(self.N)]
        Ls = []

        for i in range(len(cluster_list_index)):
            model = deepcopy(self.global_model)
            data_set = self.sampled_clients[cluster_list_index[i]]
            data_loader = dataloader.DataLoader(data_set, args.bs, True, drop_last=False)
            model, L, grads = local_epoch(model.to(device),self.global_model, data_loader, device, args, t , local_ep=1, data_set = data_set)
            client_grads.append(grads)
            # client_loss.append(local_loss)
            Ls.append(L)
        sum_L = sum(Ls)
        weights = [ ( L / sum_L * sum([torch.norm(p) ** 2 for p in g]) ** 0.5) for L, g in zip(Ls, client_grads)]
        
        sum_weights = sum(weights)
        
        P = [ w / sum_weights for w in weights]
        
        
        return P
    
    def client_gradient_norm(self, device, args, t):
        client_grads = []
        # client_loss = []
        sample_index = [i for i in range(self.N)]
        Ls = []

        for i in range(self.N):
            model = deepcopy(self.global_model)
            data_set = self.sampled_clients[sample_index[i]]
            data_loader = dataloader.DataLoader(data_set, args.bs, True, drop_last=False)
            model, L, grads = local_epoch(model.to(device),self.global_model, data_loader, device, args, t , local_ep=1, data_set=data_set )
            client_grads.append(grads)
            # client_loss.append(local_loss)
            Ls.append(L)
        sum_L = sum(Ls)
        weights = [ ( sum([torch.norm(p) ** 2 for p in g]) ** 0.5) for  g in  client_grads]
        # weights = [ ( L / sum_L * sum([torch.norm(p) ** 2 for p in g]) ** 0.5) for L, g in zip(Ls, client_grads)]
        weights = [s.to("cpu") for s in weights]
        return weights
    
    def get_cluster_index(self,args,  cluster_center_index, device, t, cluster_center):
        cluster_list = []
        cluster_list_index = []
        sampled_cluster_list = []
        cluster_center_new = []
        for i in range(len(cluster_center_index)):
            cluster_list.append([])
            cluster_list_index.append([])
            sampled_cluster_list.append([])
            cluster_center_new.append([])
        gradient_norm = self.client_gradient_norm( device, args, t )
        for i in range(self.N):
            gradient_temp = (gradient_norm[i] - cluster_center[0])**2
            I_belong = 0
            for j in range(len(cluster_center)):
                if gradient_temp >= (gradient_norm[i] - cluster_center[j])**2 :
                    gradient_temp = (gradient_norm[i] - cluster_center[j])**2 
                    I_belong = j
                else: gradient_temp = gradient_temp
            for m in range(len(cluster_center_index)):
                if I_belong == m:
                    cluster_list[m].append(gradient_norm[i]) 
                    cluster_list_index[m].append(i)   
        
        for i in range(len(cluster_center)):
            count = 0
            cluster_center_new[i] = 0
            for j in range(len(cluster_list[i])):
                count +=1
                cluster_center_new[i] +=cluster_list[i][j]
            if count == 0:
                cluster_center_new[i] = cluster_center[i]
            else:
                cluster_center_new[i] = cluster_center_new[i]/count
            
        return cluster_list, cluster_list_index, cluster_center_new
    
    def reweight_prac_theory(self, P, device, args, t ,sample_index):
        
        client_grads = []
        # client_loss = []
        # sample_index = [i for i in range(self.N)]
        Ls = []

        for i in range(self.N):
            model = deepcopy(self.global_model)
            data_set = self.sampled_clients[sample_index[i]]
            data_loader = dataloader.DataLoader(data_set, args.bs, True, drop_last=False)
            model, L, grads = local_epoch(model.to(device),self.global_model, data_loader, device, args, t , local_ep=1, data_set = data_set)
            client_grads.append(grads)
            # client_loss.append(local_loss)
            Ls.append(L)
        sum_L = sum(Ls)
      
        c = [0]*len(client_grads[0])
        for i in range(len(client_grads)):
            for j in range(len(c)):
                # print("round:" , i)
                # print("gradsij.size:", grads[i][j].size())
                c[j] += client_grads[i][j]
        
        c = [i/len(client_grads) for i in c]
        for i in range(len(client_grads)):
            for j in range(len(c)):
                client_grads[i][j] = client_grads[i][j] - c[j]
        weights2 = [ args.beta* (L / sum_L) * (sum([torch.norm(p) ** 2 for p in g])) ** 0.5 + (1-args.beta)*(L / sum_L) for L, g in zip(Ls, client_grads)]
        # print(weights2)
        sum_weights_2 = sum(weights2)
        P = [ w / sum_weights_2 for w in weights2]
        return P

  

    
    
    def run(self, args):
        # init some variables
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.global_model.to(device)
        # train procedure
        P = [1.0/self.N for _ in range(self.N)]
        df = pd.DataFrame(columns=['step','acc', 'loss'])
        df.to_csv("./save_datasetname/28class_Log_mnist_90-10_dir_1234-{}-{}-{}-{}-lr{}-round{}-ep{}-drift{}-beta{}-seed{}-{}-csd{}-cluster{}.csv".format(args.client_num, args.sample_num, args.datasetname, args.optimizer, args.lr, args.round_num,\
                                                                        args.localepoch, args.client_drift, args.beta, args.seed_num, args.localbs, args.csd_importance, args.cluster_num), index=False)          
        local_ep_list = np.random.choice(range(args.localepoch,args.localepoch+1),size=len(self.sampled_clients))       
        vt = deepcopy(self.global_model.state_dict())

        # client_data_ratio = [0 for _ in range(self.N)]
        # sum_client_data_ratio = 0
        # for i in range(self.N):
        #     client_data_ratio[i] = len(self.sampled_clients[i])
        #     sum_client_data_ratio  += client_data_ratio[i]
        # ratio_p = [ n/sum_client_data_ratio for n in client_data_ratio]


        # y_t = {}
        # yi_t = []
        # for i in range(self.N):
        #     yi_t.append({})

        # model_dict1 = self.global_model.state_dict()
        # for k, v in model_dict1.items():
        #     y_t[k] = model_dict1[k].data * 0.0
        #     for i in range(self.N):
        #         yi_t[i][k] = model_dict1[k].data * 0.0
        # y_t_middle = copy.deepcopy(y_t)
     
        for t in range(self.T):

            self.set_weights()
            Ls = []
            
            # N_uniform_sample_index = np.random.randint(0, len(self.sampled_clients), self.N)
            # sample_index, N_index = self.sample_clients(args, P, N_uniform_sample_index)
            # random.seed(t)
            if args.optimizer == "FedAvg":
                # sample_index = np.random.randint(0, len(self.sampled_clients), self.S)
                sample_index_temp = random.sample( [i for i in range(len(self.sampled_clients))] , self.N)
                sample_index_2 = random.sample([i for i in range(len(sample_index_temp))], self.S)
                sample_index = [sample_index_temp[i] for i in sample_index_2]
                sample_index_temp_index = random.sample([i for i in range(self.N)], self.S)
            elif args.optimizer == "IS_prac_theory":
                sample_index_temp = random.sample([i for i in range(len(self.sampled_clients))], self.N)
                P = self.reweight_prac_theory(P, device, args, t, sample_index_temp)
                # print(P)
                sample_index_temp_index = torch.multinomial(torch.tensor(P), self.S, True)
                # sample_index_temp_index = list(map(P.index,heapq.nlargest(self.S,P)))
                sample_index = [sample_index_temp[i] for i in sample_index_temp_index]
                # print(sample_index)
            elif args.optimizer == "IS_theory":
                sample_index_temp = random.sample([i for i in range(len(self.sampled_clients))], self.N)
                P = self.reweight_theory(P, device, args, t, sample_index_temp)
                # sample_index_temp_index = list(map(P.index,heapq.nlargest(self.S,P)))
                # print(len(self.sampled_clients))
                # print(self.N)
                # print(P)
                sample_index_temp_index = torch.multinomial(torch.tensor(P), self.S, True)
                
                sample_index = [sample_index_temp[i] for i in sample_index_temp_index]
            elif args.optimizer == "cluster":
                # n_samples = np.array([len(self.sampled_clients[i]) for i in range(len(self.sampled_clients))])
                # weights = n_samples / np.sum(n_samples)
                # distri_clusters = get_clusters_with_alg1(5, weights)
                # sample_index = sample_clients_cluster(distri_clusters)
                # sample_index_temp_index = random.sample([i for i in range(self.N)], self.S)
                gradient_norm = self.client_gradient_norm( device, args, t )
                gradient_norm_sum = sum(gradient_norm)
                weights = [w/gradient_norm_sum for w in gradient_norm]
             
                distri_clusters = get_clusters_with_alg1(5, weights)
                sample_index = sample_clients_cluster(distri_clusters)
                sample_index_temp_index = random.sample([i for i in range(self.N)], self.S)
            elif args.optimizer == "Cluster_sampling":
                
                if t == 0:
                    cluster_center_index = random.sample(  [i for i in range(len(self.sampled_clients))] , args.cluster_num)
                    gradient_norm = self.client_gradient_norm( device, args, t )
                    cluster_center = [gradient_norm[i] for i in cluster_center_index ]
                cluster_list, cluster_list_index, cluster_center = self.get_cluster_index(args, cluster_center_index, device, t, cluster_center )
                # print(cluster_list)
                sample_index_temp = []
                sample_index = []
                sample_index_total = []
                P_cluster = []
                for i in range(args.cluster_num):
                    P_cluster.append([])
                    if len(cluster_list[i])==0:
                        P_cluster[i] = []
                    else:
                        P_cluster[i].append([1/len(cluster_list[i])])
                for i in range(len(cluster_list)):
                    sample_index_temp.append([])
                    sample_index.append([])
                for i in range(len(cluster_list)):
                    if P_cluster[i] == []:
                        sample_index_temp[i] = []
                        sample_index[i] = random.sample( [i for i in range(self.N)] , int(self.S/args.cluster_num))
                    else:
                        if sum(cluster_list[i]) ==0:
                            aa = [1/len(cluster_list[i]) for _ in range(len(cluster_list[i]))]
                            sample_index_temp[i] =torch.multinomial(torch.tensor(aa), int(self.S/args.cluster_num), replacement=True)
                        else:
                            print(cluster_list[i])
                            sample_index_temp[i] = torch.multinomial(torch.tensor(cluster_list[i]), int(self.S/args.cluster_num), replacement=True)
                        P_cluster[i] = self.reweight_theory_cluster(P_cluster[i], device, args, t, cluster_list_index[i])
                        sample_index[i] = [cluster_list_index[i][j] for j in sample_index_temp[i]]
                    
                    sample_index_total += sample_index[i]
      
            else:
                print("you have input a wrong optimizer!")
            # for i in range(self.N):
            #     sum_data_len +=  len(self.sampled_clients[sample_index_temp[i]])
                
            client_grads = []
            # client_loss = []
            # grads_mean_his = 0
            # pro = [P[i]/sum(P) for i in range(self.N)]
            # Grad_accumulator = [0 for _ in range(self.S)]
            for i in range(self.S):
                if args.optimizer == "Cluster_sampling":
                    data_set = self.sampled_clients[sample_index_total[i]]
                else:
                    data_set = self.sampled_clients[sample_index[i]]
                
                
                if args.localbs == "fullbs":
                    data_loader = dataloader.DataLoader(data_set, len(data_set), drop_last=False)
  
                else:
                    data_loader = dataloader.DataLoader(data_set, args.bs, True, drop_last=False, )
                    
                if args.optimizer == "Cluster_sampling":
                    self.models[i], L, grads = local_epoch(self.models[i].to(device), self.global_model, data_loader, device, args, t,  local_ep_list[sample_index_total[i]], data_set)
                else:
                    self.models[i], L, grads = local_epoch(self.models[i].to(device), self.global_model, data_loader, device, args, t,  local_ep_list[sample_index[i]], data_set)
                self.models[i].to("cpu")
                Ls.append(L)
                client_grads.append(grads)
                # client_loss.append(local_loss)

                # Grad_accumulator[i] =  sum([torch.norm(p) ** 2 for p in grads]) ** 0.5 /pro[sample_index[i]]**2
            
        
                     # grad_norm = 1/self.N**2 * 1/self.S**2 * sum(Grad_accumulator)
            # if args.optimizer == 'FedVARP':
            #     y_t, yi_t = self.combine_weights_VARP(args, Ls, [P[i] for i in sample_index], y_t, yi_t, sample_index) 
            # elif args.optimizer == "FedVARP_IS_prac" or "FedVARP_IS_weight":
            #     y_t, yi_t = self.combine_weights_VARP_IS(args, Ls, [P[i] for i in sample_index], y_t, yi_t, sample_index) 
            # else:  
            #     self.combine_weights(args, Ls, [P[i] for i in sample_index],vt)
            # y_t, yi_t = self.combine_weights_VARP(args, Ls, y_t, yi_t, sample_index) 
            # y_t, yi_t = self.combine_weights_VARP_IS_modify(args, Ls, [P[i] for i in sample_index], y_t, yi_t, sample_index,y_t_middle)  
            # y_t, yi_t = self.combine_weights_VARP(args, Ls,  y_t, yi_t, sample_index) 
            # self.combine_weights(args, Ls, [P[i] for i in sample_index],vt)
            # if args.optimizer == "IS" or "FedVARP_IS_weight":
            #     P = self.reweight(P, sample_index, client_grads, Ls)
            # elif args.optimizer == "IS_prac" or "FedVARP_IS_prac": 
            #     P = self.reweight_prac(P,sample_index, client_grads,Ls,t, args)
            # elif args.optimizer == "FedVARP" or "FedAvg":
            #     pass
            if args.optimizer == "Cluster_sampling":
                self.conbine_weights_cluster(args, Ls, sample_index_temp,  P_cluster)
            else:
                self.combine_weights(args, Ls, [P[i] for i in sample_index_temp_index],vt)
            
            # if args.optimizer == "IS_prac": 
            #     P = self.reweight_prac(P, N_index, client_grads,Ls,t, args)
            # else:
            #     P = self.reweight(P, N_index, client_grads, Ls)
                
            # P = self.reweight_theory(P, device, args, t, local_ep_list)   
            # P = self.reweight_prac(P,sample_index, client_grads,Ls,t, args)
           
            # P = self.reweight_prac_theory( P, device, args, t , local_ep_list)

            
            # test_loader = dataloader.DataLoader(self.test_loader, args.bs, shuffle = True, drop_last=True)
            
            acc, loss = self.test(self.test_loader, device) 


            results_list = [t, acc, loss.item()]  #, grad_norm
            results_data = pd.DataFrame([results_list])
            results_data.to_csv('./save_datasetname/28class_Log_mnist_90-10_dir_1234-{}-{}-{}-{}-lr{}-round{}-ep{}-drift{}-beta{}-seed{}-{}-csd{}-cluster{}.csv'.format(args.client_num, args.sample_num, args.datasetname, args.optimizer,args.lr,args.round_num,\
                                                                                        args.localepoch, args.client_drift, args.beta,args.seed_num, args.localbs, args.csd_importance, args.cluster_num), mode = 'a', header = False, index=False)
            print('acc: {}, loss: {} on round {}'.format(acc, loss, t))  #, grad_norm    , grad_norm: {}


    
    def save_model(self, args, t, k):
        parent_dir = Path(args.savepath)
        if not parent_dir.exists():
            parent_dir.mkdir()
        save_dir = Path(args.savepath + '{}-{}-{}-{}-{}-{}'.format(self.N, self.T, args.split_num, args.round_drift, args.client_drift, args.optimizer))
        if not save_dir.exists():
            save_dir.mkdir()
        file_name = '/{}-{}.pt'.format(t, k)
        torch.save(self.global_model, save_dir.as_posix() + file_name)


    def test(self, data, device):
        loss = 0
        total = 0
        correct = 0
        
        
        for (X, Y) in data:
            # print("Y=============",Y)
            # if args.datasetname == "cookup_train_1" or "cookup_train_2" or "cookup_train_3":
            #     X = X.unsqueeze(1)
            X, Y = X.to(device), Y.to(device)
            X, Y = Variable(X), Variable(Y)
            self.global_model.to(device)
            self.global_model.eval()
            with torch.no_grad(): 
                out = self.global_model(X)
            Y = Y.long()
            loss += F.cross_entropy(out, Y) * Y.size(0)
            _, predicted = torch.max(out.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
            self.global_model.to("cpu")
            X.to("cpu"), Y.to("cpu")
        return correct / total, loss / total

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR-10 baseline')
    parser.add_argument('--client_num','-cN', 
                    default=1000, 
                    type=int, 
                    help='the number of clients')
    parser.add_argument('--datasetname','-name', 
                    default="Cifar10", 
                    type=str, 
                    help='the name of dataset')
    parser.add_argument('--round_num','-rN', 
                    default=500, 
                    type=int, 
                    help='the number of rounds')
    parser.add_argument('--sample_num','-sN', 
                    default=30, 
                    type=int, 
                    help='the number of communication rounds')
    parser.add_argument('--client_drift','-cd', 
                    default=0.1, 
                    type=float, 
                    help='client drift')
    parser.add_argument('--lr', 
                    default=0.1, 
                    type=float, 
                    help='client learning rate')
    parser.add_argument('--bs', 
                    default=20,  #10
                    type=int, 
                    help='batch size on each worker/client')
    parser.add_argument('--NIID',
                    default=True,
                    action='store_true',
                    help='whether the dataset is non-iid or not')
    parser.add_argument('--datapath',
                    default='./data/',
                    type=str,
                    help='directory to load data')
    parser.add_argument('--savepath',
                    default='./results/checkpoints/',
                    type=str,
                    help='directory to save results')
    parser.add_argument('--optimizer',
                    default='FedAvg',
                    type=str,
                    help='type of optimizer')
    parser.add_argument('--globallr',
                    default=1.0, 
                    type=float,
                    help='global learning rate')
    parser.add_argument('--localepoch',
                    default=5,
                    type=int,
                    help='number of local epochs')
    parser.add_argument('--seed_num','-seed',
                    default=1234,
                    type=int,
                    help='number of seed')
    parser.add_argument('--beta',
                     default=0.5,
                    type=float,
                     help='contral the weight of gradient norm')
    parser.add_argument('--localbs',
                    default='args_bs',
                    type=str,
                    help='local dataset size')
    parser.add_argument('--csd_importance',
                    type=float, 
                    default=0)
    parser.add_argument('--cluster_num',
                    default= 5,
                    type=int
                    )
    args = parser.parse_args()
    args = parser.parse_args()

    torch.cuda.set_device(3)
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    fedTrain = FedTrain(args)
    fedTrain.run(args)