import pandas as pd
import copy
import math
import time
from torch.nn import parameter
from torch import optim
from torch.optim import optimizer
from cNNmodel import   Net, Logistic, CNNNet, Net_celeba, CNN_chatgpt
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
from comm_helpers import combine_reduction, fedAvg_communicate, IS_communicate, Avg_communicate, FedVARP_communicate,FedVARP_IS_communicate,FedVARP_communicate_modify
# from pyhessian import hessian # Hessian computation
from hessian_tools import combine_hessian, get_SI_omega, get_diag_fisher, get_diag_hessian, copy_hessian
import os
from pathlib import Path
from py_func.clustering import sample_clients_cluster, get_clusters_with_alg1
import model_utils
import model_utils_celeba
def schedule_lr(t, args):

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

def feddecorr_loss( z):
        N,d = z.shape
        if N==1:
            pass
        else:
            z = (z - z.mean(0)) / (z.std(0) + 1e-4)
        corr_mat = 1/N*torch.matmul(z.t(), z)
        loss_fed_decorr = (corr_mat.pow(2)).mean()
        return loss_fed_decorr
    

def local_epoch(model, global_model, data, device, args, t , local_ep, data_set):
    model.train()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=schedule_lr(t, args), momentum=0.9)
    gradients = [torch.zeros_like(p) for p in model.parameters()]
    count = 0
    for k in range(local_ep):
        # loss_record = []
        for (X, Y) in data:
            
            count += 1
            X, Y = X.to(device), Y.to(device)
            X, Y = Variable(X), Variable(Y)
            model.zero_grad() 
            z = model.feature(X)
            loss_proximal = 0
            for pm, ps in zip(model.parameters(), global_model.parameters()):
                loss_proximal += torch.sum(torch.pow(pm - ps, 2))
            output = model.classifier(z)
            ce_loss = criterion(output, Y)
            loss = ce_loss +  0.5*args.br* loss_proximal
            loss.backward()
            
            gradients = [g + p.grad.clone().detach() if p.grad is not None else g for g, p in zip(gradients, model.parameters())]
            optimizer.step()
    if count>0:
        gradients = [g / count for g in gradients]
    return model, len(data_set), gradients




class FedTrain():

    def __init__(self, args) -> None:
        # useful parameters
        self.N= args.client_num
        self.T = args.round_num
        self.S= args.sample_num
        self.global_lr = args.globallr
        train_clients, train_groups, train_data, test_data = model_utils.read_data('./leaf/data/femnist/data/train','./leaf/data/femnist/data/test')
        self.N = len(train_clients)
        # print('N=======', self.N)
        self.sampled_clients = train_data
        self.train_clients = train_clients
        # print(self.sampled_clients[train_clients[0]])
        self.test_loader = test_data
        # initial global model and local models
        self.global_model, self.models = self.init_models()
    def init_models(self):
        local_models = [Net() for i in range(self.S)]
        global_model = Net()


        return global_model, local_models
    
    def set_weights(self):
        # transmit parameters to local models

        global_dict = self.global_model.state_dict()

        for i in range(len(self.models)):
            self.models[i].load_state_dict(deepcopy(global_dict))
    
    def combine_weights(self, args, Ls, P,vt):
        # combine local parameters for global model
        if args.optimizer == 'FedAvg':
            self.global_model= fedAvg_communicate(self.global_model, self.models, args, Ls)
        elif args.optimizer == 'MD':
            self.global_model = Avg_communicate(self.global_model, self.models, args, Ls,vt)
        else:
            self.global_model = IS_communicate(self.global_model, self.models, args, P, self.N, Ls,vt,self.S)
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

    def sample_clients(self, args, P):
        if args.optimizer == "FedAvg":
            return np.random.randint(0, self.N, self.S)
        else:
            return torch.multinomial(torch.tensor(P), self.S, False)

   

    def client_gradient_norm(self, device, args, t):
        client_grads = []
        sample_index = [i for i in range(self.N)]
        Ls = []

        for i in range(self.N):
            model = deepcopy(self.global_model)
            data_set = self.sampled_clients[self.train_clients[sample_index[i]]]
            data_loader = model_utils.batch_data(data_set,args.bs, seed)
            model, L, grads = local_epoch(model.to(device),self.global_model, data_loader, device, args, t , local_ep=1, data_set=data_set )
            client_grads.append(grads)
            Ls.append(L)
        sum_L = sum(Ls)
        weights = [ ( sum([torch.norm(p) ** 2 for p in g]) ** 0.5) for  g in  client_grads]
        
        weights = [s.to("cpu") for s in weights]
        return weights

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
        probab_before = [sum([torch.norm(p) ** 2 for p in g]) ** 0.5 for g in  grads]

        c = [0]*len(grads[0])
        for i in range(len(grads)):
            for j in range(len(c)):
                c[j] += grads[i][j]
                for k in range(len(grads[i][j])):
                    if math.isnan(k):
                        print('something wrong here.')

        c = [i/len(grads) for i in c]
        sum_L = sum(Ls)

        for i in range(len(grads)):
            for j in range(len(c)):
                grads[i][j] = grads[i][j] - c[j]
        probab = [sum([torch.norm(p) ** 2 for p in g]) ** 0.5 for g in  grads]
        weights2 = [ L / sum_L * sum([torch.norm(p) ** 2 for p in g]) ** 0.5 for L, g in zip(Ls, grads)]
        weights = [b for b in weights2] 
        sum_weights = sum(weights)
        sum_old_weights = sum([P[i] for i in sample_index])
        weights = [ w / sum_weights * sum_old_weights for w in weights]
        for i in range(len(weights)):
            P[sample_index[i]] = weights[i]
        
        return P

    def reweight_theory(self, P, device, args, t , sample_index):
        client_grads = []
        Ls = []

        for i in range(self.N):
            model = deepcopy(self.global_model)
            data_set = self.sampled_clients[self.train_clients[sample_index[i]]]
            data_loader = model_utils.batch_data(data_set,args.bs, seed)
            model, L, grads = local_epoch(model.to(device),self.global_model, data_loader, device, args, t , local_ep=1, data_set=data_set)
            client_grads.append(grads)
            Ls.append(L)
        sum_L = sum(Ls)
        weights = [ ( L / sum_L * sum([torch.norm(p) ** 2 for p in g]) ** 0.5) for L, g in zip(Ls, client_grads)]
        
        sum_weights = sum(weights)
        
        P = [ w / sum_weights for w in weights]
        
        
        return P
    
   
    def reweight_prac_theory(self, P, device, args, t ,sample_index):
        
        client_grads = []
        Ls = []

        for i in range(self.N):
            model = deepcopy(self.global_model)
            data_set = self.sampled_clients[self.train_clients[sample_index[i]]]
            data_loader = model_utils.batch_data(data_set,args.bs, seed)
            model, L, grads = local_epoch(model.to(device),self.global_model, data_loader, device, args, t , local_ep=1, data_set=data_set)
            client_grads.append(grads)

            Ls.append(L)
        sum_L = sum(Ls)
        c = [0]*len(client_grads[0])
        for i in range(len(client_grads)):
            for j in range(len(c)):

                c[j] += client_grads[i][j]
        
        c = [i/len(client_grads) for i in c]
        for i in range(len(client_grads)):
            for j in range(len(c)):
                client_grads[i][j] = client_grads[i][j] - c[j]
        weights2 = [ (args.beta*L / sum_L * sum([torch.norm(p) ** 2 for p in g]) ** 0.5 + (1-args.beta)*(L / sum_L)**0.5) for L, g in zip(Ls, client_grads)]
        sum_weights_2 = sum(weights2)
        P = [ w / sum_weights_2 for w in weights2]
        return P

  

    
    
    def run(self, args):
        start = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model.to(device)
        # train procedure
        P = [1.0/self.N for _ in range(self.N)]
        df = pd.DataFrame(columns=['step','acc', 'loss', 'time'])
        df.to_csv("./save/curve_leafFEMNIST_prox2-{}-{}-{}-{}-lr{}-round{}-ep{}-drift{}-beta{}-seed{}-{}-csd{}-br{}.csv".format(args.client_num, args.sample_num, args.datasetname, args.optimizer, args.lr, args.round_num,\
                                                                        args.localepoch, args.client_drift, args.beta, args.seed_num, args.localbs, args.csd_importance,args.br), index=False)          
        local_ep_list = np.random.choice(range(args.localepoch,args.localepoch+1),size=self.N)       
        vt = deepcopy(self.global_model.state_dict())
        for k in vt.keys():
            vt[k].zero_()

        
        
        for t in range(self.T):

            self.set_weights()
            Ls = []
            acc_ave = 0
            loss_ave =0
   
            sample_index = self.sample_clients(args, P)
            if args.optimizer == "FedAvg":
                sample_index_temp = random.sample( [i for i in range(len(self.sampled_clients))] , self.N)
                sample_index_2 = random.sample([i for i in range(len(sample_index_temp))], self.S)
                sample_index = [sample_index_temp[i] for i in sample_index_2]
            elif args.optimizer == "DELTA":
                sample_index_temp = random.sample([i for i in range(len(self.sampled_clients))], self.N)
                P = self.reweight_prac_theory(P, device, args, t, sample_index_temp)
                sample_index_temp_index = torch.multinomial(torch.tensor(P), self.S, False)
                sample_index = [sample_index_temp[i] for i in sample_index_temp_index]
            elif args.optimizer == "IS":
                sample_index_temp = random.sample([i for i in range(len(self.sampled_clients))], self.N)
                P = self.reweight_theory(P, device, args, t, sample_index_temp)
                sample_index_temp_index = torch.multinomial(torch.tensor(P), self.S, False)
                sample_index = [sample_index_temp[i] for i in sample_index_temp_index]
            else:
                gradient_norm = self.client_gradient_norm( device, args, t )
                gradient_norm_sum = sum(gradient_norm)
                P = weights = [w/gradient_norm_sum for w in gradient_norm]
                distri_clusters = get_clusters_with_alg1(5, weights)  
                sample_index = sample_clients_cluster(distri_clusters)

            client_grads = []
            for i in range(self.S):
                data_set = self.sampled_clients[self.train_clients[sample_index[i]]]
                if args.localbs == "fullbs":
                    data_loader = dataloader.DataLoader(data_set, len(data_set), drop_last=True)
                else:
                    data_loader = model_utils.batch_data(data_set,args.bs, seed)
               
                self.models[i], L, grads = local_epoch(self.models[i].to(device), self.global_model, data_loader, device, args, t,  local_ep_list[sample_index[i]], data_set)
                self.models[i].to("cpu")
                Ls.append(L)
                client_grads.append(grads)
               
            self.combine_weights(args, Ls, [P[i] for i in sample_index],vt)
           
            end = time.time()
            runningtime = end - start
            for i in range(self.N):   
                test_dataset = self.test_loader[self.train_clients[i]]
                testdata_loader = model_utils.batch_data(test_dataset,args.bs, seed)
                acc, loss = self.test(testdata_loader, device) 
                acc_ave = acc_ave + acc
                loss_ave = loss_ave + loss

            acc = acc_ave/self.N
            loss = loss_ave/self.N


            results_list = [t, acc, loss, runningtime]  #, grad_norm
            results_data = pd.DataFrame([results_list])
            results_data.to_csv('./save/curve_leafFEMNIST_prox2-{}-{}-{}-{}-lr{}-round{}-ep{}-drift{}-beta{}-seed{}-{}-csd{}-br{}.csv'.format(args.client_num, args.sample_num, args.datasetname, args.optimizer,args.lr,args.round_num,\
                                                                                        args.localepoch, args.client_drift, args.beta,args.seed_num, args.localbs, args.csd_importance, args.br), mode = 'a', header = False, index=False)
            print('acc: {}, loss: {} on round {}, training time: {}'.format(acc, loss, t, end-start))  #, grad_norm    , grad_norm: {}


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
        self.global_model.to(device)
        for (X, Y) in data:
            X, Y = X.to(device), Y.to(device)
            X, Y = Variable(X), Variable(Y)
            self.global_model.eval()

            out = self.global_model(X)
            loss += F.cross_entropy(out, Y) * Y.size(0)
            _, predicted = torch.max(out.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
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
    parser.add_argument('--br',
                     default=0.1,
                    type=float,
                     help='contral the weight of br loss')
    parser.add_argument('--localbs',
                    default='args_bs',
                    type=str,
                    help='local dataset size')
    parser.add_argument('--csd_importance',
                    type=float, 
                    default=0)
    args = parser.parse_args()

    torch.cuda.set_device(0)
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
    