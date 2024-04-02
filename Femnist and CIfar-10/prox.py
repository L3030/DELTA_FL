import pandas as pd
import copy
from torch.nn import parameter
from torch.optim import optimizer
from cNNmodel import   Net, Logistic, CNNNet
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
from pro_comm_helpers import combine_reduction, fedAvg_communicate, IS_communicate
from hessian_tools import combine_hessian, get_SI_omega, get_diag_fisher, get_diag_hessian, copy_hessian
import os
from pathlib import Path

def schedule_lr(t, args):
    # args.lr = args.lr/(t+1)**0.5
    # if t >= 0.75 * args.round_num:
    #     return args.lr * 0.01
    # if t >= 0.5 * args.round_num:
    #     return args.lr * 0.1
    
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


def local_epoch(model, global_model, data, device, args, t , local_ep):
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = choose_optimizer(model, args, t)
    gradients = [torch.zeros_like(p) for p in model.parameters()]
    count = 0
    new_mu = dict()
    global_dict = global_model.state_dict()
    for name, param in model.named_parameters():
            new_mu[name] = deepcopy(global_dict[name])
    for k in range(local_ep):
        # loss_record = []
        for (X, Y) in data:
            
            count += 1
            X, Y = X.to(device), Y.to(device)
            X, Y = Variable(X), Variable(Y)
            model.zero_grad()
            output = model(X)
            ce_loss = criterion(output, Y)
            csd_loss = get_csd_loss(model, new_mu) if args.csd_importance > 0 else 0
            loss = ce_loss + args.csd_importance * csd_loss
            loss.backward()
            # loss_record.append(loss)
            gradients = [g + p.grad.clone().detach() if p.grad is not None else g for g, p in zip(gradients, model.parameters())]
            optimizer.step()
    
    # losses = sum(loss_record)/len(loss_record)   # final loss
    gradients = [g / count for g in gradients]

    return model, len(data), gradients



class FedTrain():

    def __init__(self, args) -> None:
        # useful parameters
        self.N= args.client_num
        self.T = args.round_num
        self.S= args.sample_num
        self.global_lr = args.globallr
        # generate data
        self.new_partitioner = SplitDataset(args)
        self.total_train_sets = self.new_partitioner.trainset
        # self.train_sets = self.new_partitioner.client_sets

        self.sampled_clients = self.new_partitioner.sampled_clients

        self.test_loader = self.new_partitioner.test_loader
        # initial global model and local models
        self.global_model, self.models = self.init_models()

    def init_models(self):
        # load finetune models
        local_models = [models.resnet18(pretrained=True) for i in range(self.S)]
        global_model = models.resnet18(pretrained=True)
        # change output layer
        num_ftrs = global_model.fc.in_features
        global_model.fc = nn.Linear(num_ftrs, 10)
        for i in range(self.S):
            local_models[i].fc = nn.Linear(num_ftrs, 10)
        # local_models = [Net() for i in range(self.S)]
        # global_model = Net()


        return global_model, local_models
    
    def set_weights(self):
        # transmit parameters to local models

        global_dict = self.global_model.state_dict()

        for i in range(len(self.models)):
            self.models[i].load_state_dict(deepcopy(global_dict))
        # return global_dict
    
    def combine_weights(self, args, Ls, P,vt):
        # combine local parameters for global model
        if args.optimizer == 'FedAvg':
            self.global_model = fedAvg_communicate(self.global_model, self.models, args, Ls,vt)
        else:
            self.global_model = IS_communicate(self.global_model, self.models, args, P, self.N, Ls,vt)
        return 

    def sample_clients(self, args, P):
        if args.optimizer == 'FedAvg':
            return np.random.randint(0, self.N, self.S)
        else:
            return torch.multinomial(torch.tensor(P), self.S, False)

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
        # if t <0.5*args.round_num:
        #     beta = 0.0
        # else:
        #     beta = args.beta
        # beta = t/args.round_num
        beta = args.beta
        # beta = 1 - 1/(t+1)**(1/100)
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
        # c=[c[i]+grads[i] for i in range(len(grads[0]))]
        
        # if t == 0:
        #     grads_mean_his = sum(g for g in grads)/len(grads) 
        
        # grads_mean = sum(g for g in grads)/len(grads) 
        # grads_mean_his = (grads_mean + grads_mean_his )/2
        # variance_l1norm = [ i - grads_mean_his for i in grads   ]
        # sum_L = sum(Ls)
        weights2 = [ L / sum_L * sum([torch.norm(p) ** 2 for p in g]) ** 0.5 for L, g in zip(Ls, grads)]
        # weights2 = [L / sum_L * sum([torch.norm(p) ** 2 for p in g]) ** 0.5 for L, g in zip(Ls, variance_l1norm)]
        # print(weights2)
        # print(sum(weights2))
        weights =[(1-beta)* a + beta*b for a,b in zip(weights1,weights2) ] 
        sum_weights = sum(weights)
        sum_old_weights = sum([P[i] for i in sample_index])
        weights = [ w / sum_weights * sum_old_weights for w in weights]
        for i in range(len(weights)):
            P[sample_index[i]] = alpha * P[sample_index[i]] + (1 - alpha) * weights[i]
        
        return P

    def reweight_theory(self, P, device, args, t , local_ep_list):
        client_grads = []
        # client_loss = []
        sample_index = [i for i in range(self.N)]
        Ls = []

        for i in range(self.N):
            model = deepcopy(self.global_model)
            data_set = self.sampled_clients[i]
            data_loader = dataloader.DataLoader(data_set, args.bs, True, drop_last=True)
            model, L, grads = local_epoch(model.to(device),self.global_model, data_loader, device, args, t , local_ep_list[i])
            client_grads.append(grads)
            # client_loss.append(local_loss)
            Ls.append(L)

        P = self.reweight(P, sample_index, client_grads, Ls)
        return P
    
    def reweight_prac_theory(self, P, device, args, t , local_ep_list):
        client_grads = []
        # client_loss = []
        sample_index = [i for i in range(self.N)]
        Ls = []

        for i in range(self.N):
            model = deepcopy(self.global_model)
            data_set = self.sampled_clients[i]
            data_loader = dataloader.DataLoader(data_set, args.bs, True, drop_last=True)
            model, L, grads = local_epoch(model.to(device),self.global_model, data_loader, device, args, t , local_ep_list[i])
            client_grads.append(grads)
            # client_loss.append(local_loss)
            Ls.append(L)

        P = self.reweight_prac(P, sample_index, client_grads, Ls,t, args)
        return P

#     def reweight_cos(self, P, sample_index, grads, Ls, grads_record):
#         alpha = 0.4
#         similarity = []
#         sum_L = sum(Ls)

#         weights = [ L / sum_L * sum([torch.norm(p) ** 2 for p in g[len(g) - 2:len(g)]]) ** 0.5 for L, g in zip(Ls, grads)]

        
#         grads_record = grads_record + grads
#         grads_sum = [0 for _ in range(len(grads[0]))]
#         for i in range(len(grads_record)):
#             for j in range(len(grads[0])):
#                 grads_sum[j] += grads_record[i][j]
#         grads_true = [i/len(grads_record) for i in grads_sum]


#         for i in range(len(grads)):
#             similarity_tensor = [0 for _ in range(len(grads[0])) ]
#             for j in range(len(grads[i])):
#                 similarity_tensor[j] = torch.cosine_similarity(grads[i][j],grads_true[j], dim=0)+1

#             for p in range(len(similarity_tensor)):
#                 similarity_tensor[p] = torch.mean(similarity_tensor[p])
#             similarity.append(torch.mean(torch.stack(similarity_tensor)))
#         p_similarity = [i/sum(similarity) for i in similarity]
#         sum_p_similarity = sum(p_similarity)
#         sum_weights = sum(weights)
#         sum_old_weights = sum([P[i] for i in sample_index])
#         weights = [( alpha * s/sum_p_similarity + (1-alpha)*w/ sum_weights )* sum_old_weights for s, w in zip(p_similarity, weights)]

#         for i in range(len(weights)):
#             P[sample_index[i]] =  weights[i]
#         return P, grads_record
  

    
    
    def run(self, args):
        # init some variables
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model.to(device)
        # train procedure
        P = [1.0/self.N for _ in range(self.N)]
        df = pd.DataFrame(columns=['step','acc', 'loss'])
        df.to_csv("./save/new-{}-{}-{}-{}-lr{}-round{}-ep{}-drift{}-beta{}-seed{}-{}-csd{}.csv".format(args.client_num, args.sample_num, args.datasetname, args.optimizer, args.lr, args.round_num,\
                                                                        args.localepoch, args.client_drift, args.beta, args.seed_num, args.localbs, args.csd_importance), index=False)          
        local_ep_list = np.random.choice(range(args.localepoch,args.localepoch+1),size=self.N)       
        vt = deepcopy(self.global_model.state_dict())

        for t in range(self.T):

            self.set_weights()
            Ls = []
            sample_index = self.sample_clients(args, P)
            client_grads = []
            # client_loss = []
            # grads_mean_his = 0
            # pro = [P[i]/sum(P) for i in range(self.N)]
            # Grad_accumulator = [0 for _ in range(self.S)]
            for i in range(self.S):
                # data_set = self.train_sets[sample_index[i]]
                data_set = self.sampled_clients[sample_index[i]]
                
                # data_loader = dataloader.DataLoader(data_set, len(data_set), drop_last=True)
                data_loader = dataloader.DataLoader(data_set, args.bs, True, drop_last=True)

                self.models[i], L, grads = local_epoch(self.models[i].to(device), self.global_model, data_loader, device, args, t,  local_ep_list[sample_index[i]])
                self.models[i].to("cpu")
                Ls.append(L)
                client_grads.append(grads)
                # client_loss.append(local_loss)

                # Grad_accumulator[i] =  sum([torch.norm(p) ** 2 for p in grads]) ** 0.5 /pro[sample_index[i]]**2
            
        
            # grad_norm = 1/self.N**2 * 1/self.S**2 * sum(Grad_accumulator)

                
            self.combine_weights(args, Ls, [P[i] for i in sample_index],vt)
            
            P = self.reweight(P, sample_index, client_grads, Ls)
            # P = self.reweight_theory(P, device, args, t, local_ep_list)   
            # P = self.reweight_prac(P,sample_index, client_grads,Ls,t, args)
            # P = self.reweight_prac_theory( P, device, args, t , local_ep_list)

            #             P, grads_record = self.reweight_cos(P, sample_index, client_grads, Ls, grads_record)
            #             P ,weights_all = self.reweight_history(P, sample_index, client_grads, Ls, weights_all)

            acc, loss = self.test(self.test_loader, device) 


            results_list = [t, acc, loss]  #, grad_norm
            results_data = pd.DataFrame([results_list])
            results_data.to_csv('./save/new-{}-{}-{}-{}-lr{}-round{}-ep{}-drift{}-beta{}-seed{}-{}-csd{}.csv'.format(args.client_num, args.sample_num, args.datasetname, args.optimizer,args.lr,args.round_num,\
                                                                                        args.localepoch, args.client_drift, args.beta,args.seed_num, args.localbs, args.csd_importance), mode = 'a', header = False, index=False)
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
    parser.add_argument('--seed_num',
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
    args = parser.parse_args()

    torch.cuda.set_device(0)
    seed = args.seed_num
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    fedTrain = FedTrain(args)
    fedTrain.run(args)