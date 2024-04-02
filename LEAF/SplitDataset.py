from copy import deepcopy
import os
import random
import numpy as np
import time
import argparse
# import logging

from random import Random
from numpy.core.fromnumeric import sort
from numpy.core.numeric import Inf

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Process
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torch.distributions.dirichlet as dirichlet
import matplotlib.pyplot as plt
# import scipy.stats



class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index
        # self.reformulate()
        self.targets = [self.data.targets[i] for i in self.index]
        self.labels = {}
        for i in self.index:
            if self.data[i][1] in self.labels:
                self.labels[self.data[i][1]] += 1.0 / len(self.index)
            else:
                self.labels[self.data[i][1]] = 1.0 / len(self.index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

    def reformulate(self):
        while isinstance(self.data, Partition):
            print(self.index, self.data.index)
            self.data, self.index = self.data.data, [self.data.index[i] for i in self.index]
            print(self.index)


    def subset(self, index):
        new_index = [self.index[i] for i in index]
        new_set = Partition(self.data, new_index)
        return new_set

    def combine(self, new_set):
        new_index = self.index + new_set.index
        return Partition(self.data, new_index)

    def naive_core_set(self, number):
        index = list(np.random.randint(0, len(self.index), min(len(self.index), number)))
        return self.subset(index)

    def latest_core_set(self, number):
        index = self.index[len(self.index) - number:len(self.index)]
        return self.subset(index)


    def weighted_core_set(self, number):
        number = min(len(self.index), number)
        P = torch.tensor([i + 1.0 for i in range(len(self.index))])
        index = list(torch.multinomial(P, number))
        return self.subset(index)


class DataPartition(object):

    def __init__(self, dataset, args):
        self.dataset = dataset
        self.sampled_clients = self.partition(dataset, args.client_num, args.client_drift)

    def divide_by_label(self, dataset, class_num = 10):
        index_map = [[] for i in range(class_num)]
        len_map = [0 for _ in range(class_num)]
        for i in range(len(dataset)):
            index_map[dataset[i][1]].append(i)
            len_map[dataset[i][1]] += 1
        return index_map, len_map

    def reweight(self, q, empty_class):
        sum_q = sum(q)
        q[empty_class] = 0
        q = q / sum_q
        return q


    def partition(self, dataset, N, alpha, class_num = 10):
        S, len_S = self.divide_by_label(dataset, class_num)
        sampled_clients = [[] for _ in range(N)]
        M = len(dataset) // N
        for i in range(N):
            p = torch.tensor(len_S) / sum(len_S)
            q = dirichlet.Dirichlet(alpha * p).sample()
            print(q, i)
            while(len(sampled_clients[i]) < M):
                sample_index = torch.multinomial(q, 1)
                if len_S[sample_index] > 0:
                    sampled_clients[i].append(S[sample_index][0])
                    S[sample_index].pop(0)
                    len_S[sample_index] -= 1
                    if len_S[sample_index] == 0:
                        q = self.reweight(q, sample_index)
            print(sort(sampled_clients[i])[1:100])
            sampled_clients[i] = Partition(dataset, sampled_clients[i])
        return sampled_clients






class SplitDataset():

    def __init__(self, args) -> None:
        self.client_num = args.client_num
        self.client_drift = args.client_drift

        if args.datasetname == 'femnist':
            self.trainset, self.test_loader = self.load_femnist_data(args)
        elif args.datasetname == 'Cifar10':
            self.trainset, self.test_loader = self.load_data(args)
        else:
            raise ValueError('Not valid dataset')
        if os.path.exists('./data/{}-{}-{}.pt'.format(args.datasetname, args.client_num, args.client_drift)):
            self.sampled_clients = torch.load('./data/{}-{}-{}.pt'.format(args.datasetname, args.client_num, args.client_drift))
        else:
            data_partition = DataPartition(self.trainset, args)
            self.sampled_clients = data_partition.sampled_clients
            torch.save(self.sampled_clients, './data/{}-{}-{}.pt'.format(args.datasetname, args.client_num, args.client_drift))
        self.draw_distributions(self.sampled_clients)

    def get_mean_std(self, lst):
        mean = sum(lst) / len(lst)
        std = sum([abs(i - mean) ** 2 for i in lst]) ** 0.5
        print(mean, std)

    
    def get_distance_cur(self, pics):
        distances = []
        for i in range(1, len(pics)):
            p1 = torch.tensor([p / sum(pics[i-1]) for p in pics[i-1]])
            p2 = torch.tensor([p / sum(pics[i]) for p in pics[i]])
            # KL = scipy.stats.entropy(p1, p2)
            KL = torch.norm(p1 - p2, 2)
            if KL != Inf:
                distances.append(KL)
        return distances


    
    def simulate_stateless(self, dataset, args):
        pass

    def divide_by_label(self, dataset, class_num = 10, outlier = None):
        index_map = [[] for i in range(class_num)]
        len_map = [0 for _ in range(class_num)]
        for i in range(len(dataset)):
            if outlier is not None and i in outlier.index:
                continue 
            index_map[dataset[i][1]].append(i)
            len_map[dataset[i][1]] += 1
        return index_map, len_map

    def reweight(self, q, empty_class):
        sum_q = sum(q)
        q[empty_class] = 0
        q = q / sum_q
        return q

    def draw_samples(self, dataset, alpha, M, class_num = 10, outlier = None):
        S, len_S = self.divide_by_label(dataset, class_num, outlier)
        sampled_clients = []
        p = torch.tensor(len_S) / sum(len_S)
        q = dirichlet.Dirichlet(alpha * p).sample()
        while(len(sampled_clients) < M):
            sample_index = torch.multinomial(q, 1)
            if len_S[sample_index] > 0:
                i = random.randint(0, len(S[sample_index]) - 1)
                sampled_clients.append(S[sample_index][i])
                S[sample_index].pop(i)
                len_S[sample_index] -= 1
                if len_S[sample_index] == 0:
                    q = self.reweight(q, sample_index)
        sampled_clients = Partition(dataset, sampled_clients[i])
        print(sampled_clients)
        return sampled_clients

    def get_distribution_pics(self, dataset, class_num = 10):
        pic = [0 for i in range(class_num)]
        for k, v in dataset.labels.items():
            pic[k] = v
        return pic
        

    def simulation_overlap(self, client, args, type='Naive'):
        M = len(client) // args.round_num
        R = int(M * args.step)
        subset = self.draw_samples(client, args.round_drift, M)
        pictures = []
        subset = Partition(client, subset)
        pictures.append(self.get_distribution_pics(subset))
        for t in range(30):
            residual = Partition(client, self.draw_samples(client, args.round_drift, R, 10, subset))
            if type == 'Naive':
                subset = subset.naive_core_set(M - R)
            else:
                subset = subset.latest_core_set(M - R)
            subset = subset.combine(residual)
            pictures.append(self.get_distribution_pics(subset))
        # plt.imshow(pictures)
        # plt.show()
        return pictures





    @classmethod
    def empty_set(cls, train_sets):
        return Partition(train_sets, [])

    def draw_distributions(self, sampled_clients, class_num = 10):
        picture = [[0 for _ in range(class_num)] for _ in range(len(sampled_clients))]
        for i, client in enumerate(sampled_clients):
            for k, v in client.labels.items():
                picture[i][k] = v
        # print(picture)
        # plt.imshow(picture)
        # plt.show()
        

    def load_mnist(self, args):
        trainset = torchvision.datasets.MNIST(
            root=args.datapath, train=True, transform=torchvision.transforms.ToTensor(), download=True
        )
        testset = torchvision.datasets.MNIST(
            root=args.datapath, train=False, transform=torchvision.transforms.ToTensor(), download=True
        )
        test_loader = torch.utils.data.DataLoader(testset, 
                                            batch_size=64, 
                                            shuffle=False)
        return trainset, test_loader
    


    def load_cifar100_data(self, args):
        CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
        CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TEST_MEAN, CIFAR100_TEST_STD),
        ])
        trainset = torchvision.datasets.CIFAR100(root=args.datapath, 
                                            train=True, 
                                            download=True,
                                            transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=args.datapath, 
                                            train=False, 
                                            download=True,
                                            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, 
                                            batch_size=64, 
                                            shuffle=False)
        return trainset, test_loader


    def load_femnist_data(self, args):
        trainset = torchvision.datasets.FashionMNIST(root=args.datapath, 
                                            train=True, 
                                            download=True,
                                            transform=transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081))]))
        testset = torchvision.datasets.FashionMNIST(root=args.datapath, 
                                            train=False, 
                                            download=True,
                                            transform=transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081))]))
        test_loader = torch.utils.data.DataLoader(testset, 
                                            batch_size=64, 
                                            shuffle=False)
        return trainset, test_loader
        




    def load_data(self, args):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root=args.datapath, 
                                            train=True, 
                                            download=True, 
                                            transform=transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root=args.datapath, 
                                        train=False, 
                                        download=True, 
                                        transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, 
                                            batch_size=64, 
                                            shuffle=False)
        return trainset, test_loader
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR-10 baseline')
    parser.add_argument('--client_num','-cN', 
                    default=10, 
                    type=int, 
                    help='the number of clients')
    parser.add_argument('--round_num','-rN', 
                    default=10, 
                    type=int, 
                    help='the number of rounds')
    parser.add_argument('--client_drift','-cd', 
                    default=1, 
                    type=float, 
                    help='the drift among clients')
    parser.add_argument('--step', 
                    default=1, 
                    type=float, 
                    help='the level of overlap')    
    parser.add_argument('--round_drift','-rd', 
                    default=100, 
                    type=float, 
                    help='the drift among rounds')                            
    parser.add_argument('--datapath', 
                    default='./data', 
                    type=str, 
                    help='the data path')
    args = parser.parse_args()
    splited = SplitDataset(args)        

