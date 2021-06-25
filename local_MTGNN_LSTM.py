import torch
import numpy as np
import torch.nn.functional as F
import torch.nn
from torch.autograd import Variable
import scipy
from data_loader import data_test
from data_loader import data_train
from sklearn.metrics import roc_auc_score
import pickle
import os.path
from scipy import io
import sys

use_cuda = torch.cuda.is_available()

class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, example, bias=True):
        super(E2EBlock, self).__init__()
        self.d = example.size(3)
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


class multitask_dynamic_lstm(torch.nn.Module):
    def __init__(self, example, num_classes=10):
        super(multitask_dynamic_lstm, self).__init__()
        self.in_planes = example.size(1)
        self.d = example.size(3)
        self.e2econv1 = E2EBlock(1, 45, example)
        self.E2N = torch.nn.Conv2d(45, 45, (1, self.d))
        self.N2G = torch.nn.Conv1d(45, 45, (self.d, 1))

        self.fc1 = torch.nn.Linear(45, 100)
        self.fc2 = torch.nn.Linear(100, 50)

        self.fc3_Fi = torch.nn.Linear(50, 3)
        self.fc3_Fo = torch.nn.Linear(50, 3)
        self.fc3_T = torch.nn.Linear(50, 3)
        self.fc3_L = torch.nn.Linear(50, 3)

        self.lstm = torch.nn.LSTM(45, 2, 2)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):

        LSTM_in = torch.empty(1,21,45)
        FC_Fi = torch.empty(384,3,21)
        FC_Fo = torch.empty(384, 3, 21)
        FC_T = torch.empty(384, 3, 21)
        FC_L = torch.empty(384, 3, 21)
        for i in range(21):
            map = x[:,:,:,:,i]

            out = self.e2econv1(map)
            out = F.leaky_relu(out, negative_slope=0.1)
            out = self.E2N(out)
            out = F.leaky_relu(out, negative_slope=0.1)
            out_FC = out.view(out.size(2),out.size(1))
            out_FC = self.fc1(out_FC)
            out_FC = F.leaky_relu(out_FC,negative_slope=0.1)
            out_FC = self.fc2(out_FC)
            out_FC = F.leaky_relu(out_FC, negative_slope=0.1)

            FC_Fi[:,:,i] = F.leaky_relu(self.fc3_Fi(out_FC), negative_slope=0.1)
            FC_Fo[:, :, i] = F.leaky_relu(self.fc3_Fo(out_FC), negative_slope=0.1)
            FC_T[:, :, i] = F.leaky_relu(self.fc3_T(out_FC), negative_slope=0.1)
            FC_L[:, :, i] = F.leaky_relu(self.fc3_L(out_FC), negative_slope=0.1)

            out_lstm_branch = self.N2G(out)
            out_lstm_branch = F.leaky_relu(out_lstm_branch,negative_slope=0.1)
            LSTM_in[:,i,:] = out_lstm_branch.view(out_lstm_branch.size(0), out_lstm_branch.size(1))

        Attn, states = self.lstm(LSTM_in)
        Attn = torch.squeeze(Attn)
        Attn = self.sm(Attn)

        #first column for language, second for motor
        out_Fi = torch.matmul(FC_Fi,Attn[:,1])
        out_Fo = torch.matmul(FC_Fo, Attn[:,1])
        out_T = torch.matmul(FC_T, Attn[:,1])
        out_L = torch.matmul(FC_L, Attn[:,0])

        return out_Fi, out_Fo, out_T, out_L, Attn


import torch.utils.data.dataset


for test in range(8):
    test_index = test+1
    lr = 0.005
    nbepochs = 300
    BATCH_SIZE = 1
    class_0 = 0.2
    class_M = 1.5
    class_L = 2.25
    class_2 = 0.5


    trainset = data_train(index=test_index,fold=8)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    testset = data_test(index=test_index)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

    net = multitask_dynamic_lstm(trainset.X)
    if use_cuda:
        net = net.cuda(0)
        net = torch.nn.DataParallel(net, device_ids=[0])

    momentum = 0.9
    wd = 0.00005  ## Decay for L2 regularization

    def init_weights_he(m):
        print(m)
        if type(m) == torch.nn.Linear:
            fan_in = net.dense1.in_features
            he_lim = np.sqrt(6) / fan_in
            m.weight.data.uniform_(-he_lim, he_lim)
            print(m.weight)

    class_weight_M = torch.FloatTensor([class_0, class_M, class_2])
    criterion1 = torch.nn.CrossEntropyLoss(weight=class_weight_M)
    class_weight_L = torch.FloatTensor([class_0, class_L, class_2])
    criterion2 = torch.nn.CrossEntropyLoss(weight=class_weight_L)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)

    def train(epoch,alpha=1.0):
        net.train()
        for batch_idx, (X, L, Fi, Fo, T) in enumerate(trainloader):


            if use_cuda:
                X, L,Fi,Fo,T = X.cuda(), L.cuda(), Fi.cuda(), Fo.cuda(), T.cuda()
            optimizer.zero_grad()
            X, L, Fi, Fo, T = Variable(X), Variable(L), Variable(Fi), Variable(Fo), Variable(T)
            out_Fi, out_Fo, out_T, out_L, Attn = net(X)

            L = L.view(L.size(0) * L.size(1), 1)
            L = np.squeeze(L)
            L = Variable(L)
            Fi = Fi.view(Fi.size(0) * Fi.size(1), 1)
            Fi = np.squeeze(Fi)
            Fi = Variable(Fi)
            Fo = Fo.view(Fo.size(0) * Fo.size(1), 1)
            Fo = np.squeeze(Fo)
            Fo = Variable(Fo)
            T = T.view(T.size(0) * T.size(1), 1)
            T = np.squeeze(T)
            T = Variable(T)

            #one way to code multi-task learning with missing labels
            loss1 = criterion2((out_L),L)
            if Fi[0] == 6:
                loss2 = 0
            else:
                loss2 = criterion1((out_Fi),Fi)
            if Fo[0] == 6:
                loss3 = 0
            else:
                loss3 = criterion1((out_Fo),Fo)
            if T[0] == 6:
                loss4 = 0
            else:
                loss4 = criterion1((out_T),T)


            loss_total = loss1 + loss2 + loss3 + loss4
            loss_total.backward()
            optimizer.step()

        return 

    def test():
        net.eval()
        test_loss = 0
        running_loss = 0.0

        total_Fi_out = []
        total_Fo_out = []
        total_T_out = []
        total_L_out = []

        for batch_idx, (X, L, Fi, Fo, T) in enumerate(testloader):

            if use_cuda:
                X,L,Fi,Fo,T = X.cuda(), L.cuda(), Fi.cuda(), Fo.cuda(), T.cuda()

            with torch.no_grad():
                if use_cuda:
                    X,  L, Fi, Fo, T = X.cuda(), L.cuda(), Fi.cuda(), Fo.cuda(), T.cuda()
                optimizer.zero_grad()
                X, L, Fi, Fo, T = Variable(X), Variable(L), Variable(Fi), Variable(Fo), Variable(T)
                out_Fi, out_Fo, out_T, out_L, attn = net(X)
                total_Fi_out.append(out_Fi)
                total_Fo_out.append(out_Fo)
                total_T_out.append(out_T)
                total_L_out.append(out_L)

        return total_Fi_out, total_Fi_out, total_T_out, total_L_out, attn
    for epoch in range(nbepochs):

        train(epoch)

    out_Fi, out_Fo, out_T, out_L, attn = test()

