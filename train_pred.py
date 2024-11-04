import pandas as pd
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import random
from utils import cal_auc, _generate_G_from_H_weight, getData
from models import brainSpan_MLP, hypergrph_HGNN, graph_GCN, CrossViewAttention, Classifier_1

def train_test(trainIndex, testIndex, labelFrame, brainSpanFeature, incidenceMatrix, PPI_grpah, geneList, lr, epochs, dropout, weight_decay, nhead):
    trainFrame = labelFrame.iloc[trainIndex]
    trainPositiveGene = list(trainFrame.where(trainFrame==1).dropna().index)
    trainNegativeGene = list(trainFrame.where(trainFrame==0).dropna().index)
    positiveMatrixSum = incidenceMatrix.loc[trainPositiveGene].sum()
        
    # disease-specific hyperedge weight
    selHyperedgeIndex = np.where(positiveMatrixSum>=3)[0]
    selHyperedge = incidenceMatrix.iloc[:, selHyperedgeIndex]
    hyperedgeWeight = positiveMatrixSum[selHyperedgeIndex].values
    selHyperedgeWeightSum = incidenceMatrix.iloc[:, selHyperedgeIndex].values.sum(0)
    hyperedgeWeight = hyperedgeWeight/selHyperedgeWeightSum
        
    H = np.array(selHyperedge).astype('float')
    DV = np.sum(H * hyperedgeWeight, axis=1)
    for i in range(DV.shape[0]):
        if(DV[i] == 0):
            t = random.randint(0, H.shape[1]-1)
            H[i][t] = 0.0001
    G = _generate_G_from_H_weight(H, hyperedgeWeight)
    N = H.shape[0]
    adj_hyperGraph = torch.Tensor(G).float()
    oneHotFeature = torch.eye(N).float()
    labels = torch.from_numpy(labelFrame.values.reshape(-1,))
        
    model_hypergrph = hypergrph_HGNN(nfeat = N, n_hid = 256, dropout=0.2)
    model_graph = graph_GCN(nfeat = N,dropout=0.5)
    model_brainSpan = brainSpan_MLP(nfeat = 800, dropout = 0.2, nhid1 = 256)
    
    classifier_hgnn = Classifier_1(in_dim = 256,out_dim=2)
    classifier_gcn = Classifier_1(in_dim = 256,out_dim=2)
    classifier_brainSpan = Classifier_1(in_dim = 256,out_dim=2)
            
    optimizer_hgnn = optim.Adam(list(model_hypergrph.parameters())+list(classifier_hgnn.parameters()), lr=0.0005, weight_decay=0.00005)
    optimizer_gcn = optim.Adam(list(model_graph.parameters())+list(classifier_gcn.parameters()), lr=0.0005, weight_decay=0.005)
    optimizer_brainSpan = optim.Adam(list(model_brainSpan.parameters())+list(classifier_brainSpan.parameters()), lr=0.002, weight_decay=0.05)
    schedular_hgnn = optim.lr_scheduler.MultiStepLR(optimizer_hgnn,milestones=[100,200,300,400],gamma=0.5)

    model_CVAttention = CrossViewAttention(featureDim = 256, nhead = nhead, dropout = dropout)
    optimizer_CVAttention = optim.Adam(model_CVAttention.parameters(), lr = lr, weight_decay = weight_decay)
    if torch.cuda.is_available():
        model_hypergrph.cuda()
        model_graph.cuda()
        model_CVAttention.cuda()
        model_brainSpan.cuda()
        classifier_gcn.cuda()
        classifier_hgnn.cuda()
        classifier_brainSpan.cuda()
        oneHotFeature = oneHotFeature.cuda()
        adj_hyperGraph = adj_hyperGraph.cuda()
        labels = labels.cuda()
    for epoch in range(epochs):
        model_hypergrph.train()
        model_graph.train()
        model_brainSpan.train()
        classifier_hgnn.train()
        classifier_gcn.train()
        classifier_brainSpan.train()
        optimizer_hgnn.zero_grad()
        optimizer_gcn.zero_grad()
        optimizer_brainSpan.zero_grad()
        
        output_hgnn = classifier_hgnn(model_hypergrph(oneHotFeature, adj_hyperGraph))
        output_gcn = classifier_gcn(model_graph(oneHotFeature, PPI_grpah))
        output_brainSpan = classifier_brainSpan(model_brainSpan(brainSpanFeature))
        
        loss_train_hgnn = F.nll_loss(output_hgnn[trainIndex], labels[trainIndex])
        loss_train_gcn = F.nll_loss(output_gcn[trainIndex], labels[trainIndex]) 
        loss_train_brainSpan = F.nll_loss(output_brainSpan[trainIndex], labels[trainIndex])
        loss_train_hgnn.backward()
        loss_train_gcn.backward()
        loss_train_brainSpan.backward()

        optimizer_hgnn.step()
        optimizer_gcn.step()
        optimizer_brainSpan.step()
        schedular_hgnn.step()
                    
        if(epoch>epochs/2):
            model_CVAttention.train()
            optimizer_CVAttention.zero_grad()
            output = model_CVAttention(model_hypergrph(oneHotFeature, adj_hyperGraph), model_graph(oneHotFeature, PPI_grpah), model_brainSpan(brainSpanFeature))
            loss = F.nll_loss(output[trainIndex], labels[trainIndex])
            loss.backward()
            optimizer_CVAttention.step()
    model_hypergrph.eval() 
    model_graph.eval()
    model_brainSpan.eval()
    model_CVAttention.eval()
    classifier_hgnn.eval()
    classifier_gcn.eval()
    classifier_brainSpan.eval()
    with torch.no_grad():
        output = model_CVAttention(model_hypergrph(oneHotFeature, adj_hyperGraph), model_graph(oneHotFeature, PPI_grpah), model_brainSpan(brainSpanFeature))
        AUROC_val, AUPRC_val = cal_auc(output[testIndex], labels[testIndex])
        outputFrame = pd.DataFrame(data = output.exp().cpu().detach().numpy(), index = geneList)
    return AUROC_val, AUPRC_val, outputFrame


def trainPred(geneList, brainSpanFeature, incidenceMatrix, PPI_graph, positiveGenePath,
              negativeGenePath, lr, epochs, dropout, weight_decay, nhead):
    auc_fusion_list=list()
    prc_fusion_list=list()
    train, the_label, PPI_labels = getData(positiveGenePath, negativeGenePath, geneList)
    for i in range(5):
        sk_X=train.reshape([-1,1])
        sfolder=StratifiedKFold(n_splits=5,random_state=i,shuffle=True)
        for train_index,test_index in sfolder.split(sk_X,the_label):
            X_train,X_test, y_train, y_test=train[train_index],train[test_index],the_label[train_index],the_label[test_index]
            idx_train=X_train
            temp=PPI_labels.iloc[X_train]
            trainPositiveGene=list(temp.where(temp==1).dropna().index)
            positiveMatrixSum = incidenceMatrix.loc[trainPositiveGene].sum()
            selHyperedgeIndex = np.where(positiveMatrixSum>=3)[0]
            selHyperedge = incidenceMatrix.iloc[:, selHyperedgeIndex]
            hyperedgeWeight = positiveMatrixSum[selHyperedgeIndex].values
            selHyperedgeWeightSum = incidenceMatrix.iloc[:, selHyperedgeIndex].values.sum(0)
            hyperedgeWeight = hyperedgeWeight/selHyperedgeWeightSum

            H = np.array(selHyperedge).astype('float')
            DV = np.sum(H * hyperedgeWeight, axis=1)
            for i in range(DV.shape[0]):
                if(DV[i] == 0):
                    t = random.randint(0, H.shape[1]-1)
                    H[i][t] = 0.0001
            G = _generate_G_from_H_weight(H, hyperedgeWeight)
            N = H.shape[0]
            adj_hyperGraph = torch.Tensor(G)
            features= torch.eye(N)
            labels=PPI_labels
            # 构建模型
            features=features.float()
            #adj=adj.float()
            adj_hyperGraph=adj_hyperGraph.float()
            labels=labels.values
            labels=labels.reshape(-1,)
            labels=torch.from_numpy(labels)

            model_hgnn = hypergrph_HGNN(nfeat=N,n_hid=256,dropout=0)
            model_gcn = graph_GCN(nfeat=N,dropout=0.5)
            model_brainSpan = brainSpan_MLP(nfeat = brainSpanFeature.shape[1], dropout = 0.2, nhid1 = 256)

            classifier_hgnn = Classifier_1(in_dim = 256,out_dim=2)
            classifier_gcn = Classifier_1(in_dim = 256,out_dim=2)
            classifier_brainSpan = Classifier_1(in_dim = 256,out_dim=2)

            optimizer_hgnn = optim.Adam(list(model_hgnn.parameters())+list(classifier_hgnn.parameters()), lr=0.0005, weight_decay=0.00005)
            optimizer_gcn = optim.Adam(list(model_gcn.parameters())+list(classifier_gcn.parameters()), lr=0.0005, weight_decay=0.005)
            optimizer_brainSpan = optim.Adam(list(model_brainSpan.parameters())+list(classifier_brainSpan.parameters()), lr=0.002, weight_decay=0.05)
            schedular_hgnn = optim.lr_scheduler.MultiStepLR(optimizer_hgnn,milestones=[100,200,300,400],gamma=0.5)

            model_fusion = CrossViewAttention(featureDim=256, nhead=nhead, dropout=dropout)
            optimizer_fusion = optim.Adam(model_fusion.parameters(), lr=lr, weight_decay=weight_decay)

            if torch.cuda.is_available():
                model_hgnn.cuda()
                model_gcn.cuda()
                model_fusion.cuda()
                model_brainSpan.cuda()
                classifier_gcn.cuda()
                classifier_hgnn.cuda()
                classifier_brainSpan.cuda()
                features = features.cuda()
                adj_hyperGraph = adj_hyperGraph.cuda()
                labels = labels.cuda()
            for epoch in range(epochs):
                model_hgnn.train() 
                model_gcn.train()
                model_brainSpan.train()
                classifier_hgnn.train()
                classifier_gcn.train()
                classifier_brainSpan.train()
                optimizer_hgnn.zero_grad() # 梯度置0
                optimizer_gcn.zero_grad()
                optimizer_brainSpan.zero_grad()

                output_hgnn = classifier_hgnn(model_hgnn(features, adj_hyperGraph))
                output_gcn = classifier_gcn(model_gcn(features, PPI_graph))
                output_brainSpan = classifier_brainSpan(model_brainSpan(brainSpanFeature))
                #print(output)

                loss_train_hgnn = F.nll_loss(output_hgnn[idx_train], labels[idx_train])
                loss_train_gcn = F.nll_loss(output_gcn[idx_train], labels[idx_train])  
                loss_train_brainSpan = F.nll_loss(output_brainSpan[idx_train], labels[idx_train])

                loss_train_hgnn.backward() 
                loss_train_gcn.backward()
                loss_train_brainSpan.backward()

                optimizer_hgnn.step() 
                optimizer_gcn.step()
                optimizer_brainSpan.step()
                schedular_hgnn.step()

                if(epoch>epochs/2):
                    model_fusion.train()
                    optimizer_fusion.zero_grad()
                    output_fusion = model_fusion(model_hgnn(features, adj_hyperGraph),model_gcn(features, PPI_graph),model_brainSpan(brainSpanFeature))
                    loss_train_fusion = F.nll_loss(output_fusion[idx_train], labels[idx_train])
                    loss_train_fusion.backward()
                    optimizer_fusion.step()
            model_hgnn.eval() # 先将model置为训练状态
            model_gcn.eval()
            model_brainSpan.eval()
            model_fusion.eval()
            classifier_hgnn.eval()
            classifier_gcn.eval() # 置为 evaluation 状态 
            classifier_brainSpan.eval()
            with torch.no_grad():
                output_fusion = model_fusion(model_hgnn(features, adj_hyperGraph),model_gcn(features, PPI_graph),model_brainSpan(brainSpanFeature))
                AUROC_fusion_val,AUPRC_fusion_val=cal_auc(output_fusion[X_test], labels[X_test])
                print('fusion_AUROC: {:.4f}'.format(AUROC_fusion_val.item()),
                       'fusion_AUPRC: {:.4f}'.format(AUPRC_fusion_val.item()))
                auc_fusion_list.append(AUROC_fusion_val.item())
                prc_fusion_list.append(AUPRC_fusion_val.item())
    return auc_fusion_list,prc_fusion_list