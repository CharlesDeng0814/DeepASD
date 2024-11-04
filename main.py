import pandas as pd
import sys, os, random
import numpy as np
import scipy.sparse as sp
from train_pred import trainPred
from utils import processingIncidenceMatrix, getBrainSpanFeature, processPPINetwork
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
 

if __name__ == "__main__":    
    #_, outputPath = sys.argv
    lr = 0.001
    epochs = 200
    dropout = 0.2
    weight_decay = 5e-3
    nhead = 4
    
    positiveGenePath = r'./data/ASDgene/asd_truth_set.csv'
    negativeGenePath = r'./data/ASDgene/control_gene_set.csv'
    geneSetGenePath = r'./data/anatatedGeneSets/allGene.csv'
    PPINetPath = r'./data/PPI/STRING_PPI.npz'
    PPIGenePath = r'./data/PPI/STRING_genes.csv'
    brainSpanPath = r'./data/brainSpanAtlas/brainSpanMatrix.csv'
    geneListPath  = r'./data/geneList.txt'
    geneList = pd.read_csv(geneListPath, header=None)
    geneList = list(geneList[0].values)
    Function_hypergraph = processingIncidenceMatrix(geneSetGenePath, geneList)
    brainSpanFeature = getBrainSpanFeature(brainSpanPath, geneList)
    PPI_graph = processPPINetwork(PPINetPath, PPIGenePath, geneList)

    brainSpanFeature = brainSpanFeature.cuda()
    PPI_graph = PPI_graph.cuda()
    print("read data success")
    aurocList, auprcList = trainPred(geneList, brainSpanFeature, Function_hypergraph, PPI_graph, positiveGenePath,
                                          negativeGenePath, lr, epochs, dropout, weight_decay, nhead) 

    print(np.mean(aurocList)) # 0.920
    print(np.mean(auprcList)) # 0.761