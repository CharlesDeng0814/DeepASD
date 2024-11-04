from sklearn.metrics import auc, roc_auc_score
from sklearn.metrics import precision_recall_curve
import pandas as pd
import numpy as np
import torch
import random
import scipy.sparse as sp

def cal_auc(output, labels):
    outputTest = output.cpu().detach().numpy()
    outputTest = np.exp(outputTest)
    outputTest = outputTest[:,1]
    labelsTest = labels.cpu().numpy()
    AUROC = roc_auc_score(labelsTest, outputTest)
    precision, recall, _thresholds = precision_recall_curve(labelsTest, outputTest)
    AUPRC = auc(recall, precision)
    return AUROC,AUPRC

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() 

def _generate_G_from_H_weight(H, W):
    n_edge = H.shape[1]
    DV = np.sum(H * W, axis=1)  # the degree of the node
    DE = np.sum(H, axis=0)  # the degree of the hyperedge
    invDE = np.mat(np.diag(1/DE))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T
    G = DV2 * H * W * invDE * HT * DV2
    return G

def getData(positiveGenePath, negativeGenePath, geneList):
    positiveGene = pd.read_csv(positiveGenePath, header = None)
    positiveGene = list(positiveGene[0].values)
    positiveGene = list(set(geneList)&set(positiveGene))
    positiveGene.sort()
    negativeGene = pd.read_csv(negativeGenePath, header = None)     
    negativeGene = negativeGene[0]
    negativeGene = list(set(negativeGene)&set(geneList))
    negativeGene.sort()
    
    #print("positiveGene = ",len(positiveGene))
    labelFrame = pd.DataFrame(data = [0]*len(geneList), index = geneList)
    labelFrame.loc[positiveGene,:] = 1
    positiveIndex = np.where(labelFrame == 1)[0]
    labelFrame.loc[negativeGene,:] = -1
    negativeIndex = np.where(labelFrame == -1)[0]
    labelFrame = pd.DataFrame(data = [0]*len(geneList), index = geneList)
    labelFrame.loc[positiveGene,:] = 1
    
    positiveIndex = list(positiveIndex)
    negativeIndex = list(negativeIndex)
    sampleIndex = positiveIndex + negativeIndex
    sampleIndex = np.array(sampleIndex)
    label = pd.DataFrame(data = [1]*len(positiveIndex) + [0]*len(negativeIndex))
    label = label.values.ravel()
    return  sampleIndex, label, labelFrame

def processingIncidenceMatrix(geneSetGenePath, geneList):
    geneSetGene = pd.read_csv(geneSetGenePath,header=None)
    geneSetGene = list(geneSetGene[0].values)

    ids = ['h','c1','c2','c3','c4','c5','c6','c7','c8']
    incidenceMatrix = pd.DataFrame()
    for id in ids:
        geneSetData = sp.load_npz('./data/anatatedGeneSets/'+id+'_GenesetsMatrix.npz')
        incidenceMatrixTemp  = pd.DataFrame(data = geneSetData.A,index= geneSetGene)
        incidenceMatrix  = pd.concat([incidenceMatrix,incidenceMatrixTemp],axis=1)

    incidenceMatrix = incidenceMatrix.loc[geneList]
    incidenceMatrix.columns = np.arange(incidenceMatrix.shape[1])
    return incidenceMatrix

def getBrainSpanFeature(brainSpanPath, geneList):
    brainSpanMatrix = pd.read_csv(brainSpanPath,index_col=0)
    
    brainSpanMatrix = brainSpanMatrix.loc[geneList]
    brainSpanMatrix_dropDup = brainSpanMatrix.loc[brainSpanMatrix.index.drop_duplicates(keep=False)]
    duplicate_index_rows = brainSpanMatrix[brainSpanMatrix.index.duplicated(keep=False)]
    averaged_duplicate_rows = duplicate_index_rows.groupby(duplicate_index_rows.index).mean()
    brainSpanMatrix_dropDup = pd.concat([brainSpanMatrix_dropDup,averaged_duplicate_rows],axis=0)
    brainSpanMatrix_dropDup = brainSpanMatrix_dropDup.loc[geneList]
    brainSpanMatrix = torch.from_numpy(brainSpanMatrix_dropDup.values).float()
    
    return brainSpanMatrix

def processPPINetwork(PPINetPath, PPIGenePath, geneList):
    STRING_PPI = sp.load_npz(PPINetPath).toarray()
    PPIGene = pd.read_csv(PPIGenePath,index_col=None,header=None)
    PPIGene = list(PPIGene[0].values)
    PPI_graph = pd.DataFrame(data = STRING_PPI, index = PPIGene, columns = PPIGene)
    
    PPI_graph = PPI_graph.loc[geneList,geneList]
    PPI_graph = PPI_graph.values

    adj_PPI=sp.coo_matrix(PPI_graph)
    adj_PPI = adj_PPI + adj_PPI.T.multiply(adj_PPI.T > adj_PPI) - adj_PPI.multiply(adj_PPI.T > adj_PPI)
    adj_PPI = normalize_adj(adj_PPI + sp.eye(adj_PPI.shape[0]))
    adj_PPI = torch.FloatTensor(np.array(adj_PPI.todense()))
    return adj_PPI
