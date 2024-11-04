import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.module import Module

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
        
class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim=2):
        super().__init__()
        self.clf = nn.Linear(in_dim, out_dim)
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return F.log_softmax(x, dim=1)
    
class brainSpan_MLP(nn.Module):
    def __init__(self, nfeat,dropout,nhid1=256):
        super(brainSpan_MLP, self).__init__()
        self.fc1 = nn.Linear(nfeat,nhid1)
        self.fc2 = nn.Linear(nhid1,nhid1)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, self.dropout, training=self.training)
        return x
    
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features)) 

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
class graph_GCN(nn.Module):
    def __init__(self, nfeat,dropout,nhid1=256,nclass=2):
        super(graph_GCN, self).__init__()
        self.fc = nn.Linear(nfeat,nhid1)
        self.conv1  = GraphConvolution(nhid1, nhid1)
        self.conv2  = GraphConvolution(nhid1, nhid1)
        self.dropout = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.fc(x))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.conv1(x1, adj) + x1) 
        x2 = F.dropout(x2, self.dropout, training=self.training) 
        x3 = F.relu(self.conv2(x2, adj) + x2) 
        x3 = F.dropout(x3, self.dropout, training=self.training)
        return x3
    
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

    
class hypergrph_HGNN(nn.Module):
    def __init__(self, nfeat, n_hid, dropout=0.5, n_class=2):
        super(hypergrph_HGNN, self).__init__()
        self.dropout = dropout
        self.fc = nn.Linear(nfeat,n_hid)
        self.hgc1 = HGNN_conv(n_hid, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)
        self.hgc3 = HGNN_conv(n_hid, n_hid)
        self.outLayer = nn.Linear(n_hid, n_class)
    def forward(self, x, G):
        x1 = F.relu(self.fc(x))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        
        x2 = F.relu(self.hgc1(x1, G) + x1) 
        x2 = F.dropout(x2, self.dropout, training=self.training) 
        
        x3 = F.relu(self.hgc2(x2, G) + x2) 
        x3 = F.dropout(x3, self.dropout, training=self.training) 
        
        x4 = F.relu(self.hgc3(x3, G) + x3)
        x4 = F.dropout(x4, self.dropout, training=self.training)
        return x4

class CrossViewAttention(nn.Module):
    def __init__(self, featureDim, nhead, dropout, nhid=128, nclass=2):
        super(CrossViewAttention, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=featureDim, nhead=nhead,dim_feedforward=featureDim*2,dropout=0.5)
        self.cls = nn.Linear(featureDim, nclass)
        self.dropout = dropout
        
    def forward(self, x_1, x_2, x_3):
        x_1 = x_1.reshape(x_1.shape[0],1,x_1.shape[1])
        x_2 = x_2.reshape(x_2.shape[0],1,x_2.shape[1])
        x_3 = x_3.reshape(x_3.shape[0],1,x_3.shape[1])
        t = torch.cat([x_1,x_2, x_3],dim = 1) 
        t = t.permute(1,0,2)
        t = self.encoder_layer(t).transpose(0,1)
        h = F.relu(torch.mean(t,dim=1))
        h= F.dropout(h, self.dropout, training=self.training)
        res = self.cls(h)
        return F.log_softmax(res, dim=1) 