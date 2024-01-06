# -*- coding:utf-8 -*-
from re import T
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = torch.mean(kernels[:batch_size, :batch_size])
        YY = torch.mean(kernels[batch_size:, batch_size:])
        XY = torch.mean(kernels[:batch_size, batch_size:])
        YX = torch.mean(kernels[batch_size:, :batch_size])
        loss = torch.mean(XX + YY - XY -YX)
        return loss
    
class GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        '''
        graph convolutional operation, a simple GCN we defined in STFGNN

        Parameters
        ----------
        A: (N, N)

        X:(B, N, C)

        in_features: int, C

        out_features: int, C'

        Returns
        ----------
        output shape is (B, N, C')
        '''
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features,out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)
        
        self.reset_parameters()

    def reset_parameters(self):
        stvd = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stvd,stvd)
        if self.bias is not None:
            self.bias.data.uniform_(-stvd,stvd)

    def forward(self, X, A):
        support = torch.matmul(X,self.weight)
        output = torch.matmul(A,support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        

class GCNL(nn.Module):
    def __init__(self, number_of_filter, number_of_feature):
        '''
        graph convolutional operation, a simple GCN we defined in STFGNN

        Parameters
        ----------
        A: (N, N)

        X:(B, N, C)

        num_of_filter: int, C'

        num_of_features: int, C

        Returns
        ----------
        output shape is (B, N, C')
        '''
        super(GCNL, self).__init__()
        self.gcn1 = GCN(number_of_feature,number_of_filter,True)
        self.gcn2 = GCN(number_of_feature,number_of_filter,True)
        self.sig = nn.Sigmoid()

    def forward(self, X, A):
        lhs = self.gcn1(X, A)
        rhs = self.gcn2(X, A)
        return (lhs * self.sig(rhs))

    
class GCNBlock_Single(nn.Module):
    def __init__(self, num_of_feature, filters):
        '''
        graph convolutional operation, a simple GCN we defined in STFGNN

        Parameters
        ----------
        A: (N, N)

        num_of_node: int, N

        num_of_feature: int, C

        filters: list[int], list of C'

        X:(B, N, C)

        Returns
        ----------
        output shape is (B, N, C')
        '''
        super(GCNBlock_Single, self).__init__()
        self.gcn_layers = nn.ModuleList()
        # self.num_of_node = num_of_node
        for i in range(len(filters)):
            self.gcn_layers.append(GCNL(filters[i], num_of_feature))
            num_of_feature = filters[i]
            

    def forward(self, X, A):
        lay_output = []
        for gcn in self.gcn_layers:
            X = gcn(X, A)
            # shape (B, N, C')
            lay_output.append(X)
        # shape (L, B, N, C')
        lay_output = torch.stack(lay_output)
        # shape (B, N, C')
        z = torch.max(lay_output, dim=0)[0]
        return z
    
class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()
    

class USPBlock_Inductive_MSconv(nn.Module):

    def __init__(self, input_length, number_of_feature, filters):
        '''
        Combine the GNN and CNN together

        Parameters
        ----------

        input_length: int, length of time series, T

        num_of_node: int, N

        num_of_feature: int, C

        filters: list[int], list of C'

        X:(B, T, N, C)

        Returns
        ----------
        output shape is (B, N, T')
        '''

        super(USPBlock_Inductive_MSconv, self).__init__()
        self.input_length = input_length
        self.number_of_feature = number_of_feature

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        kernel_size = 2
        conv_channels = [64,32,64]
        num_level = len(conv_channels)
        layers = []
        for i in range(num_level):
            dilation_size = 2**i
            in_channels = number_of_feature if i == 0 else conv_channels[i-1]
            out_channels = conv_channels[i]
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation_size), padding=(0, padding))
            self.conv.weight.data.normal_(0, 0.01)
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]
        self.tempnet = nn.Sequential(*layers)
        
        
        self.ts_windows = nn.ModuleList()
        for i in range(input_length):
            # input (B, N, C) output (B, N, C')
            self.ts_windows.append(GCNBlock_Single(number_of_feature, filters))

        self.tt_windows = nn.ModuleList()
        for i in range(input_length):
            # input (B, N, C) output (B, N, C')
            self.tt_windows.append(GCNBlock_Single(number_of_feature, filters))
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.414)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

    def forward(self, X, A_s, A_t):
        # CNN
        X_temp = X.clone()
        X_temp = X_temp.permute(0, 3, 2, 1)
        X_temp = self.tempnet(X_temp)
        x_res = X_temp.permute(0, 3, 2, 1)

        # GCN for A_S
        ts_slide = []
        for i in range(self.input_length):
            # input (B, N, C') output (B, N, C')
            zs = torch.reshape(X[:,i:i+1,:,:],(-1,X.shape[2],self.number_of_feature))
            # print(zs.shape)
            zs = self.ts_windows[i](zs,A_s)
            # print(zs.shape)
            ts_slide.append(torch.unsqueeze(zs, 1))
        # (B, T, N, C)
        x_sp = torch.cat(ts_slide, dim=1)

        # GCN for A_t
        tt_slide = []
        for i in range(self.input_length):
            # input (B, N, C') output (B, N, C')
            zt = torch.reshape(X[:,i:i+1,:,:],(-1,X.shape[2],self.number_of_feature))
            zt = self.tt_windows[i](zt,A_t)
            tt_slide.append(torch.unsqueeze(zt, 1))
        # (B, T, N, C)
        x_tp = torch.cat(tt_slide, dim=1)
        x_gcn = torch.stack([x_sp,x_tp])
        # (B, T-1, N, C')
        x_gcn = torch.max(x_gcn, dim=0)[0]

        # (B, T, N, C)
        output = x_gcn+x_res

        return output

# has dynamic temporal matrix
class USPGCN_MultiD_Inductive_Tattr(nn.Module):

    def __init__(self, input_length, number_of_feature, filters, use_mask, device, output_length):
        '''
        graph convolutional operation, a simple GCN we defined in STFGNN

        Parameters
        ----------
        A: (N, N)

        T: int, length of time series, T

        num_of_node: int, N

        num_of_feature: int, C

        filters_list: list[list[int]], list of C'

        X:(B, T, N, C)

        Returns
        ----------
        output shape is (B, N, T')
        '''

        super(USPGCN_MultiD_Inductive_Tattr, self).__init__()
        self.device = device
        self.use_mask = use_mask
        self.input_length = input_length
        self.number_of_feature = number_of_feature
        self.output_length = output_length

        self.fc = nn.Linear(1, number_of_feature)
        self.fc_te = nn.Linear(1, number_of_feature)
        # self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(0.01)
        
        self.blocks = nn.ModuleList()
        for i in range(len(filters)):
            self.blocks.append(USPBlock_Inductive_MSconv(input_length, number_of_feature, filters[i]))
            # input_length = input_length - 1

        self.fc_time = nn.Linear(input_length, output_length)
        self.fc_out = nn.Linear(filters[-1][-1], 1)
        self.project = nn.Sequential(
            nn.Linear(filters[-1][-1], filters[-1][-1]),
            nn.BatchNorm1d(filters[-1][-1]),
            nn.ReLU(),
            nn.Linear(filters[-1][-1], filters[-1][-1]),
            # nn.Sigmoid()
        )
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.414)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, X, TE, A_s, A_t):
        # X:(B, T, N, C') embedding X
        X = self.fc(X)
        TE = self.fc_te(TE)
        X = torch.mul(X,TE)
        # (B, T, N, C)
        for block in self.blocks:
            X = block(X, A_s, A_t)
        
        rep = X.clone()
        rep = rep[:,-1:,:,:]
        # project head
        rep = torch.squeeze(rep).transpose(1,2)
        rep = torch.sum(rep, dim=2)
        rep = self.project(rep)  
        
        output = X.transpose(1,3)
        # split into two subsets
        output = self.leakyrelu(self.fc_time(output))
        output = output.transpose(1,3)
        output = self.leakyrelu(self.fc_out(output))
        
        return output, rep
