#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from cmath import nan
import numpy as np
import pandas as pd
import time

def gen_data(data, ntr, N, Td):
    '''
    if flag:
        data=pd.read_csv(fname)
    else:
        data=pd.read_csv(fname,header=None)
    '''
    data=np.reshape(data,[-1,Td,N])
    return data[0:ntr]

def normalize(a):
    mu=np.mean(a,axis=1,keepdims=True)
    std=np.std(a,axis=1,keepdims=True)
    return (a-mu)/(std+0.1)

def compute_dtw(a,b,Td,o=1,T=12):
    a=normalize(a)
    b=normalize(b)
    d=np.reshape(a,[-1,1,Td])-np.reshape(b,[-1,Td,1])
    # d = np.sum(np.abs(d),axis=0)
    d=np.linalg.norm(d,axis=0,ord=o)
    # print(d)
    D=np.zeros([Td,Td])
    for i in range(Td):
        for j in range(max(0,i-T),min(Td,i+T+1)):
            if (i==0) and (j==0):
                D[i,j]=d[i,j]**o
                continue
            if (i==0):
                D[i,j]=d[i,j]**o+D[i,j-1]
                continue
            if (j==0):
                D[i,j]=d[i,j]**o+D[i-1,j]
                continue
            if (j==i-T):
                D[i,j]=d[i,j]**o+min(D[i-1,j-1],D[i-1,j])
                continue
            if (j==i+T):
                D[i,j]=d[i,j]**o+min(D[i-1,j-1],D[i,j-1])
                continue
            D[i,j]=d[i,j]**o+min(D[i-1,j-1],D[i-1,j],D[i,j-1])

    return D[-1,-1]**(1.0/o)

def gen_dtw_adj(feat, Td, days, unknow_list, know_list, Atype="unknow"):
    """
    feat: time series
    Td: time intervals per day
    days: day num
    unknow_set: masked sensor ids
    Atype: "unknow" unknow-know; "know" know-know
    """
    N = feat.shape[0]
    # feat (days, Td, N)
    feat = gen_data(feat.transpose(), int(days), N, Td)
    At = np.zeros([N,N])

    if "unknow" == Atype:
        for i in range(len(know_list)):
            # print("know sensor:", know_list[i])
            for j in range(len(unknow_list)):
                tmp_dist = compute_dtw(feat[:,:,know_list[i]],feat[:,:,unknow_list[j]],Td)
                At[know_list[i],unknow_list[j]] = tmp_dist
    elif "know" == Atype:
        for i in range(len(know_list)):
            # print("know sensor:",know_list[i])
            for j in range(i+1,len(know_list)):
                tmp_dist = compute_dtw(feat[:,:,know_list[i]],feat[:,:,know_list[j]],Td)
                At[know_list[i],know_list[j]] = tmp_dist
    else:
        print("Error type, plese check")
    
    return At+At.T

def gen_temporal_adj(A_dtw, K, unknow_set, Atype="unknow"):
    """
    A_dtw: the dtw similarity between sensors
    ratio: top-ratio similarity
    unknow_set: masked sensor ids
    Atype: "unknow" unknow-know; "know" know-know
    """
    N = A_dtw.shape[0]
    W_adj = np.zeros(A_dtw.shape)
    top = K
    full_set = set(range(0,N))
    know_set = full_set-unknow_set
    know_list = list(know_set)
    unknow_list = list(unknow_set)
    
    if "unknow" == Atype:
        for i in range(len(unknow_set)):
            a = A_dtw[unknow_list[i],know_list].argsort()[0:top+1]
            # print(len(a))
            for j in range(top):
                W_adj[unknow_list[i],know_list[a[j]]]=1
                
    elif "know" == Atype:
        for i in range(len(know_set)):
            a = A_dtw[know_list[i],know_list].argsort()[0:top+1]
            for j in range(top):
                W_adj[know_list[i],know_list[a[j]]]=1
                if know_list[i] == know_list[a[j]]:
                    W_adj[know_list[i],know_list[a[top]]]=1
                

    # print(np.sum(W_adj[know_list,:][:, know_list],axis=1))

    for i in range(N):
        for j in range(N):
            if i==j:
                W_adj[i][j] = 1
    
    return W_adj
