from copyreg import pickle
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
import math
from geographiclib.geodesic import Geodesic
import random
from sklearn.metrics.pairwise import cosine_similarity
        
class MyDatasetTatt_Noslide_CL(Dataset):
    def __init__(self, data, label, TE, split_start, split_end, his_length, pred_length):
        split_start = int(split_start)
        split_end = int(split_end)
        self.data = data[:,split_start: split_end]
        self.label = label[:,split_start: split_end]
        self.mean = self.data.mean()
        self.std = self.data.std()
        self.his_length = his_length
        self.pred_length = pred_length
        self.TE = TE
    
    def __getitem__(self, index):
        x = self.data[:, index*self.his_length: (index+1)* self.his_length]
        # x = self.data[:, index: index + self.his_length]
        x = (x - self.mean) / self.std
        # X (T, N, C)
        x = x.transpose()
        x = np.expand_dims(x,axis=2)

        x1 = self.label[:, index*self.his_length: (index+1)* self.his_length]
        # x = self.data[:, index: index + self.his_length]
        x1 = (x1 - self.mean) / self.std
        # X (T, N, C)
        x1 = x1.transpose()
        x1 = np.expand_dims(x1,axis=2)

        y = self.label[:, index*self.his_length: (index+1)* self.his_length + self.pred_length]
        y = y.transpose()
        y = np.expand_dims(y,axis=2)
        # TE (T, 1)
        te = self.TE[:,index*self.his_length: (index+1)* self.his_length]
        # TE (N, T)
        te = np.tile(te, (x.shape[1],1))
        # TE (T,N,C)
        te = te.transpose()
        te = np.expand_dims(te,axis=2)
        return torch.Tensor(x), torch.Tensor(x1), torch.Tensor(y), torch.Tensor(te)
    def __len__(self):
        return self.data.shape[1] // self.pred_length - 1
        # return int((self.data.shape[1]-self.pred_length) / self.his_length)

class MyDatasetTatt_CL(Dataset):
    def __init__(self, data, label, TE, split_start, split_end, his_length, pred_length):
        split_start = int(split_start)
        split_end = int(split_end)
        self.data = data[:,split_start: split_end]
        self.label = label[:,split_start: split_end]
        self.mean = self.data.mean()
        self.std = self.data.std()
        self.his_length = his_length
        self.pred_length = pred_length
        self.TE = TE
    
    def __getitem__(self, index):
        # data: (N, T)
        x = self.data[:, index: index + self.his_length]
        x = (x - self.mean) / self.std
        # X (T, N, C)
        x = x.transpose()
        x = np.expand_dims(x,axis=2)

        x1 = self.label[:, index: index + self.his_length]
        x1 = (x1 - self.mean) / self.std
        # X (T, N, C)
        x1 = x1.transpose()
        x1 = np.expand_dims(x1,axis=2)

        y = self.label[:, index: index + self.his_length + self.pred_length]
        y = y.transpose()
        y = np.expand_dims(y,axis=2)
        # TE (T, 1)
        te = self.TE[:,index: index + self.his_length]
        # TE (N, T)
        te = np.tile(te, (x.shape[1],1))
        # TE (T,N,C)
        te = te.transpose()
        te = np.expand_dims(te,axis=2)
        return torch.Tensor(x), torch.Tensor(x1), torch.Tensor(y), torch.Tensor(te)
    def __len__(self):
        return self.data.shape[1] - self.his_length - self.pred_length + 1



def generate_dataset_tatt_cl(data,label,TE,train_length,his_length=24,pred_length=24,batch_size=32):
    """
    Args:
        data: input dataset, shape like T * N
        batch_size: int 
        train_ratio: float, the ratio of the dataset for training
        his_length: the input length of time series for prediction
        pred_length: the target length of time series of prediction

    Returns:
        train_dataloader: torch tensor, shape like batch * N * his_length * features
        test_dataloader: torch tensor, shape like batch * N * pred_length * features
    """
    train_dataset = MyDatasetTatt_CL(data, label, TE, 0, train_length, his_length, pred_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # print(train_dataset.__len__())

    test_dataset = MyDatasetTatt_Noslide_CL(data, label, TE, train_length, data.shape[1], his_length, pred_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def generate_dataset_tatt_no_slide_cl(data,label,TE,train_length,his_length=24,pred_length=24,batch_size=32):
    """
    Args:
        data: input dataset, shape like T * N
        batch_size: int 
        train_ratio: float, the ratio of the dataset for training
        his_length: the input length of time series for prediction
        pred_length: the target length of time series of prediction

    Returns:
        train_dataloader: torch tensor, shape like batch * N * his_length * features
        test_dataloader: torch tensor, shape like batch * N * pred_length * features
    """ 
    start_idx = np.random.choice(range(his_length))
    train_dataset = MyDatasetTatt_Noslide_CL(data, label, TE, start_idx, train_length, his_length, pred_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MyDatasetTatt_Noslide_CL(data, label, TE, train_length, data.shape[1], his_length, pred_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def load_mel_data(feat_fn, sensor_fn, adj_type="connectivity", normalized_k=0.05):
    #  speed = 0; volume = 1
    feat = np.load(feat_fn)[:,:]
    sensors = np.load(sensor_fn)
    lon = sensors[:,2]
    lat = sensors[:,3]
    dist_mx = latlon2dist(lat, lon)

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx/std))
    adj_mx[adj_mx < normalized_k] = 0
        
    adj_mx[np.isinf(adj_mx)] = 0
    
    return feat, sensors,dist_mx,adj_mx

def load_pems_data(feat_fn,sensor_fn, normalized_k = 0.05):
    # feat(N, T); sensors(id,lat,lon) N = 325
    df = pd.read_hdf(feat_fn)
    # transfer_set = df.as_matrix()
    sensors = pd.read_csv(sensor_fn,header=None)
    sensor_ids = sensors[0].tolist()
    sensors = np.array(sensors)

    lat = sensors[:,1]
    lon = sensors[:,2]
    dist_mx = latlon2dist(lat, lon)

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx/std))
    adj_mx[adj_mx < normalized_k] = 0
        
    adj_mx[np.isinf(adj_mx)] = 0

    feat = np.array(df).transpose()
    
    return feat, sensors,dist_mx,adj_mx

def load_pems_data_np(feat_fn,sensor_fn, normalized_k = 0.05):
    # feat(N, T, C[speed,flow]); sensors(id,lat,lon)
    df =np.load(feat_fn)[:,:,0]
    # transfer_set = df.as_matrix()
    sensors = pd.read_csv(sensor_fn)
    lat = np.array(sensors['lat'])
    lon = np.array(sensors['lon'])
    dist_mx = latlon2dist(lat, lon)
    sensors = np.array(sensors)

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx/std))
    adj_mx[adj_mx < normalized_k] = 0
        
    adj_mx[np.isinf(adj_mx)] = 0
    
    return df, sensors,dist_mx,adj_mx


def load_air(feat_fn,sensor_fn, normalized_k = 0.05):
    # feat(N, T, C[speed,flow]); sensors(id,lat,lon)
    df =np.load(feat_fn)
    # transfer_set = df.as_matrix()
    sensors = pd.read_csv(sensor_fn)
    lat = np.array(sensors['lat'])
    lon = np.array(sensors['lon'])
    dist_mx = latlon2dist(lat, lon)
    sensors = np.array(sensors)

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx/std))
    adj_mx[adj_mx < normalized_k] = 0
        
    adj_mx[np.isinf(adj_mx)] = 0
    
    return df, sensors,dist_mx,adj_mx




def select_khop_neighbour(A,sensor_ids,khop,know_list):
    for k in range(khop):
        sids_list = []
        for sid in sensor_ids:
            relation = A[sid,:]
            ids = np.where(relation!=0)[0]
            for i in ids:
                sids_list.append(i)
        sensor_ids = list(set(sids_list))
    return sensor_ids

def latlon2dist(lat, lon):
    loc_num = lat.shape[0]
    dist_mat = np.zeros([loc_num,loc_num])
    for i in range(loc_num):
        for j in range(i+1, loc_num):
            s = Geodesic.WGS84.Inverse(lat[i],lon[i],lat[j],lon[j])
            dist_mat[i,j] = s['s12']/1000
    return dist_mat+dist_mat.T

def get_normalized_connective_adj(A,is_self=False):
    """
    Returns a normallized tensor.
    """
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    if is_self:
        A_wave = A_wave + np.eye(A_wave.shape[0])

    return torch.from_numpy(A_wave.astype(np.float32))

def get_normalized_weighted_adj(adj):
    """
    Returns a normallized tensor.
    """
    A = np.copy(adj)
    node_num = A.shape[0]
    for i in range(node_num):
        adj_idx = np.where(A[i]!=0)[0]
        adj_i = A[i,adj_idx]
        adj_i = np.exp(adj_i)
        adj_sum = np.sum(adj_i)
        adj_i = adj_i/adj_sum
        A[i,adj_idx] = np.copy(adj_i)
    for i in range(node_num):
        A[i][i] = 1

    return torch.from_numpy(A.astype(np.float32))

def get_normalized_mat(A):
    """
    Returns a normallized tensor.
    """
    D = torch.sum(A, dim=1).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = torch.reciprocal(torch.sqrt(D))
    # print(diag.reshape((1, -1)).shape)
    A_wave = torch.multiply(torch.multiply(diag.reshape((-1, 1)),A),
                         diag.reshape((1, -1)))

    return A_wave

def gen_TE(T,total_day):
    num_interval = total_day*T
    TE = list(set(range(0, num_interval)))
    TE = np.array(TE)
    TE = TE % T
    TE = TE.astype(np.int32)
    TE = TE[np.newaxis]

    return TE