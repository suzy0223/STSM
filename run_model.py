from __future__ import division

import torch
import numpy as np
from torch import nn
from utils.funcs import *
from utils.gen_fake_val import *
from utils.eval import *
from utils.fastDTW_adj_gen import *
from models.cl_gcc_cnn_all import *
import torch.nn.functional as F
import random
import argparse
import sys
import os
import time
import warnings
from os import path
from sklearn.metrics.pairwise import cosine_similarity
import logging

warnings.filterwarnings("ignore")


def train_predict(loader, model, optimizer, criterion, device, A_s, A_t, know_sensor_list, masked_sensor_ids,
                  his_length, tempe, weight):
    batch_loss = 0
    batch_predict_loss = 0
    batch_ada_loss = 0
    # training embedding first
    for idx, (inputs_m, inputs_s, targets, time_mat) in enumerate(loader):
        model.train()
        optimizer.zero_grad()

        # (B,T,N,C)
        inputs_m = inputs_m.to(device)
        inputs_s = inputs_s.to(device)
        targets = targets.to(device)
        time_mat = time_mat.to(device)

        # (B,T,N,C)
        outputs_m, rep_m = model(inputs_m, time_mat, A_s, A_t)
        _, rep_s = model(inputs_s, time_mat, A_s, A_t)

        norm1 = rep_m.norm(dim=1)
        norm2 = rep_s.norm(dim=1)
        sim_matrix = torch.mm(rep_m, torch.transpose(rep_s, 0, 1)) / torch.mm(norm1.view(-1, 1), norm2.view(1, -1))
        sim_matrix = torch.exp(sim_matrix / tempe)

        diag = inputs_m.shape[0]
        pos_sim = sim_matrix[range(diag), range(diag)]
            
        u_loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        u_loss = torch.mean(-torch.log(u_loss))

        # know node: predict loss
        predict_targets = targets[:, his_length:, know_sensor_list + masked_sensor_ids, :]
        predict_outputs = outputs_m[:, :, know_sensor_list + masked_sensor_ids, :]
        loss_predict = criterion(predict_outputs, predict_targets).to(device)

        # loss_mmd = mmd(feat_know,feat_unknow).to(device)
        u_loss = weight * u_loss
        loss = loss_predict + u_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        optimizer.step()
        batch_loss += loss.detach().cpu().item()
        batch_predict_loss += loss_predict.detach().cpu().item()
        batch_ada_loss += u_loss.detach().cpu().item()
    return batch_loss / (idx + 1), batch_predict_loss / (idx + 1), batch_ada_loss / (idx + 1)


@torch.no_grad()
def test_predict(loader, model, device, A_s, A_t, know_list, unknow_list, his_length):
    test_pred = []
    test_gt = []
    # m_feat as training, truth feat as label
    for idx, (inputs_m, inputs_s, targets, time_mat) in enumerate(loader):
        model.eval()

        # (B,T,N,C)
        inputs_m = inputs_m.to(device)
        inputs_s = inputs_s.to(device)
        targets = targets.to(device)
        time_mat = time_mat.to(device)

        outputs, _ = model(inputs_m, time_mat, A_s, A_t)

        targets = targets.detach().cpu().numpy()
        outputs = outputs.detach().cpu().numpy()

        predict_targets = targets[:, his_length:, unknow_list, :]
        predict_outputs = outputs[:, :, unknow_list, :]
        predict_targets = np.reshape(predict_targets, (-1, len(unknow_list))).transpose()
        predict_outputs = np.reshape(predict_outputs, (-1, len(unknow_list))).transpose()
        test_gt.append(predict_targets)
        test_pred.append(predict_outputs)

        if idx == 0:
            predict_targets.shape

    test_pred = np.concatenate(test_pred, axis=1)
    test_gt = np.concatenate(test_gt, axis=1)
    logging.debug("Ground truth shape = {}".format(test_gt.shape))
    RMSE, MAE, MAPE, R2 = metric(test_gt, test_pred)
    return RMSE, MAE, MAPE, R2


if __name__ == "__main__":
    """
    Model training
    """

    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="pems08", type=str)
    parser.add_argument('--split_type', default=1, type=int)
    parser.add_argument('--aug_ratio', default=0.5, type=float)
    parser.add_argument('--a_sg_nk', default=0.5, type=float)
    parser.add_argument('--tempe', default=0.5, type=float)
    parser.add_argument('--lweight', default=0.5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--patience', default=100, type=int)
    parser.add_argument('--debug', default=1, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--ada', default=1, type=int)
    parser.add_argument('--k', default=35, type=int)
    parser.add_argument('--unknown_ratio', default=0.5, type=float)
    parser.add_argument('--log_dir', default='log_train', type=str)
    parser.add_argument('--model_dir', default='model_saving', type=str)
    args = parser.parse_args()

    logging_fn = '{}/{}_topk_merge_fix_w_all_ar{}_t{}_{}_s{}_stype{}_lr{}_ada{}_k{}_lw{}_unknown_ratio{}'.format(args.log_dir,args.dataset,str(args.aug_ratio),str(args.tempe),str(args.a_sg_nk),str(args.seed),str(args.split_type),str(args.lr),str(args.ada),str(args.k),str(args.lweight),str(args.unknown_ratio))
    saving_dir = '{}/{}'.format(args.model_dir,args.dataset)
    model_saving_fn = '{}/{}/topk_merge_fix_w_all_ar{}_t{}_{}_s{}_stype{}_lr{}_ada{}_k{}_lw{}_unknown_ratio{}.pkl'.format(args.model_dir,args.dataset,str(args.aug_ratio),str(args.tempe),str(args.a_sg_nk),str(args.seed),str(args.split_type),str(args.lr),str(args.ada),str(args.k),str(args.lweight),str(args.unknown_ratio))
    
    if path.isdir(saving_dir):
        pass
    else:
        os.makedirs(saving_dir)

    if path.isdir(args.log_dir):
        pass
    else:
        os.makedirs(args.log_dir)
    
    if path.isdir(args.model_dir):
        pass
    else:
        os.makedirs(args.model_dir)

    log = open(logging_fn, 'w')
    logging.basicConfig(level=logging.DEBUG if args.debug == 1 else logging.INFO,
                        format="[%(filename)s:%(lineno)s%(funcName)s()] -> %(message)s",
                        handlers=[logging.FileHandler(logging_fn, mode='w'),
                                  logging.StreamHandler()]
                        )

    logging.info('python ' + ''.join(sys.argv))
    logging.info('==============================')
    

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    logging.info('Model save at = {}, Running on {}, seed={}, ada={}, k={}'.format(model_saving_fn,device,args.seed,args.ada,args.k))
    # criterion = nn.HuberLoss('mean', 1)
    criterion = nn.MSELoss()

    if args.dataset == "mel":
        Td = 96
        his_len = 8
        pred_len = 8
        node_num = 182
        dat_dir = "Dataset/Mel/"
        f_dir = "data/Mel/"
        region_g_dir = 'data/region_graph/Mel_50/'
        feat_fn = dat_dir + "Mel_3month_nonzero.npy"
        sensor_ids_fn = dat_dir + "Mel_locations.npy"

        unknow_size = int(args.unknown_ratio*node_num)
        valid_size = int(0.1*node_num)

        unknown_set_fn = dat_dir + "unknow_continous_nodes_" + str(unknow_size) + "_" + str(args.split_type) + ".npy"
        valid_set_fn = dat_dir + "valid_continous_nodes_" + str(unknow_size) +"_"+ str(valid_size) +"_"+str(args.split_type)+".npy"

        feat, sensor_ids, sensor_dist, adj_s = load_mel_data(feat_fn, sensor_ids_fn, "gaussian")
        _, _, _, adj_s_sg = load_mel_data(feat_fn, sensor_ids_fn, "gaussian", args.a_sg_nk)

    elif args.dataset == "air":
        Td = 24
        his_len = 24
        pred_len = 24
        node_num = 63
        dat_dir = "Dataset/AirQ/"
        f_dir = "data/AirQ/"
        region_g_dir = 'data/region_graph/AirQ_500/'
        feat_fn = dat_dir + "data_air.npy"
        sensor_ids_fn = dat_dir + "graph_sensor_location_air.csv"
        
        unknow_size = int(args.unknown_ratio*node_num)
        valid_size = int(0.1*node_num)
        unknown_set_fn = dat_dir + "unknow_continous_nodes_" + str(unknow_size) + "_"+str(args.split_type)+".npy"
        valid_set_fn = dat_dir + "valid_continous_nodes_" + str(unknow_size) +"_"+ str(valid_size) +"_"+str(args.split_type)+".npy"

        feat, sensor_ids, sensor_dist, adj_s = load_air(feat_fn, sensor_ids_fn, normalized_k=0.05)
        _, _, _, adj_s_sg = load_air(feat_fn, sensor_ids_fn, args.a_sg_nk)

    elif args.dataset == "bay":
        Td = 288
        his_len = 24
        pred_len = 24
        node_num = 325
        dat_dir = "Dataset/PEMS_Bay/"
        f_dir = "data/PEMS_Bay/"
        region_g_dir = 'data/region_graph/PEMS_BAY_200/'
        feat_fn = dat_dir + "pems-bay.h5"
        sensor_ids_fn = dat_dir + "graph_sensor_locations_bay.csv"

        unknow_size = int(args.unknown_ratio*node_num)
        valid_size = int(0.1*node_num)
        unknown_set_fn = dat_dir + "unknow_continous_nodes_" + str(unknow_size) + "_"+str(args.split_type)+".npy"
        valid_set_fn = dat_dir + "valid_continous_nodes_" + str(unknow_size) +"_"+ str(valid_size) +"_"+str(args.split_type)+".npy"

        feat, sensor_ids, sensor_dist, adj_s = load_pems_data(feat_fn, sensor_ids_fn, normalized_k=0.05)
        _, _, _, adj_s_sg = load_pems_data(feat_fn, sensor_ids_fn, args.a_sg_nk)

    elif args.dataset == "pems07":
        Td = 288
        his_len = 24
        pred_len = 24
        node_num = 400
        dat_dir = "Dataset/PEMS07/"
        f_dir = "data/PEMS07/"
        region_g_dir = 'data/region_graph/PEMS07_500/'
        feat_fn = dat_dir + "pems07_data.npy"
        sensor_ids_fn = dat_dir + "graph_sensor_location_07.csv"
        
        unknow_size = int(args.unknown_ratio*node_num)
        valid_size = int(0.1*node_num)
        unknown_set_fn = dat_dir + "unknow_continous_nodes_" + str(unknow_size) + "_"+str(args.split_type)+".npy"
        valid_set_fn = dat_dir + "valid_continous_nodes_" + str(unknow_size) +"_"+ str(valid_size) +"_"+str(args.split_type)+".npy"

        feat, sensor_ids, sensor_dist, adj_s = load_pems_data_np(feat_fn, sensor_ids_fn, normalized_k=0.05)
        _, _, _, adj_s_sg = load_pems_data_np(feat_fn, sensor_ids_fn, args.a_sg_nk)

    elif args.dataset == "pems08":
        Td = 288
        his_len = 24
        pred_len = 24
        node_num = 400
        dat_dir = "Dataset/PEMS08/"
        f_dir = "data/PEMS08/"
        region_g_dir = 'data/region_graph/PEMS08_500/'
        feat_fn = dat_dir + "pems08_data.npy"
        sensor_ids_fn = dat_dir + "graph_sensor_location_08.csv"
        
        unknow_size = int(args.unknown_ratio*node_num)
        valid_size = int(0.1*node_num)
        unknown_set_fn = dat_dir + "unknow_continous_nodes_" + str(unknow_size) + "_"+str(args.split_type)+".npy"
        valid_set_fn = dat_dir + "valid_continous_nodes_" + str(unknow_size) +"_"+ str(valid_size) +"_"+str(args.split_type)+".npy"

        feat, sensor_ids, sensor_dist, adj_s = load_pems_data_np(feat_fn, sensor_ids_fn, normalized_k=0.05)
        _, _, _, adj_s_sg = load_pems_data_np(feat_fn, sensor_ids_fn, args.a_sg_nk)
    
    
    logging.debug("spatial adj for NN:")
    logging.debug(adj_s)
    logging.debug("spatial adj of subgraph:")
    logging.debug(adj_s_sg)


    if path.isdir(f_dir):
        pass
    else:
        os.makedirs(f_dir)

    # construct unobserved, observed and valid set
    unknow_set = np.sort(np.load(unknown_set_fn))
    unknow_list = list(unknow_set)
    unknow_list.sort()
    unknow_set = set(unknow_set)

    valid_set = np.sort(np.load(valid_set_fn))
    valid_list = list(valid_set)
    valid_list.sort()
    valid_set = set(valid_set)

    full_set = set(range(0, node_num))
    know_set = full_set - unknow_set - valid_set
    know_list = list(know_set)
    know_list.sort()

    know_valid = know_list + valid_list
    know_valid.sort()
    valid_idx_pos = []
    known_idx_pos = []
    for i in range(len(know_valid)):
        if know_valid[i] in valid_list:
            valid_idx_pos.append(i)
        else:
            known_idx_pos.append(i)

    total_day = int(feat.shape[1] / Td)
    TE = gen_TE(Td, total_day)
    feat = feat[:, :total_day * Td]
    feat_valid = feat[know_valid, :]

    logging.info("dataset = {}, node num = {}, knowset.size = {}, split type = {}".format(args.dataset,node_num,len(know_list),args.split_type))

    sensor_dist_valid = sensor_dist[know_valid, :][:, know_valid]
    e_feat_t = gen_fake_val_weighed_mel(feat, sensor_dist, unknow_set)
    e_valid_feat_t = gen_fake_val_weighed_mel(feat_valid, sensor_dist_valid, set(valid_idx_pos))

    # time intervals per day
    train_ratio = 0.7
    total_day = int(feat.shape[1] / Td)
    train_day = int(total_day * train_ratio)
    train_length = train_day * Td
    if args.dataset == 'mel':
        sample_length = train_length
    else:
        sample_length = 7 * Td
    K_dtw_k = 1
    K_dtw_u = 1
    observed_ratio = 0.4

    # used to test
    logging.info("Feat shape = {} TE shape = {}".format(feat.shape, TE.shape))
    if args.dataset != 'mel':
        _, test_dataloader_t = generate_dataset_tatt_no_slide_cl(e_feat_t, feat, TE, train_length, his_len, pred_len,
                                                                 args.batch_size)
        # used to valid
        _, valid_dataloader_t = generate_dataset_tatt_no_slide_cl(e_valid_feat_t, feat_valid, TE, train_length, his_len,
                                                                  pred_len, args.batch_size)
    else:
        _, test_dataloader_t = generate_dataset_tatt_cl(e_feat_t, feat, TE, train_length, his_len, pred_len,
                                                        args.batch_size)
        # used to valid
        _, valid_dataloader_t = generate_dataset_tatt_cl(e_valid_feat_t, feat_valid, TE, train_length, his_len,
                                                         pred_len, args.batch_size)

    for i in know_list:
        for j in unknow_list:
            adj_s[i, j] = 0

    if path.isfile(
            f_dir + "Dtw_similarity_continous_nodes_" + str(len(know_valid)) + "_kk_" + str(args.split_type) + ".npy"):
        A_dtw_know = np.load(
            f_dir + "Dtw_similarity_continous_nodes_" + str(len(know_valid)) + "_kk_" + str(args.split_type) + ".npy")
    else:
        A_dtw_know = gen_dtw_adj(feat[:, :sample_length], Td, train_day, unknow_list, know_valid, "know")
        np.save(
            f_dir + "Dtw_similarity_continous_nodes_" + str(len(know_valid)) + "_kk_" + str(args.split_type) + ".npy",
            A_dtw_know)
    if path.isfile(f_dir + "Adj_dtw_continous_nodes_" + str(len(know_valid)) + "_kk_" + str(args.split_type) + ".npy"):
        W_dtw_know = np.load(
            f_dir + "Adj_dtw_continous_nodes_" + str(len(know_valid)) + "_kk_" + str(args.split_type) + ".npy")
    else:
        W_dtw_know = gen_temporal_adj(A_dtw_know, K_dtw_k, unknow_set, "know")
        np.save(f_dir + "Adj_dtw_continous_nodes_" + str(len(know_valid)) + "_kk_" + str(args.split_type) + ".npy",
                W_dtw_know)

    # the temporal rela between know and unknow
    if path.isfile(
            f_dir + "Dtw_similarity_continous_nodes_" + str(len(know_valid)) + "_ku_" + str(args.split_type) + ".npy"):
        A_dtw_unknow = np.load(
            f_dir + "Dtw_similarity_continous_nodes_" + str(len(know_valid)) + "_ku_" + str(args.split_type) + ".npy")
    else:
        A_dtw_unknow = gen_dtw_adj(e_feat_t[:, :sample_length], Td, train_day, unknow_list, know_valid, "unknow")
        np.save(
            f_dir + "Dtw_similarity_continous_nodes_" + str(len(know_valid)) + "_ku_" + str(args.split_type) + ".npy",
            A_dtw_unknow)
    if path.isfile(f_dir + "Adj_dtw_continous_nodes_" + str(len(know_valid)) + "_ku_" + str(args.split_type) + ".npy"):
        W_dtw_unknow = np.load(
            f_dir + "Adj_dtw_continous_nodes_" + str(len(know_valid)) + "_ku_" + str(args.split_type) + ".npy")
    else:
        W_dtw_unknow = gen_temporal_adj(A_dtw_unknow, K_dtw_u, unknow_set, "unknow")
        np.save(f_dir + "Adj_dtw_continous_nodes_" + str(len(know_valid)) + "_ku_" + str(args.split_type) + ".npy",
                W_dtw_unknow)

    # ignore the unkown to know
    W_dtw_unknow[know_list, :] = 0
    W_dtw = W_dtw_unknow + W_dtw_know
    for i in range(W_dtw.shape[0]):
        W_dtw[i, i] = 1

    # different model filters  
    if args.dataset == "mel":
        filters = [[64], [64], [64]]
    else:
        A_s = torch.from_numpy(adj_s.astype(np.float32))
        filters = [[64, 64], [64, 64]]

    logging.info("GNN Filters")
    logging.info(filters)
    A_s = get_normalized_weighted_adj(adj_s)
    A_s = A_s.to(device)
    
    A_t = get_normalized_connective_adj(W_dtw)
    A_t = A_t.to(device)

    # generate the W_dtw for valiadation
    W_dtw_know_valid = np.copy(W_dtw_know[know_valid, :][:, know_valid])
    W_dtw_know_valid[:, valid_idx_pos] = 0
    W_dtw_know_valid[valid_idx_pos, :] = 0
    # After masking locations, ought to update the adj_temporal for them
    A_dtw_valid = gen_dtw_adj(e_valid_feat_t[:, :sample_length], Td, train_day, valid_idx_pos, known_idx_pos, "unknow")
    W_dtw_valid = gen_temporal_adj(A_dtw_valid, K_dtw_u, set(valid_idx_pos), "unknow")
    W_dtw_valid[known_idx_pos, :] = 0
    W_dtw_know_valid = W_dtw_valid + W_dtw_know_valid
    for i in range(W_dtw_know_valid.shape[0]):
        W_dtw_know_valid[i, i] = 1
    A_t_know_valid = get_normalized_connective_adj(W_dtw_know_valid)
    A_t_know_valid = A_t_know_valid.to(device)

    adj_s_valid = adj_s[know_valid, :][:, know_valid]
    for i in known_idx_pos:
        for j in valid_idx_pos:
            adj_s_valid[i, j] = 0
    A_s_valid = get_normalized_weighted_adj(adj_s_valid)
    A_s_valid = A_s_valid.to(device)

    # generate sub-graphs for each location
    adj_s_sg_o = adj_s_sg[know_list, :][:, know_list]
    khop = 1
    neighbourhood_size = []
    neighbourhood = []
    for i in range(len(know_list)):
        khop_neighbours = select_khop_neighbour(adj_s_sg_o, [i], khop, know_list)
        neighbourhood.append(khop_neighbours)
        neighbourhood_size.append(len(khop_neighbours))
    neighbourhood_size_avg = sum(neighbourhood_size) / len(neighbourhood_size)
    logging.debug("Neighbourhood Size = {}".format(neighbourhood_size))
    logging.info("Average Neighbourhood Size = {}".format(neighbourhood_size_avg))

    # Select the masking node through region similarity: region-loc; sum socre
    poi_feat_list = []
    loc_feat_list = []
    logging.info('==============================')
    logging.info('Begin Compute Location Representation and Sim Score')
    for i in range(node_num):
        location_feat = np.load(region_g_dir + str(i) + ".npy")
        region_feat = location_feat[:,:32].astype(np.float32)
        # lon lat
        loc_feat = location_feat[-1, 32:].astype(np.float32)
        region_feat_sum = np.sum(region_feat, axis=0)

        poi_feat_list.append(region_feat_sum)
        loc_feat_list.append(loc_feat)

    # merge the locations representation in unknown region
    unknow_region_feat = np.zeros(poi_feat_list[0].shape)
    unknow_loc_feat = np.zeros(loc_feat_list[0].shape)

    for i in unknow_list:
        unknow_region_feat += poi_feat_list[i]
        unknow_loc_feat += loc_feat_list[i]
    unkown_region_feat = unknow_region_feat / len(unknow_list)
    unknow_loc_feat = unknow_loc_feat / len(unknow_list)

    # compute the sub-graph's representation
    sg_feat = []
    sg_score_region = np.zeros(len(know_list))
    sg_score_dist = np.zeros(len(know_list))
    for i in range(len(know_list)):
        sg_neighbourhood = neighbourhood[i]
        tmp_feat = np.zeros(poi_feat_list[0].shape)
        tmp_loc = np.zeros(loc_feat_list[0].shape)
        for f in sg_neighbourhood:
            tmp_feat += poi_feat_list[f]
            tmp_loc += loc_feat_list[f]
        if len(sg_neighbourhood) != 0:
            tmp_feat = tmp_feat / len(sg_neighbourhood)
            tmp_loc = tmp_loc / len(sg_neighbourhood)
        sg_score_region[i] = cosine_similarity(tmp_feat.reshape(1, -1), unkown_region_feat.reshape(1, -1))
        sg_score_dist[i] = 1 / (Geodesic.WGS84.Inverse(tmp_loc[1], tmp_loc[0], unknow_loc_feat[1], unknow_loc_feat[0])[
                                    's12'] / 1000)

    
    ratio = (args.aug_ratio / (neighbourhood_size_avg / len(know_list))) / len(know_list)
    sg_score_region_copy = sg_score_region.copy()
    # sg_score_region_avg = np.mean(sg_score_region)
    sg_score_region_sort = np.argsort(-sg_score_region)
    sg_score_region[sg_score_region_sort[args.k:]] = 0
    weight = ratio / np.mean(sg_score_region)
    sg_score_region = sg_score_region * weight
    sg_score_region = torch.from_numpy(sg_score_region)
    logging.debug("region||road")
    logging.debug(sg_score_region)
    logging.debug(sg_score_region_copy)

    # sg_score_dist_avg = np.mean(sg_score_dist)
    sg_score_dist_copy = sg_score_dist.copy()
    sg_score_dist_sort = np.argsort(-sg_score_dist)
    sg_score_dist[sg_score_dist_sort[args.k:]] = 0
    weight = ratio / np.mean(sg_score_dist)
    sg_score_dist = sg_score_dist * weight
    sg_score_dist = torch.from_numpy(sg_score_dist)
    logging.debug("dist")
    logging.debug(sg_score_dist)
    logging.debug(sg_score_dist_copy)


    sg_score = (sg_score_region + sg_score_dist) / 2
    logging.info("Finial Using sg_score=(sg_score_region + sg_score_dist) / 2")
    logging.info(sg_score)

    feat_o = feat[know_list, :]
    adj_s_o = adj_s[know_list, :][:, know_list]
    W_dtw_know_o = W_dtw_know[know_list, :][:, know_list]
    A_s_o = A_s[know_list, :][:, know_list]
    observed_list = [i for i in range(len(know_list))]
    A_dtw_unknow_o = A_dtw_unknow[know_list, :][:, know_list]
    sensor_dist_o = sensor_dist[know_list, :][:, know_list]

    model_predict = USPGCN_MultiD_Inductive_Tattr(his_len, 64, filters, True, device, output_length=pred_len)
    model_predict = model_predict.to(device)
    best_model = model_predict
    best_rmse = 10000
    rand = np.random.RandomState(args.seed)  # Fixed random output
    lweight = torch.tensor(args.lweight)
    params = list(model_predict.parameters())
    optimizer_predict = torch.optim.Adam(params, lr=args.lr)
    logging.info("Initial weight = {} tempeture={}".format(lweight, args.tempe))
    total_sg_score_d = 0
    total_sg_score_r = 0
    total_sg_num = 0
    max_sg_ids = list(np.array(torch.where(sg_score>0)[0]))
    max_masked_ids = []
    for sg_id in max_sg_ids:
        max_masked_ids += neighbourhood[sg_id]
    max_masked_ids = list(set(max_masked_ids))
    logging.info('max masked node num = {}'.format(len(max_masked_ids)))

    start = time.time()
    for epoch in range(args.epochs):
        logging.info("=====Epoch {}=====".format(epoch))
        logging.debug('Preprocessing...')
        # remove masked locations relation from know locations' matrix
        W_dtw_know_cur = np.copy(W_dtw_know_o)

        cur_masked_ids = []
        while len(cur_masked_ids) < int(len(know_list) * (args.aug_ratio)):
            if args.ada == 1:
                cur_sg_ids = list(np.array(torch.where(torch.bernoulli(sg_score) == 1)[0]))
            else:
                masked_num = int((len(know_list)* (args.aug_ratio))/neighbourhood_size_avg)
                if masked_num<1:
                    masked_num = 1
                cur_sg_ids =list(rand.choice(list(range(0,len(know_list))),masked_num,replace=False))
            total_sg_score_r += sum(sg_score_region_copy[cur_sg_ids])
            total_sg_score_d += sum(sg_score_dist_copy[cur_sg_ids])
            total_sg_num += len(cur_sg_ids)

            for sg_id in cur_sg_ids:
                cur_masked_ids += neighbourhood[sg_id]
                cur_masked_ids = list(set(cur_masked_ids))
                
                if len(cur_masked_ids) >= int(len(know_list) * (args.aug_ratio)):
                    break
            if len(cur_masked_ids) == len(max_masked_ids):
                break

        if len(cur_masked_ids) == len(know_list):
            del cur_masked_ids[int(len(know_list) * args.aug_ratio):len(cur_masked_ids)]
        logging.info('masked node num = {}'.format(len(cur_masked_ids)))

        W_dtw_know_cur[:, cur_masked_ids] = 0
        W_dtw_know_cur[cur_masked_ids, :] = 0

        observed_list_cur = list(set(observed_list) - set(cur_masked_ids))

        # the features input into model
        logging.debug('Generating pseudo observations ....')
        m_feat = gen_fake_val_weighed_mel(feat_o, sensor_dist_o, set(cur_masked_ids))
        
        # After masking locations, ought to update the adj_temporal for them
        A_dtw_mask = gen_dtw_adj(m_feat[:, :sample_length], Td, train_day, cur_masked_ids, observed_list_cur, "unknow")
        W_dtw_mask = gen_temporal_adj(A_dtw_mask, K_dtw_u, set(cur_masked_ids), "unknow")
        W_dtw_mask[observed_list_cur, :] = 0
        W_dtw_cur = W_dtw_mask + W_dtw_know_cur
        for i in range(W_dtw_cur.shape[0]):
            W_dtw_cur[i, i] = 1
        A_t_cur = get_normalized_connective_adj(W_dtw_cur)
        A_t_cur = A_t_cur.to(device)

        adj_s_cur = np.copy(adj_s_o)
        for i in observed_list_cur:
            for j in cur_masked_ids:
                adj_s_cur[i, j] = 0
        adj_s_cur = get_normalized_weighted_adj(adj_s_cur)
        adj_s_cur = adj_s_cur.to(device)

        if args.dataset != 'mel':
            m_train_dataloader_m, m_test_dataloader_m = generate_dataset_tatt_no_slide_cl(m_feat, feat_o, TE,
                                                                                          train_length, his_len,
                                                                                          pred_len, args.batch_size)
        else:
            m_train_dataloader_m, m_test_dataloader_m = generate_dataset_tatt_cl(m_feat, feat_o, TE, train_length,
                                                                                 his_len, pred_len, args.batch_size)
        logging.debug('Training...')
        loss, ploss, dloss = train_predict(m_train_dataloader_m, model_predict, optimizer_predict, criterion, device,
                                           adj_s_cur, A_t_cur, observed_list_cur, cur_masked_ids, his_len, args.tempe,
                                           lweight)
        logging.debug('Evaluating...')

        valid_rmse1, valid_mae1, valid_mape1, valid_r2 = test_predict(valid_dataloader_t, model_predict, device,
                                                                      A_s_valid, A_t_know_valid, known_idx_pos,
                                                                      valid_idx_pos, his_len)
        logging.info(f'##Training## loss: {loss}, prediction loss: {ploss}, CL loss {dloss}\n' +
              f'##Validation## rmse loss: {valid_rmse1}, mae loss: {valid_mae1}, mape loss: {valid_mape1}, r2 loss: {valid_r2}')

        if valid_rmse1 < best_rmse:
            best_rmse = valid_rmse1
            logging.info("best rmse = {:.4f}\n".format(best_rmse))
            best_list = [loss, valid_rmse1, valid_mae1, valid_mape1]
            best_epoch = epoch
            torch.save(model_predict.state_dict(),model_saving_fn)
        if epoch - best_epoch > args.patience:
            logging.info("early stop at epoch {}".format(epoch))
            break
    
    end = time.time()
    logging.info("Finish training! Final weight = {}, tempeture = {}".format(lweight, args.tempe))
    logging.info("select sg num = {}, avg select sg region score = {:.6f}, avg sg distance score = {:.6f}, avg sg score = {:.6f}".format(total_sg_num, total_sg_score_r/total_sg_num, total_sg_score_d/total_sg_num, (total_sg_score_d + total_sg_score_r)/total_sg_num))
    model_predict.load_state_dict(
        torch.load(model_saving_fn))
    valid_rmse1, valid_mae1, valid_mape1, valid_r2 = test_predict(valid_dataloader_t, model_predict, device, A_s_valid,
                                                                  A_t_know_valid, known_idx_pos, valid_idx_pos, his_len)
    start_test = time.time()
    test_rmse1, test_mae1, test_mape1, test_r2 = test_predict(test_dataloader_t, model_predict, device, A_s, A_t,
                                                              know_valid, unknow_list, his_len)

    end_test = time.time()
    
    logging.info('\n##Best Epoch## {}'.format(best_epoch))
    logging.info('##on train data## loss: {:.4f}'.format(best_list[0]))
    logging.info('##on valid data predict## rmse loss: {:.4f}, mae loss: {:.4f}, mape loss: {:.4f}, r2 loss: {:.4f}\n'.format(valid_rmse1,valid_mae1,valid_mape1,valid_r2))
    logging.info('##on test data predict## rmse loss: {:.4f}, mae loss: {:.4f}, mape loss: {:.4f}, r2 loss: {:.4f}\n'.format(test_rmse1,test_mae1,test_mape1,test_r2))
    logging.info('training time: {:.1f}h'.format(((end - start))/60/60))
    logging.info('testing time: {:.1f}s'.format(((end_test - start_test))))