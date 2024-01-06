import numpy as np


def gen_fake_val_weighed_mel(feat, dist_mat, unknow_set):
    distance = np.where(dist_mat<dist_mat.transpose(),dist_mat,dist_mat.transpose())
    mask = np.ones(feat.shape)
    mask[list(unknow_set),:] = 0
    m_feat = feat*mask
    
    while len(unknow_set) != 0:
        imputation_list = []
        for sid in unknow_set:
            # all nodes
            neighbour_idx = np.where(distance[sid,:] != 0)[0]
            # known nodes
            neighbour_set = set(neighbour_idx) - unknow_set
            neighbour_list = list(neighbour_set)
            neighbour_num = len(neighbour_list)
            # known nodes dist
            n_dist = distance[sid,neighbour_list]
            if neighbour_num==0:
                continue
            # all known neighbours
            n_dist_norm = (1/n_dist)/np.sum((1/n_dist))
            fake_val = np.zeros(feat.shape[1])
            for i in range(len(n_dist_norm)):
                fake_val += m_feat[neighbour_list[i],:]*n_dist_norm[i]
            m_feat[sid,:] = fake_val
            imputation_list.append(sid)
        unknow_set=unknow_set-set(imputation_list)

    return m_feat
