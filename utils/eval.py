import numpy as np

def metric(truth,pred):
    idx = truth > 0
    RMSE = np.sqrt(np.mean((pred[idx] - truth[idx]) ** 2))
    MAE = np.mean(np.abs(pred[idx] - truth[idx]))
    MAPE = np.mean(np.abs(pred[idx] - truth[idx]) / truth[idx])
    R2 = 1 - np.mean((pred[idx] - truth[idx]) ** 2) / \
         np.mean((truth - np.mean(truth)) ** 2)
    return RMSE, MAE, MAPE, R2
