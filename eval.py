import os, cv2
import numpy as np
import torch
import tqdm
import glob
import math


def RMSE(label,pre):
    return np.sqrt(((pre - label) ** 2).mean())

def BER(y_actual, y_hat):
    y_hat = y_hat.ge(128).float()
    y_actual = y_actual.ge(128).float()

    y_actual = y_actual.squeeze(1)
    y_hat = y_hat.squeeze(1)

    #output==1
    pred_p=y_hat.eq(1).float()
    #print(pred_p)
    #output==0
    pred_n = y_hat.eq(0).float()
    #print(pred_n)
    #TP
    tp_mat = torch.eq(pred_p,y_actual)
    TP = float(tp_mat.sum())

    #FN
    fn_mat = torch.eq(pred_n, y_actual)
    FN = float(fn_mat.sum())

    # FP
    fp_mat = torch.ne(y_actual, pred_p)
    FP = float(fp_mat.sum())

    # TN
    fn_mat = torch.ne(y_actual, pred_n)
    TN = float(fn_mat.sum())


    BER=1-0.5*(TP/(TP+FN)+TN/(TN+FP))


    return BER




if __name__ == "__main__":
    pre_list = glob.glob('BER/ISTD/myTest/*.*')
    X=0
    Y=0
    Z=0
    j=0
    for i, path in tqdm.tqdm(enumerate(pre_list)):

        pre = cv2.imread(path)
        label = cv2.imread(path.replace('myTest', 'label'))


        ber= BER(torch.from_numpy(label).float(), torch.from_numpy(pre).float())
        #mse=RMSE(label, pre)
        #Y=Y+mse
        X=X+ber
        j=j+1


    BER=X*100/j
    #rmse=Y/j



    print(BER)


