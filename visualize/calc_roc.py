import pickle
import numpy as np
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import os
from glob import glob
import re
from dl.data.project_logging import CsvWriter


def calc_rocs_epoch(folder, epoch=29, bins=-1):
    f_paths = glob(f'{folder}\*epoch_{epoch}_pred_labels*.p')
    if bins != 0:
        f_paths = [p for p in f_paths if re.match(r'.*20bins.*', p) is not None]
    else:
        f_paths = [p for p in f_paths if re.match(r'.*reg.*|.*20bins.*', p) is None]
    for fp in f_paths:
        with open(fp, 'rb') as f:
            tt = pickle.load(f)
            tt = tt['test']
        fname = fp[:-2] + "_roc_bins.png"
        if bins != -1:
            new_tt = np.zeros((tt.shape[0], 4))
            new_tt[:, [0, 1]] = (tt[:, [0, 1]] >= 10).astype(np.float64)
            middle_cut = int((tt.shape[-1]-2)/2+2)
            new_tt[:, 2] = np.sum(tt[:, 2:middle_cut], axis=1)
            new_tt[:,3] = np.sum(tt[:, middle_cut:], axis=1)
            tt = new_tt

        calc_roc(tt, fname)


def calc_roc_auc(tt):
    pred = tt[:, 3]
    lab = tt[:, 1]
    fpr, tpr, thres = roc_curve(lab, pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def calc_roc(tt, fname):
    pred = tt[:,3]
    lab = tt[:,1]
    fpr, tpr, thres = roc_curve(lab, pred)
    roc_auc = auc(fpr, tpr)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(which='both')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(fname, dpi=200)
    # (os.path.join(out_path, f'{fname}.png')
    plt.clf()


def create_number_csv(folder, epoch=29, bins=-1):
    f_paths = glob(f'{folder}\*epoch_{epoch}_pred_labels*.p')
    if bins != 0:
        f_paths = [p for p in f_paths if re.match(r'.*20bins.*', p) is not None]
    else:
        f_paths = [p for p in f_paths if re.match(r'.*reg.*|.*20bins.*', p) is None]
    writer = CsvWriter(os.path.join(folder, 'key_values_more.csv'), header=['name', 'sensitivity', 'specificity', 'accuracy', 'youdens_i', 'youd_sensitivity', 'youd_specificity', 'youd_threshold'])
    for fp in f_paths:
        with open(fp, 'rb') as f:
            tt = pickle.load(f)
            tt = tt['test']
        if bins != -1:
            new_tt = np.zeros((tt.shape[0], 4))
            new_tt[:, [0, 1]] = (tt[:, [0, 1]] >= 10).astype(np.float64)
            middle_cut = int((tt.shape[-1]-2)/2+2)
            new_tt[:, 2] = np.sum(tt[:, 2:middle_cut], axis=1)
            new_tt[:,3] = np.sum(tt[:, middle_cut:], axis=1)
            tt = new_tt
        fname = os.path.basename(fp)[:-2]
        spezi = calc_specificity(tt)
        sensi = calc_sensitivity(tt)
        acc = np.mean(tt[:,0] == tt[:,1])
        j_score, threshold, sensitivity, specificity = cutoff_youdens_j_tt(tt)
        writer.write_row(name=fname, sensitivity=sensi, specificity=spezi, accuracy=acc, youdens_i=j_score, youd_sensitivity=sensitivity, youd_specificity=specificity, youd_threshold=threshold)


def calc_specificity(tt):
    pred = tt[:, 0]
    lab = tt[:, 1]
    specificity = np.sum((pred == 0) & (lab == 0))/(np.sum((pred == 0) & (lab == 0))+np.sum((pred == 1) & (lab == 0)))
    return specificity


def calc_ppv(tt):
    pred = tt[:, 0]
    lab = tt[:, 1]
    return np.sum((pred == 1) & (lab == 1))/np.sum(pred==1)

def calc_npv(tt):
    pred = tt[:, 0]
    lab = tt[:, 1]
    return np.sum((pred == 0) & (lab == 0))/np.sum(pred==0)

def cutoff_youdens_j_tt(tt):
    pred = tt[:,3]
    lab = tt[:,1]
    fpr, tpr, thres = roc_curve(lab, pred)
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thres))
    j_score = j_ordered[-1][0]
    threshold = j_ordered[-1][1]
    #use standard thresholds
    if tt[:,0].max() > 1.1:
        tt[:, 0] = pred > 1.11
    else:
        tt[:, 0] = pred >= 0.5
    #tt[:,0] = pred >= threshold
    sensitivity = calc_sensitivity(tt)
    specificity = calc_specificity(tt)
    roc_auc = calc_roc_auc(tt)
    ppv = calc_ppv(tt)
    npv = calc_npv(tt)
    return j_score, threshold, sensitivity, specificity, roc_auc, ppv, npv


def cutoff_youdens_j(fp, bins=-1):
    with open(fp, 'rb') as f:
        tt = pickle.load(f)
        tt = tt['test']
    pred = tt[:,3]
    lab = tt[:,1]
    fpr, tpr, thres = roc_curve(lab, pred)
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thres))
    return j_ordered[-1]


def calc_sensitivity(tt):
    pred = tt[:, 0]
    lab = tt[:, 1]
    sensitivity = np.sum((pred == 1) & (lab == 1))/(np.sum((pred == 1) & (lab == 1))+np.sum((pred == 0) & (lab == 1)))
    return sensitivity


# if __name__ == '__main__':
#     folder = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\incl_subjects_site_one_slices_dataset_full'
#     calc_rocs_epoch(folder, bins=20)
#
#     print('nice')
if __name__ == '__main__':
    # calculate ROC
    fpath = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments'
    paths = glob(fpath + r"\*site*")
    for p in paths:
        create_number_csv(p, bins=20)
        calc_rocs_epoch(p, bins=20)




r'''
    f_paths = glob(f'{folder}\*epoch_{29}_pred_labels*.p')
    f_paths1 = [p for p in f_paths if re.match(r'.*reg.*|.*20bins.*', p) is None]
    # for fp in f_paths1:
    #     cutoff_youdens_j(fp)
    print('nice')
    f_paths2 = [p for p in f_paths if re.match(r'.*20bins.*', p) is  not None]
    for fp in f_paths2:
        calc_rocs_epoch(fp, bins=20)
########################################################################
if __name__ == '__main__':
    fpath = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments'
    paths = glob(fpath + r"\*site*")
    for p in paths:
        # create_number_csv(p)
        calc_rocs_epoch(p)
'''