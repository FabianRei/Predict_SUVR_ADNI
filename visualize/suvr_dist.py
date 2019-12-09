import h5py
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import seaborn as sns
import os
import pickle

suvr_data_anal = True

if suvr_data_anal:
    fpath = r'C:\Users\Fabian\stanford\fed_learning\rsync\slice_data_subj.h5'
    data = h5py.File(fpath, 'r')
    scanners = []
    amyloid = []
    rcf = []
    img_ids = []
    names = []
    slices = []
    site = []
    suvr = []
    subject_ids = []

    for k, v in data.items():
        # scanners.append(f"{v['manufacturer']}, {v['model']}")
        v = v.attrs
        amyloid.append(v['label_amyloid'])
        # rcf.append([v['rows'], v['columns'], v['frames']])
        # img_ids.append(v['img_id'])
        # names.append(k)
        # slices.append(v['slices'])
        site.append(v['site'])
        suvr.append(v['label_suvr'])
        # subject_ids.append(v['rid'])


    site = np.array(site)
    # names = np.array(names)
    # slices = np.array(slices)
    # img_ids = np.array(img_ids)
    # scanners = np.array(scanners)
    amyloid = np.array(amyloid)
    # rcf = np.array(rcf)
    # subject_ids = np.array(subject_ids)
    suvr = np.array(suvr)
    density = gaussian_kde(suvr)
    xs = np.linspace(0.7,2,400)
    density.covariance_factor = lambda : .1
    density._compute_covariance()
    plt.plot(xs,density(xs))
    plt.clf()
    plt.grid(which='both')
    plt.hist(suvr, bins=200)
    fname = os.path.join(os.path.dirname(fpath), 'SVUR_dist.png')
    plt.savefig(fname, dpi=200)
    plt.show()
    print('nice')
    sns.set_style('whitegrid')
    sns.kdeplot(suvr)
    print('nice')


pckl_file = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\incl_subjects_site_one_slices_dataset_full\epoch_29_pred_labels_train_test_epoch_08-30_17-48_lr_0_001_pretrained_reg_30epochs_rod_0_1_da_10.p'
with open(pckl_file, 'rb') as f:
    tt = pickle.load(f)
test = tt['test']
train = tt['train']
print('nice')