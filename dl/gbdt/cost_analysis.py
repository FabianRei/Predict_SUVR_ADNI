from glob import glob
import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error


def rearrange_pred_labs(preds, labs, train_test_split):
    preds = [list(p) for p in preds]
    labs = [list(l) for l in labs]
    results_pred = []
    results_lab = []
    for s in train_test_split:
        results_pred.append(preds[s].pop(0))
        results_lab.append(labs[s].pop(0))
    results_pred = np.array(results_pred)
    results_lab = np.array(results_lab)
    return results_pred, results_lab


def get_top_subjects(number):
    global subs
    global preds
    global labs
    perc_lab = np.percentile(labs, 100 - number*100/len(subs))
    subs_lab = subs[labs>perc_lab]
    count_lab = number
    while len(np.unique(subs_lab)) < number:
        count_lab += 1
        perc_lab = np.percentile(labs, 100 - count_lab*100/len(subs))
        subs_lab = subs[labs>perc_lab]
    perc_pred = np.percentile(preds, 100 - number*100/len(subs))
    subs_pred = subs[preds>perc_pred]
    count_pred = number
    while len(np.unique(subs_pred)) < number:
        count_pred += 1
        perc_pred = np.percentile(preds, 100 - count_pred*100/len(subs))
        subs_pred = subs[preds>perc_pred]
    # percentile might get bigger number, if both have the dame delta SUVR pred/lab
    return perc_pred, perc_lab


def get_unique_t0_suvr():
    global t0_suvr
    global subs
    t0, t0_idx = np.unique(subs, return_index=True)
    suvr_unique = t0_suvr[t0_idx]
    return suvr_unique


def get_matches_target(study_size, target_size, detailled=False):
    # returns filter label, filter pred & filter combinded
    global preds
    global labs
    global subs
    perc_pred, perc_lab = get_top_subjects(study_size)
    _, perc_lab = get_top_subjects(target_size)
    median_lab = np.percentile(labs, 50)
    sub_ids = np.unique(subs[(labs > perc_lab) & (preds > perc_pred)])
    sub_ids_study, study_idx = np.unique(subs[preds>perc_pred], return_index=True)
    sub_ids_label = np.unique(subs[labs>perc_lab])
    print(f"{len(sub_ids_study)} in study, {len(np.unique(sub_ids))} also in top {len(sub_ids_label)} of labels.")
    # print(result)
    # print(len(np.unique(sub_ids)), sub_ids)
    if detailled:
        return (labs>perc_lab), (preds>perc_pred), (labs > perc_lab) & (preds > perc_pred)
    else:
        return(labs > perc_lab) & (preds > perc_pred)


def ch(num, changes_pred, changes):
    print('pred:', sum(changes_pred>num)/len(changes_pred))
    print('all:', sum(changes>num)/len(changes))

def plot_densities():
    global preds
    global labs
    sns.kdeplot(preds, label='GBDT predictions')
    sns.kdeplot(labs, label='Ground truth')
    plt.legend()
    plt.show()


def to_ampos_subjects(filter=None):
    global preds
    global labs
    global t0_suvr
    global subs
    if filter is not None:
        p = preds[filter]
        l = labs[filter]
        t0 = t0_suvr[filter]
        s = subs[filter]
    else:
        # copy, as otherwise might mess up global
        p = np.copy(preds)
        l = np.copy(labs)
        t0 = np.copy(t0_suvr)
        s = np.copy(subs)
    s_res = []
    suvr = t0+l
    pred_suvr = t0+p
    for su in np.unique(s):
        s_filt = su==s
        suvr_f = suvr[s_filt]
        pred_suvr_f = pred_suvr[s_filt]
        t0_f = t0[s_filt]
        if suvr_f.max()>0.79:
            s_res.append(s)
    print(f'out of {len(np.unique(s))} amyloid negative subjects, we find that {len(s_res)} turn to amyloid positive')
    return np.array(s_res)


def get_unique_changes(changes, subjects, subjects2=None, t0_suvr=None, selector='max', delta_time=None, apoe=None):
    # gets max changes for each subject
    if subjects2 is None:
        subjects2 = subjects
    result = []
    filt_idxs = changes == -np.inf
    # include unique t0_suvr results
    if t0_suvr is not None:
        t0_results = []
    else:
        t0_results = None
    if apoe is not None:
        apoe_results = []
    else:
        apoe_results = None
    for s in np.unique(subjects):
        if not s in subjects2:
            continue
        filt = s==subjects
        if selector == 'max':
            # get maximum change
            change = changes[filt].max()
        if selector == 'last':
            delta = delta_time[filt]
            change = changes[filt][delta.argmax()]
        if t0_suvr is not None:
            t0_results.append(t0_suvr[filt].max())
        if apoe is not None:
            apoe_results.append(apoe[filt][0])
        result.append(change)
        # create corresponding filter for maxval only
        filt_vals = filt[filt==True]
        filt_vals[:] = False
        filt_vals[changes[filt].argmax()] = True
        filt[filt == True] = filt_vals
        filt_idxs += filt
    result = np.array(result)
    results = [result, filt_idxs, t0_results, apoe_results]
    results = [np.array(res) for res in results if res is not None]
    return results


def visualize_top_changes():
    global filter_lab
    global filter_pred
    global filter_comb
    global subs
    global preds
    global labs
    global t0_suvr
    global delta_time
    global apoe
    changes_lab, filter_idxs_lab, t0_suvr_lab = get_unique_changes(labs[filter_lab], subs[filter_lab], t0_suvr=t0_suvr[filter_lab])
    changes_pred, filter_idxs_pred, t0_suvr_pred, apoe_pred = get_unique_changes(labs[filter_pred], subs[filter_pred], t0_suvr=t0_suvr[filter_pred], delta_time=delta_time[filter_pred], selector='last', apoe=apoe[filter_pred])
    changes_comb, filter_idxs_comb = get_unique_changes(labs[filter_lab], subs[filter_lab], subs[filter_comb])
    changes, filter_idxs, t0_suvr_ind, apoe_all = get_unique_changes(labs, subs, t0_suvr=t0_suvr, delta_time=delta_time, selector='last', apoe=apoe)
    # filter out top changes that are also predicted
    # chose filter to apply
    appl_filter = filter_lab
    flt = np.copy(appl_filter)
    flt[flt==True] = filter_idxs_lab
    flt2 = np.copy(filter_lab)
    flt2[flt2==True] = filter_idxs_comb
    colors = sns.color_palette()
    plt.hist(labs[flt], color=colors[0], alpha=0.66,
             label=['Subject in study via top 100 amyloid negative GBDT predictions','Max changes of top 100 amyloid negative subjects'])
    sns.distplot((labs[flt2], labs[flt & ~flt2]), stacked=True)
    plt.xlabel('Delta SUVR')
    plt.ylabel('Frequency')

    sns.distplot(changes, color=colors[3])
    print('db')

# results file (pickle) with predictions
gbdt_results = 'C:\\Users\\Fabian\\stanford\\gbdt\\analysis\\1576074362_w_acs__0_03483757920168396\\all_results.pickle'
# file with additional metadata
additional_data = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\rf_data_train_test_crossval.pickle'
# path_detailled_data = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\test_meta_data_complete.pickle'
# diagnostic data:
path_diagnosis_data = 'C:\\Users\\Fabian\\stanford\\diagnoses_DXSUM.pickle'
# load data
with open(gbdt_results, 'rb') as f:
    data = pickle.load(f)
with open(additional_data, 'rb') as f:
    more_data = pickle.load(f)
with open(path_diagnosis_data, 'rb') as f:
    diagnosis_data = pickle.load(f)

# with open(path_detailled_data, 'rb') as f:
#     detailled_data = pickle.load(f)

gbms = data['gbm']
x_names = data['x_names']
preds = data['predictions']
labs = data['labels']
y = data['y']
train_test_split = data['train_test_split']
preds, labs = rearrange_pred_labs(preds, labs, train_test_split)
subs = more_data['subs']
t0_suvr = more_data['t0_suvr']
age = more_data['age']
weight = more_data['age']
apoe = more_data['apoe']
faq = more_data['faqtotal']
mmse = more_data['mmsescore']
delta_time = more_data['delta_time']
# get diagnosis data. Order is same as data above (tested via dia_subs, without t0_data)
dia_subs = diagnosis_data['subs']
diagnoses = diagnosis_data['diagnoses']
t0_diagnoses = diagnosis_data['t0_diagnoses']
dia_delta_time = diagnosis_data['delta_time']
# filter to data points, not t0 only
t0_diagnoses = t0_diagnoses[dia_delta_time>0]
# filter to amyloid -
amneg_filter = t0_suvr<0.79
# filter to amyloid +
ampos_filter = t0_suvr>0.79
# filter to mildly a+
perc_75 = 0.95665
mildly_filter = (t0_suvr>0.79) & (t0_suvr <perc_75)
severe_pos_filter = t0_suvr >= perc_75

all_filter = t0_suvr > -np.inf
apoe_filter = apoe >= 4

# choose filter to apply
curr_filter = ampos_filter
# apply filter
labs = labs[curr_filter]
preds = preds[curr_filter]
subs = subs[curr_filter]
t0_suvr = t0_suvr[curr_filter]
t0_diagnoses = t0_diagnoses[curr_filter]
delta_time = delta_time[curr_filter]
apoe = apoe[curr_filter]

# filter for unique subjects:
_, unique_idxs = np.unique(subs, return_index=True)

# look at highest suvr amyloid- subjects
# changes, filter_idxs, t0_suvr_ind = get_unique_changes(labs, subs, t0_suvr=t0_suvr)
# sort_idxs = np.argsort(t0_suvr_ind)[::-1]
# t0_suvr_ind = t0_suvr_ind[sort_idxs]
# changes = changes[sort_idxs]

# filter for subjects with at least "questionable" diagnosis
# questionable_dia_filter = t0_diagnoses >= 0.5
# labs = labs[questionable_dia_filter]
# preds = preds[questionable_dia_filter]
# subs = subs[questionable_dia_filter]
# t0_suvr = t0_suvr[questionable_dia_filter]
# to_ampos_subjects()

# creating 100 subj study, how many are also top 100 pos change subject?
filter_lab, filter_pred, filter_comb = get_matches_target(50, 50, True)
visualize_top_changes()
subjects = to_ampos_subjects(filter_pred)
subjects = to_ampos_subjects(filter_lab)
# 50 subj in study..
filter_lab, filter_pred, filter_comb = get_matches_target(100, 20, True)
subjects = to_ampos_subjects(filter_pred)
subjects = to_ampos_subjects(filter_lab)
