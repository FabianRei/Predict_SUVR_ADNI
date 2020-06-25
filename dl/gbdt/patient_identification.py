from glob import glob
import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

def rmse(target, prediction):
    return mean_squared_error(target, prediction)**0.5

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


def get_unique_changes(changes, subjects, subjects2=None, t0_suvr=None):
    # gets max changes for each subject
    if subjects2 is None:
        subjects2 = subjects
    result = []
    filt_idxs = changes == -np.inf
    # include unique t0_suvr results
    if t0_suvr is not None:
        t0_results = []
    for s in np.unique(subjects):
        if not s in subjects2:
            continue
        filt = s==subjects
        change = changes[filt].max()
        if t0_suvr is not None:
            t0_results.append(t0_suvr[filt].max())
        result.append(change)
        # create corresponding filter for maxval only
        filt_vals = filt[filt==True]
        filt_vals[:] = False
        filt_vals[changes[filt].argmax()] = True
        filt[filt == True] = filt_vals
        filt_idxs += filt
    if t0_suvr is not None:
        return np.array(result), filt_idxs, np.array(t0_results)
    return np.array(result), filt_idxs


def visualize_top_changes():
    global filter_lab
    global filter_pred
    global filter_comb
    global subs
    global preds
    global labs
    global t0_suvr
    changes_lab, filter_idxs_lab, t0_suvr_lab = get_unique_changes(labs[filter_lab], subs[filter_lab], t0_suvr=t0_suvr[filter_lab])
    changes_pred, filter_idxs_pred, t0_suvr_pred = get_unique_changes(labs[filter_pred], subs[filter_pred], t0_suvr=t0_suvr[filter_pred])
    changes_comb, filter_idxs_comb = get_unique_changes(labs[filter_lab], subs[filter_lab], subs[filter_comb])
    changes, filter_idxs, t0_suvr_ind = get_unique_changes(labs, subs, t0_suvr=t0_suvr)
    # filter out top changes that are also predicted
    flt = np.copy(filter_lab)
    flt[flt==True] = filter_idxs_lab
    flt2 = np.copy(filter_lab)
    flt2[flt2==True] = filter_idxs_comb
    colors = sns.color_palette()
    plt.hist((labs[flt2], labs[flt & ~flt2]), stacked=True, color=[colors[0], colors[1]], alpha=0.66,
             label=['Subject in study via top 100 amyloid negative GBDT predictions','Max changes of top 100 amyloid negative subjects'])
    sns.distplot((labs[flt2], labs[flt & ~flt2]), stacked=True)
    plt.xlabel('Delta SUVR')
    plt.ylabel('Frequency')

    sns.distplot(changes, color=colors[3])
    print('db')


def changing_diagnosis(t0_diagnoses, diagnoses, subs):
    change_to_severe = []
    all_follow_up_rating = 0
    for s in np.unique(subs):
        t0_dig = t0_diagnoses[subs==s]
        diags = diagnoses[subs==s]
        diff = diags-t0_dig
        print(diags)
        if diff.max()>0:
            change_to_severe.append(diff.max())
    print(f'{len(change_to_severe)} subjects changed to a more severe CDR rating')
    return change_to_severe

# results file (pickle) with predictions
# gbdt_results = r'C:\Users\Fabian\stanford\gbdt\rsync\.special\1576022195_w_acs__0_03478181179857318\all_results.pickle'
gbdt_results = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\more_train\1587473346_w_acs__0_033932022300775154\all_results.pickle'
# gbdt_results = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\better_weighting\1587490519_w_acs__0_03409988352699489\all_results.pickle'
lin_reg_results = r'C:\Users\Fabian\stanford\gbdt\rsync\.special\linear_regression\all_results.pickle'
gbdt_no_acs_results = r'C:\Users\Fabian\stanford\gbdt\rsync\.special\1576073951_wo_acs__0_0354701097223875\all_results.pickle'

################################################################
################################################################
selected_results = gbdt_results
# file with additional metadata
additional_data = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\rf_data_train_test_crossval_more_trained_activations.pickle'
# path_detailled_data = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\test_meta_data_complete.pickle'
# diagnostic data:
path_diagnosis_data = 'C:\\Users\\Fabian\\stanford\\diagnoses_DXSUM.pickle'
# load data
with open(selected_results, 'rb') as f:
    data = pickle.load(f)
with open(additional_data, 'rb') as f:
    more_data = pickle.load(f)
with open(path_diagnosis_data, 'rb') as f:
    diagnosis_data = pickle.load(f)

# with open(path_detailled_data, 'rb') as f:
#     detailled_data = pickle.load(f)

# gbms = data['gbm']
# x_names = data['x_names']
preds = data['predictions']
labs = data['labels']
# y = data['y']
train_test_split = data['train_test_split']
preds, labs = rearrange_pred_labs(preds, labs, train_test_split)
subs = more_data['subs']
t0_suvr = more_data['t0_suvr']
age = more_data['age']
weight = more_data['weight']
apoe = more_data['apoe']
faq = more_data['faqtotal']
sex = more_data['sex']
mmse = more_data['mmsescore']
delta_time = more_data['delta_time']
# get diagnosis data. Order is same as data above (tested via dia_subs, without t0_data)
dia_subs = diagnosis_data['subs']
diagnoses = diagnosis_data['diagnoses']
t0_diagnoses = diagnosis_data['t0_diagnoses']
dia_delta_time = diagnosis_data['delta_time']

######################################
# diagnosis progressors analysis
dia_filter = diagnoses>-1
dig_filt = diagnoses[dia_filter]
delta_dig_filt = dia_delta_time[dia_filter]
dig_subs_filt = dia_subs[dia_filter]

progressors = []
all_progressors = 0
weirdos = []
weirdo_subs = []
good_guys = []
good_subs = []

for s in np.unique(dig_subs_filt):
    digs = dig_filt[dig_subs_filt==s]
    delts = delta_dig_filt[dig_subs_filt==s]
    digs = digs[np.argsort(delts)]
    if digs.max() >= 0.5 and digs.min() < 0.5 and digs.argmin() < digs.argmax():
        progressors.append(s)
        # print(digs)
    else:
        progressors.append(-1)
    if digs.max()-digs.min()>=0.5:
        all_progressors += 1
        print(digs)
    if (np.where(digs==digs.max())[0][0] < np.where(digs==digs.min())[0][-1]) and digs.min()<=digs.max() and digs.max()==0.5:
        print('weird')
        weirdos.append(list(digs))
        weirdo_subs.append(s)
    if not (np.where(digs==digs.max())[0][0] < np.where(digs==digs.min())[0][-1]) and digs.min()<digs.max() and digs.min()==1:
        print('good guy')
        good_guys.append(list(digs))
        good_subs.append(s)
progressors = np.array(progressors)
progressors = progressors[progressors>-1]


progressors = np.array(good_subs)

progressor_filter = np.copy(subs)
for i in range(len(progressor_filter)):
    if progressor_filter[i] in progressors:
        progressor_filter[i] = 1
    else:
        progressor_filter[i] = 0
progressor_filter = progressor_filter>0


t0_diagnoses = t0_diagnoses[dia_delta_time>0]
# apoe = apoe[dia_delta_time>0]
# apoe
apoe_filter = apoe >= 4
# filter by t0_diagnoses mild dementia
mild_dementia_filter = t0_diagnoses == 0.5
normal_cdr = t0_diagnoses == 0
less_mild_dementia = t0_diagnoses == 1
moderate_dementia = t0_diagnoses == 2
severe_dementia = t0_diagnoses == 3

# filter to worse dementia
worse_dementa_filter = t0_diagnoses > 0.5
# filter to data points, not t0 only
# filter to amyloid -
amneg_filter = t0_suvr<0.79
# filter to amyloid +
ampos_filter = t0_suvr>0.79
# filter to mildly a+
perc_50pos = 0.9535
mildly_pos_filter = (t0_suvr>0.79) & (t0_suvr <perc_50pos)

#filter to mildly a-
perc_50neg =  0.7390000000000001
mildly_neg_filter = (t0_suvr<0.79) & (t0_suvr > perc_50neg)

severe_pos_filter = t0_suvr >= perc_50pos
# filter to all cases
all_filter = t0_suvr > -1
#############################################
############################################
# choose filter to apply
curr_filter = all_filter
# apply filter
labs = labs[curr_filter]
preds = preds[curr_filter]
subs = subs[curr_filter]
t0_suvr = t0_suvr[curr_filter]
t0_diagnoses = t0_diagnoses[curr_filter]
progressor_filter = progressor_filter[curr_filter]

# filter for unique subjects:
_, unique_idxs = np.unique(subs, return_index=True)
# unique_filter = np.copy(all_filter)
# unique_filter[unique_idxs] = False
# unique_filter = ~unique_filter

print(np.mean(labs), np.mean(t0_suvr[unique_idxs]), len(unique_idxs))
print(f'{len(unique_idxs)} subjects')
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
print(f'rmse is {rmse(labs,preds)}')
# creating 100 subj study, how many are also top 100 pos change subject?
filter_lab, filter_pred, filter_comb = get_matches_target(46, 46, True)
# visualize_top_changes()
subjects = to_ampos_subjects(filter_pred)
filter_lab, filter_pred, filter_comb = get_matches_target(61, 61, True)
matched_progressors = filter_pred & progressor_filter
print(f'{len(np.unique(subs[matched_progressors]))} of 52 progressors were matched for {len(np.unique(subs[filter_pred]))} selected subjects')
interesting = t0_suvr[np.unique(subs[progressor_filter], return_index=True)[1]]
filter_lab, filter_pred, filter_comb = get_matches_target(122, 122, True)
matched_progressors = filter_pred & progressor_filter
print(f'{len(np.unique(subs[matched_progressors]))} of 52 progressors were matched for {len(np.unique(subs[filter_pred]))} selected subjects')
subjects = to_ampos_subjects(filter_pred)

filter_lab, filter_pred, filter_comb = get_matches_target(30, 30, True)
filter_lab, filter_pred, filter_comb = get_matches_target(31, 31, True)
filter_lab, filter_pred, filter_comb = get_matches_target(64, 64, True)



subjects = to_ampos_subjects(filter_lab)
# 50 subj in study..
filter_lab, filter_pred, filter_comb = get_matches_target(100, 20, True)
subjects = to_ampos_subjects(filter_pred)
subjects = to_ampos_subjects(filter_lab)
