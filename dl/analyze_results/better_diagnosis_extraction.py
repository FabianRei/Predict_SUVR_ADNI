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
    perc_lab = np.percentile(labs, 100 - number*100/1137)
    subs_lab = subs[labs>perc_lab]
    count_lab = number
    while len(np.unique(subs_lab)) < number:
        count_lab += 1
        perc_lab = np.percentile(labs, 100 - count_lab*100/1137)
        subs_lab = subs[labs>perc_lab]
    perc_pred = np.percentile(preds, 100 - number*100/1137)
    subs_pred = subs[preds>perc_pred]
    count_pred = number
    while len(np.unique(subs_pred)) < number:
        count_pred += 1
        perc_pred = np.percentile(preds, 100 - count_pred*100/1137)
        subs_pred = subs[preds>perc_pred]
    # percentile might get bigger number, if both have the dame delta SUVR pred/lab
    return perc_pred, perc_lab


def get_unique_t0_suvr():
    global t0_suvr
    global subs
    t0, t0_idx = np.unique(subs, return_index=True)
    suvr_unique = t0_suvr[t0_idx]
    return suvr_unique

def get_matches(number):
    global preds
    global labs
    global subs
    perc_pred = np.percentile(preds, 100 - number*100 / 1137)
    perc_lab = np.percentile(labs, 100 - number*100 / 1137)
    # perc_pred, perc_lab = get_top_subjects(76)
    # _, perc_lab = get_top_subjects(77)
    median_lab = np.percentile(labs, 50)
    sub_ids = np.unique(subs[(labs > perc_lab) & (preds > perc_pred)])
    sub_ids_study, study_idx = np.unique(subs[preds>perc_pred], return_index=True)
    sub_ids_label = np.unique(subs[labs>perc_lab])
    print(f"{len(sub_ids_study)} in study, {len(np.unique(sub_ids))} also in top {len(sub_ids_label)} of labels.")
    # print(result)
    # print(len(np.unique(sub_ids)), sub_ids)
    return (labs > perc_lab) & (preds > perc_pred)


def get_matches_target(study_size, target_size):
    global preds
    global labs
    global subs
    # perc_pred = np.percentile(preds, 100 - number*100 / 1137)
    # perc_lab = np.percentile(labs, 100 - number*100 / 1137)
    perc_pred, perc_lab = get_top_subjects(study_size)
    _, perc_lab = get_top_subjects(target_size)
    median_lab = np.percentile(labs, 50)
    sub_ids = np.unique(subs[(labs > perc_lab) & (preds > perc_pred)])
    sub_ids_study, study_idx = np.unique(subs[preds>perc_pred], return_index=True)
    sub_ids_label = np.unique(subs[labs>perc_lab])
    print(f"{len(sub_ids_study)} in study, {len(np.unique(sub_ids))} also in top {len(sub_ids_label)} of labels.")
    # print(result)
    # print(len(np.unique(sub_ids)), sub_ids)
    return (labs > perc_lab) & (preds > perc_pred)


def get_high_t0suvr_matches(number):
    global preds
    global labs
    global t0_suvr
    high_suvr = np.percentile(t0_suvr, 100 - number*100 / 1137)
    perc_lab = np.percentile(labs, 100 - number*100 / 1137)
    # high_suvr, perc_lab = get_top_subjects(number)
    median_lab = np.percentile(labs, 50)
    sub_ids = np.unique(subs[(labs > perc_lab) & (t0_suvr > high_suvr)])
    sub_ids_study = np.unique(subs[t0_suvr>high_suvr])
    sub_ids_label = np.unique(subs[labs>perc_lab])
    print('matches based on high suvr selection')
    print(f"{len(sub_ids_study)} in study, {len(np.unique(sub_ids))} also in top {len(sub_ids_label)} of labels.")
    # print(result)
    # print(len(np.unique(sub_ids)), sub_ids)
    return (labs > perc_lab) & (t0_suvr > high_suvr)


def get_minimal_change_matches(number):
    global preds
    global labs
    perc_low = np.percentile(preds, 50 - number*50/1137)
    perc_high = np.percentile(preds, 50 + number*50.5/1137)
    perc2_low = np.percentile(labs, 50 - number*50/1137)
    perc2_high = np.percentile(labs, 50 + number*50/1137)
    sub_ids = np.unique(subs[(preds > perc_low) & (preds < perc_high) & (labs > perc2_low) & (labs < perc2_high)])
    sub_ids_study = np.unique(subs[(preds > perc_low) & (preds < perc_high)])
    sub_ids_label = np.unique(subs[(labs > perc2_low) & (labs < perc2_high)])
    print(f"{len(sub_ids_study)} in study, {len(np.unique(sub_ids))} also in top {len(sub_ids_label)} of labels.")


def plot_densities():
    global preds
    global labs
    sns.kdeplot(preds, label='GBDT predictions')
    sns.kdeplot(labs, label='Ground truth')
    plt.legend()
    plt.show()


def rmse(target, prediction):
    return mean_squared_error(target, prediction)**0.5


def rmse_over_time():
    global preds
    global labs
    global delta_time
    delta_time_yrs = delta_time/(365*24*60*60)
    results = []
    for i in range(4):
        start = 1+i*2
        end = 3+i*2
        time_filter = (delta_time_yrs>=start) & (delta_time_yrs < end)
        time_preds = preds[time_filter]
        time_labs = labs[time_filter]
        results.append(rmse(time_preds, time_labs))
    print(results)
    return 0


def get_examdate(s, delta_time):
    global subs
    global extra_exam_date
    global extra_subs
    fi = extra_subs == s
    fi2 = subs==s
    exam_dates = extra_exam_date[fi]
    deltas = delta_time[fi2]
    s1 = np.argsort(exam_dates)
    s2 = np.argsort(deltas)
    # get exam dates in same relative order as deltas (required for correct insertion later on)
    exam_dates = exam_dates[s1][np.argsort(s2)]
    return exam_dates

def get_diagnosis(s, ex, d_data, name='-1'):
    """
    Looks into the berkeley study csv file and returns the corresponding label.
    If rid+examdate yield more than one corresponding row, it returns -1, as we then cannot find the
    exact label in the table.
    :param s:
    :param ex:
    :return:
    """
    global diagnosis_data
    ts = pd.Timestamp(ex, unit='s')
    # d_data = diagnosis_data['EXAMDATE']
    # d_data = list(d_data)
    # d_results = []
    # for d in d_data:
    #     try:
    #         d = d.timestamp()
    #     except:
    #         d = -1
    #     d_results.append(d)
    # d_data = np.array(d_results)
    # d_data = d_data[diagnosis_data['RID'] == s]
    e = ts.timestamp()
    diffs = np.abs(d_data-e)/(3600*24)

    row_idx = (diffs<50) & (diagnosis_data['RID'] == s)
    if row_idx.sum() > 1:
        print('more than one result for rid & examdate combination in berkeley study data')
        print(f'file name is {name}, rid is {s}, exam date is {ts}')
        return -100, -1
    if row_idx.sum() == 0:
        print('no result found for rid & examdate combination in berkeley study data')
        print(f'file name is {name}, rid is {s}, exam date is {ts}')
        return -100, -1
    label = diagnosis_data['ADNI_MEM'][row_idx]
    return float(label), float(diffs[row_idx])


def get_ed_registry(r, v, u):
    global not_found_counter
    # look for examdate in registry, as recommended by ADNI
    global registry_data
    row_idx = (registry_data['VISCODE'] == v) & (registry_data['RID'] == r) & (registry_data['USERDATE'] == u)
    reg_ex_date = registry_data['EXAMDATE'][row_idx]
    if len(reg_ex_date) > 1:
        print(len(reg_ex_date))
        print('not found exactly, using userdate')
        not_found_counter += 1
        return u.timestamp()
    if len(reg_ex_date) < 1:
        print('not found, using userdate')
        not_found_counter += 1
        return u.timestamp()
    if pd.isnull(reg_ex_date.iloc[0]):
        print('not found, using userdate')
        not_found_counter += 1
        return u.timestamp()
    reg_ex_date = reg_ex_date.iloc[0].timestamp()
    return reg_ex_date


def create_diagnosis_data():
    # I want diagnosis, suvr, faqtotal, subid, examdate, delta_time, delta_suvr, t0_suvr
    # I want to add t0 values
    global subs
    global t0_suvr
    global delta_suvr
    global delta_time
    global faq
    global suvr
    # append numbers for t0 values
    zeros = np.zeros(np.unique(subs).shape)
    delta_time = np.append(delta_time, zeros)
    delta_suvr = np.append(delta_suvr, zeros)
    append_t0_suvr = []
    for s in np.unique(subs):
        t0_s = t0_suvr[s==subs][0]
        append_t0_suvr.append(t0_s)
    append_t0_suvr = np.array(append_t0_suvr)
    t0_suvr = np.append(t0_suvr, append_t0_suvr)
    suvr = np.append(suvr, append_t0_suvr)
    subs = np.append(subs, np.unique(subs))
    exam_dates = np.copy(delta_time)
    diagnoses = np.copy(delta_time)
    differences = np.copy(delta_time)
    t0_diagnoses = np.copy(delta_time)
    ex_date = diagnosis_data['EXAMDATE']
    # name not found dates via -1
    ex_data = []
    for d in ex_date:
        try:
            d = d.timestamp()
        except:
            d = -1
        ex_data.append(d)
    ex_rid = diagnosis_data['RID']
    ex_vis = diagnosis_data['VISCODE']
    ex_us = diagnosis_data['USERDATE']
    new_ex_data = ex_data.copy()

    # fill in exam dates extracted from registry metadata
    for i, (ed, r, v, u) in enumerate(zip(ex_data, ex_rid, ex_vis, ex_us)):
        if ed == -1:
            ex_date = get_ed_registry(r, v, u)
            new_ex_data[i] = ex_date
    new_ex_data = np.array(new_ex_data)

    # search for diagnosis
    found = []
    found_t0 = []
    for s in np.unique(subs):
        # delta time is to order correctly. Needed to put exam_data on the correct spot.
        ex = get_examdate(s, delta_time)
        digs = []
        diffs = []
        for e in ex:
            dig, diff = get_diagnosis(s, e, new_ex_data)
            digs.append(dig)
            diffs.append(diff)
        if min(digs)==-1:
            found.append(-1)
        else:
            found.append(1)
        if digs[ex.argmin()] == -1:
            found_t0.append(0)
        else:
            found_t0.append(1)
        exam_dates[subs==s] = ex
        diagnoses[subs==s] = digs
        differences[subs==s] = diffs
        t0_diagnoses[subs==s] = digs[ex.argmin()]
    found = np.array(found)
    found_t0 = np.array(found_t0)
    diagnoses[pd.isna(diagnoses)] = -100
    output_folder = r'C:\Users\Fabian\stanford'
    results = {'ADNI_MEM': diagnoses, 'differences': differences, 'subs': subs, 'suvr': suvr,
               't0_suvr': t0_suvr, 'exam_dates': exam_dates, 'delta_time': delta_time, 'delta_suvr': delta_suvr, 'found_subject': found,
               't0_ADNI_MEM': t0_diagnoses, 'found_t0': found_t0}
    with open(os.path.join(output_folder, 'diagnoses_PSYCHSUM.pickle'), 'wb') as f:
        pickle.dump(results, f)
    # just for analysis purposes..
    t, idxs = np.unique(subs, return_index=True)
    t = t0_diagnoses[idxs]
    t = t[t>-100]

    print('nice')



target_folder = r'C:\Users\Fabian\stanford\gbdt\analysis'
target_file = 'all_results.pickle'
folders = glob(os.path.join(target_folder, '157*'))


# select GBDT predictions in folder
folder = folders[-1]
target = os.path.join(folder, target_file)
with open(target, 'rb') as f:
    data = pickle.load(f)

# load additional data
in_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\rf_data_train_test_crossval.pickle'
with open(in_path, 'rb') as f:
    more_data = pickle.load(f)

path_detailled_data = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\test_meta_data_complete.pickle'
with open(path_detailled_data, 'rb') as f:
    detailled_data = pickle.load(f)

diagnosis_csv = r'C:\Users\Fabian\stanford\UWNPSYCHSUM_03_26_20.csv'
diagnosis_data = pd.read_csv(diagnosis_csv, parse_dates=['EXAMDATE', 'USERDATE'])

registry_csv = r'C:\Users\Fabian\stanford\REGISTRY.csv'
registry_data = pd.read_csv(registry_csv, parse_dates=['EXAMDATE', 'USERDATE'])

# get data as variables
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
delta_suvr = more_data['delta_suvr']
delta_time = more_data['delta_time']
suvr = more_data['suvr']

extra_subs = detailled_data['sub_id']
extra_exam_date = detailled_data['exam_date']

#########################################
# get diagnosis data
#########################################
not_found_counter = 0
create_diagnosis_data()
