import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from dl.data.project_logging import CsvWriter
import h5py
import plotly.express as px
import statsmodels
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error


def rmse(target, prediction):
    return mean_squared_error(target, prediction)**0.5


def mixed_lm_crossval(data, cval_fold=1):
    train_test = data.pop('test')
    dependent_var = data.pop('delta_suvr')
    apoe = data.pop('apoe')
    for i in np.unique(apoe):
        d_key = f'apoe_{i}'
        d_val = apoe==i
        data[d_key] = d_val
    keys = list(data.keys())
    print(keys)
    rmse_test = []
    rmse_train = []
    pickle_data = []
    predictions = []
    labels = []
    for i in range(cval_fold):
        pdata = dict()
        test = train_test == i
        if 'faqtotal' in keys:
            data['faqtotal'][data['faqtotal'] == -1] = np.mean(data['faqtotal'][~test][data['faqtotal'][~test] != -1])
        if 'mmsescore' in keys:
            data['mmsescore'][data['mmsescore'] == -1] = np.mean(data['mmsescore'][~test][data['mmsescore'][~test] != -1])
        independent_var = np.column_stack((data['t0_suvr'], data['sex'], data['weight'], data['delta_time'],
                                           data['mmsescore'], data['faqtotal'], data['age'],
                                           data['apoe_0'], data['apoe_1'], data['apoe_2'], data['apoe_3'], data['apoe_4'], data['apoe_5']))
        dep_test = dependent_var[test]
        indep_test = independent_var[test]
        dep_train = dependent_var[~test]
        indep_train = independent_var[~test]
        model = sm.OLS(dep_train, indep_train)
        results = model.fit()
        dep_test_pred = model.predict(params=results.params, exog=indep_test)
        dep_train_pred = model.predict(params=results.params, exog=indep_train)
        print(rmse(dep_test_pred, dep_test))
        print(rmse(dep_train_pred, dep_train))
        rmse_test.append(rmse(dep_test_pred, dep_test))
        rmse_train.append(rmse(dep_train_pred, dep_train))
        predictions.append(dep_test_pred)
        labels.append(dep_test)
        pdata['model'] = model
        pdata['model_params'] = results.params
        pdata['y'] = dep_test
        pdata['x'] = indep_test
        pdata['y_train'] = dep_train
        pdata['x_train'] = indep_train
        pickle_data.append(pdata)
        print('db')
    patient_analysis_data = {}
    patient_analysis_data['predictions'] = predictions
    patient_analysis_data['labels'] = labels
    patient_analysis_data['train_test_split'] = train_test
    print(f'mean rmse test is {np.mean(rmse_test)}')
    print(f'mean rmse train is {np.mean(rmse_train)}')
    print('db')


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


def get_pred_labels(data):
    # gbms = data['gbm']
    # x_names = data['x_names']
    preds = data['predictions']
    labs = data['labels']
    # y = data['y']
    train_test_split = data['train_test_split']
    preds, labs = rearrange_pred_labs(preds, labs, train_test_split)
    return preds, labs


data_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\rf_data_train_test_crossval.pickle'
gbdt_w_acs = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\more_train\1587473346_w_acs__0_033932022300775154\all_results.pickle'
gbdt_no_acs_results = r'C:\Users\Fabian\stanford\gbdt\rsync\.special\1576073951_wo_acs__0_0354701097223875\all_results.pickle'

with open(gbdt_w_acs, 'rb') as f:
    data_acs = pickle.load(f)

with open(gbdt_no_acs_results, 'rb') as f:
    data_no_acs = pickle.load(f)

ac_pred, ac_lab = get_pred_labels(data_acs)
met_pred, met_lab = get_pred_labels(data_no_acs)

all_pred = np.concatenate((ac_pred, met_pred))
all_lab = np.concatenate((ac_lab, met_lab))
sample_group = np.array([i for i in range(len(ac_lab))])
sample_groups = np.concatenate((sample_group, sample_group))
all_sq_error = (all_lab-all_pred)**2
labels = np.copy(all_pred)
labels[:] = 0
labels[:len(ac_pred)] = 1

data = {'if_acs': labels, 'predictions': all_pred, 'squared_error': all_sq_error, 'sample_groups': sample_groups}
# mixed_lm_crossval(data)
pd_data = pd.DataFrame(data)
#
md = smf.mixedlm("squared_error ~ if_acs", pd_data, groups=pd_data['sample_groups'])

# md = smf.mixedlm("delta_suvr ~ delta_time + a1_e4 + a2_e4 + weight_meaned + amyloid_status + t0_suvr + sex_f_true", pd_data, groups=pd_data['img_id'])
mdf = md.fit()
print(mdf.summary())
print('done')
#
# dependent_arr = data['delta_suvr']
# independent_arr = np.column_stack((data['delta_time'], data['a1_e4'], data['a2_e4'], data['t0_suvr'], data['sex_f_true'], data['weight_meaned']))
# independent_arr = sm.add_constant(independent_arr)
# # gamma_model = sm.GLM(dependent_arr, independent_arr, family=sm.families.Gamma())
# # gamma_results = gamma_model.fit()
# # print(gamma_results.summary())
# model = sm.OLS(dependent_arr, independent_arr)
# results = model.fit()
# results.mse_resid
# print(results.summary())
