import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import pickle
import numpy as np
from matplotlib import pyplot as plt
import warnings
import pandas as pd
import os
import time
import pprint


def rmse(target, prediction):
    return mean_squared_error(target, prediction) ** 0.5

def get_num(x):
    return float(x[0])

def get_stroke_xy(data, stroke_keys, group_mrs=False, delta=False):
    # get relevant data
    arrs = []
    y = np.array(data['mrs'])
    dep_not_nan_filter = ~pd.isna(y)
    for k in stroke_keys:
        arr = np.array(data[k])
        not_nan_filt = ~pd.isna(arr)
        # remove string description from numeric values
        try:
            float(arr[not_nan_filt][0])
        except:
            arr = np.array([get_num(i) if not pd.isna(i) else np.nan for i in arr])
        # if more than 10 categories, normalize [not needed for GBDT, see documentation]
        # try:
        #     if len(np.unique(arr)) > 10:
        #         arr = arr/arr[not_nan_filt].std()
        #         arr = arr-arr[not_nan_filt].mean()
        # except:
        #     print('db')
        if k == 'pre-eventrankin':
            old_mrs = np.copy(arr)
        print(k)
        # if np.sum(arr==-1) > 0:
        #     arr[arr == -1] = np.nan
        arr = np.expand_dims(arr, 1)
        arrs.append(arr)
    x = np.concatenate(arrs, axis=1)
    x = x[dep_not_nan_filter]
    y = y[dep_not_nan_filter]
    # group 0-2, 3-4, 5-6
    if group_mrs:
        for i, v in enumerate(y):
            if v <= 2:
                y[i] = 0
            elif v <= 4:
                y[i] = 1
            else:
                y[i] = 2
    # make delta_mrs
    if delta:
        old_mrs = old_mrs[dep_not_nan_filter]
        old_mrs[pd.isna(old_mrs)] = 0
        y -= old_mrs
    # fabricate train_test_split
    train_test_split = []
    gen = train_test_gen(5)
    for i in range(len(x)):
        train_test_split.append(next(gen))
    np.random.seed(42)
    train_test_split = np.random.permutation(train_test_split)
    x_names = stroke_keys

    return x, y, train_test_split, x_names

def custom_eval(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    loss = np.where(np.abs(residual) > 1, (residual**2)*10.0, residual**2)
    return "custom_eval", np.mean(loss), False


def train_test_gen(nums):
    while True:
        for i in range(nums):
            yield i


def save_results(res):
    global path_output
    res_csv = {'RMSE_TRAIN': res['rmse_train'], 'RMSE_TEST': res['rmse'], 'ACC': res['accuracy'],
               'ACC_+-1': res['accuracy+-1'], 'ACC_+-2': res['accuracy+-2']}
    res_csv = pd.DataFrame.from_dict(res_csv)
    res_csv.to_csv(os.path.join(path_output, 'results.csv'))
    with open(os.path.join(path_output, 'params.txt'), 'w') as f:
        pprint.pprint(res['params'], f)
        pprint.pprint('Features used:', f)
        pprint.pprint(res['x_names'], f)
    with open(os.path.join(path_output, 'all_results.pickle'), 'wb') as f:
        pickle.dump(res, f)
    res['x'] = None
    print('nice')


def cross_validation_gbdt(data, params, cval_range=5, exclude='', extra_folder='', group_mrs=False, delta=False):
    params['verbose'] = -1
    x, y, train_test_split, x_names = get_stroke_xy(data, stroke_keys, group_mrs=group_mrs, delta=delta)

    res = {'predictions': [], 'labels': [], 'rmse': [], 'pred_train': [], 'labels_train': [],
           'rmse_train': [], 'accuracy': [], 'accuracy+-1': [], 'accuracy+-2': [], 'gbm': []}

    for split in range(cval_range):
        test = train_test_split == split
        # use train mean val for not known faqtotal and mmse
        try:
            faq_index = x_names.index('faqtotal')
            x[:, faq_index][x[:, faq_index] == -1] = np.mean(x[:, faq_index][~test][x[:, faq_index][~test] != -1])
        except:
            pass
        try:
            mmse_index = x_names.index('mmsescore')
            x[:, mmse_index][x[:, mmse_index] == -1] = np.mean(x[:, mmse_index][~test][x[:, mmse_index][~test] != -1])
        except:
            pass

        x_train = x[~test]
        y_train = y[~test]
        d_train = lgb.Dataset(x_train, label=y_train)
        x_test = x[test]
        y_test = y[test]
        warnings.simplefilter('ignore', UserWarning)
        gbm = lgb.train(params, d_train)
        warnings.simplefilter('default', UserWarning)
        y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
        y_pred_train = gbm.predict(x_train, num_iteration=gbm.best_iteration)
        res['predictions'].append(y_pred)
        res['labels'].append(y_test)
        # res['rmse'].append(rmse(y_test, y_pred))
        y_pred_rnd = y_pred.round()
        if params['objective'] == 'multiclass':
            y_pred_rnd = np.array([np.argmax(i) for i in y_pred])
        res['accuracy'].append((y_test==y_pred_rnd).mean())
        res['accuracy+-1'].append((np.abs(y_pred_rnd-y_test) <= 1).mean())
        res['accuracy+-2'].append((np.abs(y_pred_rnd-y_test) <= 2).mean())
        print(f'acc = {(y_test==y_pred_rnd).mean()}, acc1 = {(np.abs(y_pred_rnd-y_test) <= 1).mean()}, acc2 = {(np.abs(y_pred_rnd-y_test) <= 2).mean()}')
        # res['rmse_train'].append(rmse(y_pred_train, y_train))
        res['pred_train'].append(y_pred_train)
        res['labels_train'].append(y_train)
        res['gbm'].append(gbm)
        if not params['objective'] == 'multiclass':
            print(rmse(y_test, y_pred))
            print(rmse(y_train, y_pred_train))
        else:
            print((y_test==y_pred_rnd).mean())
            y_tr = np.array([np.argmax(i) for i in y_pred_train])
            print((y_tr==y_train).mean())
    res['params'] = params
    res['x_names'] = x_names
    res['x'] = x
    res['y'] = y
    res['train_test_split'] = train_test_split
    print(np.mean(res['accuracy']))
    print(np.mean(res['accuracy+-1']))
    save_results(res)
    return res


path_csv = r'C:\Users\Fabian\Desktop\sample_lesion\tabular_data2.csv'
path_output = r'C:\Users\Fabian\Desktop\sample_lesion'
np.random.seed(42)

data = pd.read_csv(path_csv)
keys = np.array(data.keys())
dep_key = keys[3]
indep_keys = np.concatenate((keys[2:3], keys[4:]))

stroke_keys = ['lesion volume', 'age', 'baselinenihssscore', 'priorstroketia', 'treatment', 'tpa', 'male', 'myocardialinfarction',
             'hypertension', 'atrialfibrillation', 'hypercholesterolemia', 'diabetes', 'haspatienteverhadastrokepriortoq',
             'pre-eventrankin', '1a. Level of Consciousness', '1b. LOC Questions', '1c. LOC Commands ',
             '2. Best Gaze', '3. Visual Fields', '4. Facial Palsy', '5a. Motor: Left Arm', '5b. Motor: Right Arm', '6a. Motor: Left leg',
             '6b. Motor: Right Leg', '7. Limb Ataxia', '8. Sensory', '9. Best Language', '10. Dysarthria', '11. Extinction and Inattention',
             'Initial Systolic Blood Pressure on arrival at STUDY SITE', 'Initial Diastolic Blood Pressure on arrival at STUDY SITE',
             'Glucose']


stroke_keys += ['tci_num', 'onset_to_imaging', 'image_to_dsa', 'image_to_eariest_treat']

x, y, train_test_split, x_names = get_stroke_xy(data, stroke_keys)
# params = {
#     'boosting_type': 'gbdt',
#     'objective': 'multiclass', # default is 'multiclass'
#     'num_classes': 3,
#     # 'objective': 'regression', # default is 'regression'
#     # 'metric': 'mse',
#     'max_leaves': 50, #default is 50
#     'max_depth': 5,  # 9 optimal for activations, 4 optimal for metadata only
#     'max_bin': 255, # default is 255
#     'num_iterations': 4000,
#     'learning_rate': 0.0005,  # 0.0006 optimal for metadata, 0.0045 for activations
#     'feature_fraction': 0.8, # 0.8 for metadata only
#     'bagging_fraction': 0.5,
#     'bagging_freq': 2,
#     'verbose': 0,
#     'min_data_in_leaf': 25 # default is 9
# }

params = {
    'boosting_type': 'gbdt',
    # 'objective': 'multiclass', # default is 'multiclass'
    # 'num_classes': 3,
    'objective': 'regression', # default is 'regression'
    'metric': 'mse',
    'max_leaves': 200, #default is 50
    'max_depth': 4,  # 9 optimal for activations, 4 optimal for metadata only
    'max_bin': 255, # default is 255
    'num_iterations': 4000,
    'learning_rate': 0.0015,  # 0.0006 optimal for metadata, 0.0045 for activations
    'feature_fraction': 0.8, # 0.8 for metadata only
    'bagging_fraction': 0.5,
    'bagging_freq': 2,
    'verbose': 0,
    'min_data_in_leaf': 33 # default is 9
}

cat_features = list(range(len(stroke_keys)))

# params['categorical_feature'] = cat_features[3:]
cross_validation_gbdt(data, params, group_mrs=False, delta=True)
print('db')