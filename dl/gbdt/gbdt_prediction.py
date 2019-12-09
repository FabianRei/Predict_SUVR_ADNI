import os, sys, inspect
# this is done to run things from console
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
parend_dir2 = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir2)

import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from dl.gbdt.five_fold_cv import cross_validation_gbdt

# in_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\rf_data_train_test.pickle'
in_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\rf_data_train_test_crossval.pickle'
with open(in_path, 'rb') as f:
    data = pickle.load(f)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'max_leaves': 50,
    'max_depth': 4,  # 11 optimal for activations, 4 optimal for metadata only
    'max_bin': 255,
    'num_iterations': 4000,
    'learning_rate': 0.0007,  # 0.0007 optimal for metadata, 0.002 for activations
    'feature_fraction': 0.8,
    'bagging_fraction': 0.5,
    'bagging_freq': 1,
    'verbose': 0,
    'min_data_in_leaf': 23  # 23 for metadata, 9 for activations
}
# for i in range(3, 12):
#     params['max_depth'] = i
#     cross_validation_gbdt(data, params, activations=True)

# params['max_depth'] = 4
#
# # for i in range(5, 30):
# #     params['num_iterations'] = i*100
# #     cross_validation_gbdt(data, params)
#
# params['num_iterations'] = 2000
# # params['learning_rate'] = 0.004
# for i in range(5, 30, 2):
#     params['min_data_in_leaf'] = i
#     cross_validation_gbdt(data, params, activations=False, cval_range=5)
#
# print('done')

# cross_validation_gbdt(data, params, activations=False, cval_range=5)

# features = ['t0_suvr', 'sex', 'weight', 'delta_time', 'apoe', 'mmsescore', 'faqtotal', 'age']
# cross_validation_gbdt(data, params, activations=False, cval_range=5)

# new params: 0.009 lr, its 2000, max_depth 9, min_data_in_leaf 9
params['learning_rate'] = 0.0045
params['num_iterations'] = 4000
params['min_data_in_leaf'] = 9
params['max_depth'] = 11
cross_validation_gbdt(data, params, activations=True, cval_range=5)
# for i in range(4, 14):
#     params['max_depth'] = i
#     cross_validation_gbdt(data, params, activations=True, cval_range=1)
#
# params['max_depth'] = 11
#
# for i in range(6, 21, 3):
#     params['min_data_in_leaf'] = i
#     cross_validation_gbdt(data, params, activations=True, cval_range=1)
#
# params['min_data_in_leaf'] = 9
#
# for i in range(1, 12, 2):
#     params['learning_rate'] = 0.001*i
#     cross_validation_gbdt(data, params, activations=True, cval_range=1)


# for feat in features:
#     cross_validation_gbdt(data, params, activations=True, cval_range=5, exclude=feat)
