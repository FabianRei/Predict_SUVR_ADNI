from glob import glob
import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import shap


def get_xy():
    global data
    x = data['x']
    y = data['y']
    xx = []
    yy = []
    tts = data['train_test_split']
    for split in np.unique(tts):
        x_vals = x[tts == split]
        xx.append(x_vals)
        y_vals = y[tts==split]
        yy.append(y_vals)
    return xx, yy


def get_shap_names():
    global x_names

    return np.array(x_names)


class reg_model:
    def __init__(self, mod, mod_params):
        self.mod = mod
        self.mod_params = mod_params

    def __call__(self, data):
        res = self.mod.predict(params=self.mod_params, exog=data)
        return res


def get_shap_values():
    global data
    model = []
    model_params = []
    xx = []
    y = []
    y_train = []
    x_train = []
    for d in data:
        model.append(d['model'])
        model_params.append(d['model_params'])
        xx.append(d['x'])
        y.append(d['y'])
        y_train.append(d['y_train'])
        x_train.append(d['x_train'])
    all_shap = []
    shap_names = get_shap_names()
    for l, g, g_params, x, bd in zip(y, model, model_params, xx, x_train):
        r_model = reg_model(g, g_params)
        k_meaned = shap.kmeans(bd, 20)
        shap_values = shap.KernelExplainer(r_model, k_meaned).shap_values(x)
        all_shap.extend(shap_values)
    all_shap = np.array(all_shap)
    all_x = np.concatenate(xx)
    shap.summary_plot(all_shap, features=all_x, feature_names=shap_names, max_display=8)
    print('nice')


pdata = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\mixed_effects_data.pickle'
with open(pdata, 'rb') as f:
    data = pickle.load(f)

x_names = ['t0_suvr', 'sex', 'weight', 'delta_time', 'mmsescore', 'faqtotal', 'age', 'apoe_0', 'apoe_1', 'apoe_2', 'apoe_3', 'apoe_4', 'apoe_5']


get_shap_values()

print('done')