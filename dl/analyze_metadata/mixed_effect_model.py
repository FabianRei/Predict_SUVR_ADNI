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


data_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\test_meta_data_no_t0.pickle'


with open(data_path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())
pd_data = pd.DataFrame(data)

md = smf.mixedlm("delta_suvr ~ delta_time + a1_e4 + a2_e4 + weight_meaned + amyloid_status + t0_suvr + sex_f_true", pd_data, groups=pd_data['img_id'])
mdf = md.fit()
# print(mdf.summary())
print('done')

dependent_arr = data['delta_suvr']
independent_arr = np.column_stack((data['delta_time'], data['a1_e4'], data['a2_e4'], data['t0_suvr'], data['sex_f_true'], data['weight_meaned']))
independent_arr = sm.add_constant(independent_arr)
# gamma_model = sm.GLM(dependent_arr, independent_arr, family=sm.families.Gamma())
# gamma_results = gamma_model.fit()
# print(gamma_results.summary())
model = sm.OLS(dependent_arr, independent_arr)
results = model.fit()
results.mse_resid
print(results.summary())
