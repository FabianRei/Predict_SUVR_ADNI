import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from dl.data.project_logging import CsvWriter
# import h5py
import plotly.express as px
import statsmodels
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sys

sys.path.append(r'C:\Users\Fabian\Anaconda3\envs\pynew\poppler-0.90.1\bin')

data_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\test_meta_data_no_t0_complete.pickle'


with open(data_path, 'rb') as f:
    data = pickle.load(f)

d_suvrs = data['delta_suvr']
d_suvr_csf = data['delta_suvr_csf']
delta_t = data['delta_time']
suvr_cerebnorm = data['suvrs']
suvr_comp = data['suvrs_comp']
d_suvr = data['delta_suvr']
d_suvr_comb = data['delta_suvr_comp']
t_sum = np.array(suvr_cerebnorm)>1.11
t_sum_comp = np.array(suvr_comp)>0.79
print(np.mean(d_suvr_csf[t_sum]))
print(np.mean(d_suvr_csf[~t_sum]))
t_sum_names = ['positive' if x else 'negative' for x in t_sum_comp]
df_long = pd.DataFrame({'Amyloid status': np.flip(t_sum_names), 'Delta SUVR': np.flip(d_suvr_comb), 'Delta time (years)': np.flip(delta_t)})
# df_cerebnorm = pd.DataFrame({'Amyloid_status': t_sum_names, 'Delta SUVR CEREBNORM': delta_s, 'Delta time (years)': delta_t})

# df_csf = df_csf.melt(id_vars='Delta time (years)')
# fig = px.scatter(df_cerebnorm, x='Delta time (years)', y='Delta SUVR CEREBNORM', color='Amyloid_status', trendline='ols')
fig = px.scatter(df_long, x='Delta time (years)', y='Delta SUVR', color='Amyloid status', trendline='ols')
fig.write_image(r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\figure_1b.eps')
fig.show()
results = px.get_trendline_results(fig)
print(results)
results.query("Amyloid_status == 'negative'").px_fit_results.iloc[0].summary()
results.query("Amyloid_status == 'positive'").px_fit_results.iloc[0].summary()


# fig = px.scatter(df_refnorm, x='Delta time (years)', y='Delta SUVR REFNORM', color='Amyloid_status', trendline='ols')
# fig.show()
# results = px.get_trendline_results(fig)
# print(results)
# results.query("Amyloid_status == 'negative'").px_fit_results.iloc[0].summary()
# results.query("Amyloid_s