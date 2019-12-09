from glob import glob
import os
import pickle
import numpy as np


target_folder = r'C:\Users\Fabian\stanford\gbdt'
target_file = 'all_results.pickle'
folders = glob(os.path.join(target_folder, '157*'))

folder = folders[-1]
target = os.path.join(folder, target_file)
with open(target, 'rb') as f:
    data = pickle.load(f)

gbms = data['gbm']
x_names = data['x_names']
fis = []
for gbm in gbms:
    fis.append(gbm.feature_importance())
fi = fis[0]
for f in fis[1:]:
    fi += f
fi = (fi/5).astype(np.int)

f_names = []
f_names = list(x_names)
for i in range(len(fi)-len(x_names)):
    f_names.append(f'act_{i}')
f_names = np.array(f_names)

print('done')