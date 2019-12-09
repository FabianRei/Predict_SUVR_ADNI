import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from dl.data.project_logging import CsvWriter

# label_path = '/share/wandell/data/reith/federated_learning/labels_detailled.pickle'
label_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\labels_detailled_suvr.pickle'
with open(label_path, 'rb') as f:
    labels = pickle.load(f)
excel_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\Scanner information.gz.xls'
with open(excel_path, 'rb') as f:
    scan_sheet = pd.read_excel(f)

scanners = []
amyloid = []
rcf = []
img_ids = []
names = []
slices = []
site = []
suvr = []
subject_ids = []
scan_keys = np.array(scan_sheet['Scanner'])
scanner_keys = np.array([','.join(f.split(',')[:-1]) for f in scan_keys])
scan_numbers = np.array(scan_sheet['Type'])
for k, v in labels.items():
    scanners.append(f"{v['manufacturer']}, {v['model']}")
    amyloid.append(v['label'])
    rcf.append([v['rows'], v['columns'], v['frames']])
    img_ids.append(v['img_id'])
    names.append(k)
    slices.append(v['slices'])
    site.append(v['site'])
    suvr.append(v['label_suvr'])
    subject_ids.append(v['rid'])

site = np.array(site)
names = np.array(names)
slices = np.array(slices)
img_ids = np.array(img_ids)
scanners = np.array(scanners)
amyloid = np.array(amyloid)
rcf = np.array(rcf)
subject_ids = np.array(subject_ids)
suvr = np.array(suvr)
types = []

for s in scanners:
    t = scan_numbers[scanner_keys == s]
    types.append(int(t))

for n, t in zip(names, types):
    labels[n]['scanner_type'] = t

csv_filepath = os.path.join(os.path.dirname(label_path), 'site_analysis.csv')
writer = CsvWriter(csv_filepath, header=['site', 'num_samples', 'num_amyloid_positive', 'num_amyloid_negative', 'percentage_amyloid_positive',
                                         'avg_suvr', 'min_suvr', 'max_suvr', 'std_suvr'])

s_amyloid = amyloid
s_suvr = suvr
s = 'all'
print(f'Site {s}: positive amyloid: {s_amyloid.sum()}, negative amyloid: {len(s_amyloid) - s_amyloid.sum()}')
num_samples = len(s_amyloid)
num_amyloid_positive = s_amyloid.sum()
num_amyloid_negative = len(s_amyloid) - s_amyloid.sum()
percentage_amyloid_positive = num_amyloid_positive / num_samples
avg_suvr = s_suvr.mean()
min_suvr = s_suvr.min()
max_suvr = s_suvr.max()
std_suvr = s_suvr.std()
row = {'site': s, 'num_samples': num_samples, 'num_amyloid_positive': num_amyloid_positive,
       'num_amyloid_negative': num_amyloid_negative,
       'percentage_amyloid_positive': percentage_amyloid_positive, 'avg_suvr': avg_suvr, 'min_suvr': min_suvr,
       'max_suvr': max_suvr,
       'std_suvr': std_suvr}
writer.write_row(**row)
for s in np.unique(site):
    s_amyloid = amyloid[site == s]
    s_suvr = suvr[site == s]
    print(f'Site {s}: positive amyloid: {s_amyloid.sum()}, negative amyloid: {len(s_amyloid)-s_amyloid.sum()}')
    num_samples = len(s_amyloid)
    num_amyloid_positive = s_amyloid.sum()
    num_amyloid_negative = len(s_amyloid) - s_amyloid.sum()
    percentage_amyloid_positive = num_amyloid_positive/num_samples
    avg_suvr = s_suvr.mean()
    min_suvr = s_suvr.min()
    max_suvr = s_suvr.max()
    std_suvr = s_suvr.std()
    row = {'site': s, 'num_samples': num_samples, 'num_amyloid_positive': num_amyloid_positive, 'num_amyloid_negative': num_amyloid_negative,
           'percentage_amyloid_positive': percentage_amyloid_positive, 'avg_suvr': avg_suvr, 'min_suvr': min_suvr, 'max_suvr': max_suvr,
           'std_suvr': std_suvr}
    writer.write_row(**row)

fig = plt.figure()
plt.title('scanners used in data')
plt.hist(scanners)
plt.savefig(os.path.join(os.path.dirname(label_path), 'scanners'))
fig = plt.figure()
plt.title('amyloid status in data')
plt.hist(amyloid)
plt.savefig(os.path.join(os.path.dirname(label_path), 'amyloid'))

print('nice!')