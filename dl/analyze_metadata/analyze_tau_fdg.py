import pandas as pd
import numpy as np



def get_csv_column(csv_path, col_name, sort_by=None, exclude_from=None):
    df = pd.read_csv(csv_path, delimiter=',', parse_dates=True)
    col = df[col_name].tolist()
    col = np.array(col)
    if sort_by is not None:
        sort_val = get_csv_column(csv_path, sort_by)
        sort_idxs = np.argsort(sort_val)
        col = col[sort_idxs]
    if exclude_from is not None:
        sort_val = sort_val[sort_idxs]
        col = col[sort_val >= exclude_from]
    return col

def get_posix_col(csv_path, col_name, **kwargs):
    col = get_csv_column(csv_path, col_name, **kwargs)
    col = [pd.Timestamp(c).timestamp() for c in col]
    col = np.array(col)
    return col

# csv_path = r'C:\Users\Fabian\Dropbox\amtau\UCBERKELEYAV1451_PVC_02_04_20.csv'
csv_path = r'C:\Users\Fabian\Dropbox\amtau\UCBERKELEYFDG_02_04_20.csv'
col_name = 'RID'


col = get_csv_column(csv_path, col_name)
# get dates
dates = get_posix_col(csv_path, 'EXAMDATE')

# get and combine ROI values
roi_name = get_csv_column(csv_path, 'ROINAME')
roi_lat = get_csv_column(csv_path, 'ROILAT')
roi_comb = []
for n, l in zip(roi_name, roi_lat):
    roi_comb.append(f'{n} {l}')
roi_comb = np.array(roi_comb)


UIDs = get_csv_column(csv_path, 'UID')
n, c = np.unique(col, return_counts=True)
un, uc = np.unique(UIDs, return_counts=True)

test = roi_comb[UIDs == UIDs[12]]
test.sort()

# do all UID scans contain the same regions?
for u in UIDs:
    te = roi_comb[UIDs==u]
    te.sort()
    if not (test == te).all():
        print(te)


year_seconds = 365*24*3600

# get max follow up time for subjects
max_diffs = []
for sub in n:
    dats = dates[sub==col]
    dats -= dats.min()
    max_diffs.append(max(dats))
mds = np.array(max_diffs)
print('db')