import numpy as np
import os
from glob import glob
import pickle
import h5py
import nibabel as nib
import re
import shutil

def get_fname(path_name):
    return os.path.splitext(os.path.basename(path_name))[0]


windows_db = False

if windows_db:
    fpath = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\trial_sample'
    prefix = '\\\\?\\'
    outpath = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\all_relevant_data'
    os.makedirs(outpath, exist_ok=True)
else:
    fpath = '/scratch/reith/fl/data'
    prefix = ''
    outpath = '/scratch/reith/fl/experiments/all_relevant_data'
    os.makedirs(outpath, exist_ok=True)

xml_path = os.path.join(fpath, 'xml')
nifti_path = os.path.join(fpath, 'nifti')
pickle_path = os.path.join(fpath, 'labels_detailled_suvr.pickle')

with open(pickle_path, 'rb') as p:
    pdata = pickle.load(p)

pickle_fnames = list(pdata.keys())
nifti_files = glob(f'{nifti_path}/**/*.nii', recursive=True)
xml_files = glob(f'{xml_path}/**/ADNI*.xml', recursive=True)
ids = [re.findall(r'I\d{3,20}', f)[-1] for f in xml_files]

xml_out = os.path.join(outpath, 'xml')
nifti_out = os.path.join(outpath, 'nifti')
os.makedirs(xml_out, exist_ok=True)
os.makedirs(nifti_out, exist_ok=True)
shutil.copy(pickle_path, outpath)
ids = np.array(ids)
xml_files = np.array(xml_files)
sizes = []
h5_file = h5py.File(os.path.join(outpath, 'slice_data.h5'), 'w')
# labels_amyloid = {}
# labels_suvr = {}
write_file = open(os.path.join(outpath, 'faulty_nii_files.txt'), 'w')
for i, f in enumerate(nifti_files):
    basename = get_fname(f)
    if basename in pickle_fnames:
        try:
            nii_image_id = re.findall(r'I\d{3,20}', basename)[-1]
            xml_file = xml_files[ids == nii_image_id][0]
            if windows_db:
                continue
            shutil.copy2(f, nifti_out+'/')
            shutil.copy2(xml_file, xml_out+'/')
        except Exception as e:
            print(f'{basename} sucks, error is: {e}')
            write_file.write(f'{basename}, {e} \n')
            continue
        # labels_amyloid[basename] = pdata[basename]['label']
        # labels_suvr[basename] = pdata[basename]['label_suvr']
        print(f"{i*100/len(nifti_files):.2f}%. I did {basename}")


write_file.close()
# with open(os.path.join(outpath, 'labels_amyloid.pickle'), 'wb') as f:
#     pickle.dump(labels_amyloid, f)
#
# with open(os.path.join(outpath, 'labels_suvr.pickle'), 'wb') as f:
#     pickle.dump(labels_suvr, f)
h5_file.close()
print('done!')

