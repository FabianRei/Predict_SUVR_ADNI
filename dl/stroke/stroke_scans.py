import nibabel as nib
import numpy as np
from glob import glob

super_path = r'C:\Users\Fabian\Desktop\sample_lesion'
lesions = glob(f'{super_path}\**\LESION.nii')
scans = glob(f'{super_path}\**\LESION.nii')

path_lesion = r'C:\Users\Fabian\Desktop\sample_lesion\00025\LESION.nii'
path_scan = r'C:\Users\Fabian\Desktop\sample_lesion\00025\FUDWI.nii'

l_slices = []
l_percentage = []
for lesion in lesions:
    les = nib.load(lesion).get_fdata()
    x = 0; y = 0; z = 0
    print(lesion)
    print(les.shape)
    print('is nan freq:', np.isnan(les).sum(), np.isnan(les).sum()/len(les.flatten()))
    les[np.isnan(les)] = 0
    les[les>0.5] = 1
    les[les<0.5] = 0
    for i in range(les.shape[0]):
        if np.sum(les[i, :, :]) > 0:
            x += 1
    for i in range(les.shape[1]):
        if np.sum(les[:, i, :]) > 0:
            y += 1
    for i in range(les.shape[2]):
        if np.sum(les[:, :, i]) > 0:
            z += 1
    l_slices.append((x, y, z))
    print(l_slices[-1])
    les = les.flatten()
    l_percentage.append(np.sum(les>0)/len(les))
    print('lesion volume:', l_percentage[-1])

scan = nib.load(path_scan)
lesion = nib.load(path_lesion)

l = lesion.flatten()
np.sum(l>0)/len(l)
print('db')