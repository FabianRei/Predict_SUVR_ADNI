import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
from torch import nn
from dl.neural_network.resnet import ResNet50, ResNet50Reg
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
from glob import glob
import nibabel as nib
from matplotlib import pyplot as plt


def detach(arr, transpose=False):
    arr = arr.cpu().numpy()
    if transpose:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


device = 'cuda' if torch.cuda.is_available() else 'cpu'
fname = 'C:\\Users\\Fabian\\stanford\\fed_learning\\federated_learning_data\\trial_sample\\nifti\\941_S_4377\\AV45_Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution\\2012-02-08_15_45_34.0\\I283764\\ADNI_941_S_4377_PT_AV45_Coreg,_Avg,_Std_Img_and_Vox_Siz,_Uniform_Resolution_Br_20120210083545715_46_S140216_I283764.nii'
prefix = '\\\\?\\'
img = nib.load(prefix + fname)
arr = img.get_fdata()
# arr = arr[:, :, 50, 0]
arr = arr[:, :, 50, 0]
img = np.copy(arr)
data_mean = 0.5600362821380893
data_std = 0.7418505925670522
arr -= data_mean
arr /= data_std
arr = torch.tensor(arr)
arr = arr.view(-1, 1, 160, 160)
arr = arr.cuda()
arr = arr.type(torch.float32)
img = torch.tensor(img)
img = img.cuda()
img = img.type(torch.float32)
model_path = r'C:\Users\Fabian\stanford\fed_learning\rsync\fl\experiments\incl_subjects_site_one_slices_dataset_full\resnet_model_08-30_17-48_lr_0_001_pretrained_30epochs_rod_0_1_da_10.pth'
resnet = torch.load(model_path)
# resnet = nn.Sequential(*list(resnet.children())[:-2])
configs = [dict(model_type='resnet', arch=resnet, layer_name='layer4')]
for config in configs:
    config['arch'].to(device).eval()

cams = [
    [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
    for config in configs
]
images = []
for gradcam, gradcam_pp in cams:
    mask, _ = gradcam(arr)
    test = mask.cpu().numpy()[0, 0, :, :]
    heatmap, result = visualize_cam(mask, img)
    result = detach(result, transpose=True)
    hm = detach(heatmap, transpose=True)
#     mask_pp, _ = gradcam_pp(normed_torch_img)
#     heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)
#
#     images.extend([torch_img.cpu(), heatmap, heatmap_pp, result, result_pp])
#
# grid_image = make_grid(images, nrow=5)
print('nice')
