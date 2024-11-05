import sys

sys.path.append('E:\Py_codes\LVM_Comparision')
import pycvvdp
cvvdp = pycvvdp.cvvdp(display_name='standard_4k')
import numpy as np
import torch
from Contrast_masking_generator.generate_plot_contrast_masking import generate_contrast_masking
import torch.nn.functional as F
import pandas as pd
import json
from tqdm import tqdm
import os
from display_encoding import display_encode

display_encode_tool = display_encode(400)

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt

# Only test cpd right now
# Dino input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/cvvdp/different_c'
os.makedirs(save_root_path, exist_ok=True)

default_W = 224
default_H = 224
default_rho = 2
default_O = 0
default_L_b = 32
contrast_mask_list = np.logspace(np.log10(0.005), np.log10(0.5), 20)
contrast_test_list = np.logspace(np.log10(0.01), np.log10(0.5), 20)
default_ppd = 60
default_gabor_radius = 0.5

json_plot_data = {}
json_plot_data['contrast_mask_matrix'] = []
json_plot_data['contrast_test_matrix'] = []
json_plot_data['JOD_score_matrix'] = []

plot_contrast_mask_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
plot_contrast_test_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
plot_JOD_score_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])

for contrast_mask_index in tqdm(range(len(contrast_mask_list))):
    contrast_mask_value = contrast_mask_list[contrast_mask_index]
    for contrast_test_index in range(len(contrast_test_list)):
        contrast_test_value = contrast_test_list[contrast_test_index]
        plot_contrast_mask_matrix[contrast_mask_index, contrast_test_index] = contrast_mask_value
        plot_contrast_test_matrix[contrast_mask_index, contrast_test_index] = contrast_test_value
        T_vid, R_vid = generate_contrast_masking(W=default_W, H=default_H, rho=default_rho, O=default_O,
                                                 L_b=default_L_b, contrast_mask=contrast_mask_value,
                                                 contrast_test=contrast_test_value, ppd=default_ppd,
                                                 gabor_radius=default_gabor_radius)
        T_vid = np.stack([T_vid] * 3, axis=-1)
        R_vid = np.stack([R_vid] * 3, axis=-1)
        T_vid_c = display_encode_tool.L2C_sRGB(T_vid) * 255
        R_vid_c = display_encode_tool.L2C_sRGB(R_vid) * 255
        T_vid_ct = torch.tensor(T_vid_c, dtype=torch.uint8)
        R_vid_ct = torch.tensor(R_vid_c, dtype=torch.uint8)
        JOD, m_stats = cvvdp.predict(T_vid_ct, R_vid_ct, dim_order="HWC")
        plot_JOD_score_matrix[contrast_mask_index, contrast_test_index] = JOD
json_plot_data['contrast_test_matrix'].append(plot_contrast_test_matrix.tolist())
json_plot_data['contrast_mask_matrix'].append(plot_contrast_mask_matrix.tolist())
json_plot_data['JOD_score_matrix'].append(plot_JOD_score_matrix.tolist())
with open(os.path.join(save_root_path,
                       f'cvvdp_test_on_contrast_masking_different_c_contour_plot_ppd_{default_ppd}_final.json'),
          'w') as fp:
    json.dump(json_plot_data, fp)
