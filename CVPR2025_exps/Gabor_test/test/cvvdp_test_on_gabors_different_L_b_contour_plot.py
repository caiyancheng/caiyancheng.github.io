import sys

sys.path.append('E:\Py_codes\LVM_Comparision')
import pycvvdp
cvvdp = pycvvdp.cvvdp(display_name='standard_4k')
import numpy as np
import torch
from Gabor_test_stimulus_generator.generate_plot_gabor_functions_new import generate_gabor_patch
import torch.nn.functional as F
import pandas as pd
import json
from tqdm import tqdm
import os

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt
from display_encoding import display_encode

display_encode_tool = display_encode(400)
# Only test cpd right now
# Dinov2 input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/cvvdp/different_luminance'
os.makedirs(save_root_path, exist_ok=True)

default_W = 224
default_H = 224
default_R = 1
default_rho = 2
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
default_O = 0
default_contrast = 1
L_b_list = np.logspace(np.log10(0.1), np.log10(200), 20)
default_ppd = 60

json_plot_data = {}
json_plot_data['L_b_matrix'] = []
json_plot_data['contrast_matrix'] = []
json_plot_data['JOD_score_matrix'] = []

plot_L_b_matrix = np.zeros([len(L_b_list), len(contrast_list)])
plot_contrast_matrix = np.zeros([len(L_b_list), len(contrast_list)])
plot_JOD_score_matrix = np.zeros([len(L_b_list), len(contrast_list)])

for L_b_index in tqdm(range(len(L_b_list))):
    L_b_value = L_b_list[L_b_index]
    for contrast_index in range(len(contrast_list)):
        contrast_value = contrast_list[contrast_index]
        plot_L_b_matrix[L_b_index, contrast_index] = L_b_value
        plot_contrast_matrix[L_b_index, contrast_index] = contrast_value
        T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=default_R, rho=default_rho, O=default_O,
                                            L_b=L_b_value, contrast=contrast_value, ppd=default_ppd,
                                            color_direction='ach')
        T_vid_c = display_encode_tool.L2C_sRGB(T_vid) * 255
        R_vid_c = display_encode_tool.L2C_sRGB(R_vid) * 255
        T_vid_ct = torch.tensor(T_vid_c, dtype=torch.uint8)
        R_vid_ct = torch.tensor(R_vid_c, dtype=torch.uint8)
        JOD, m_stats = cvvdp.predict(T_vid_ct, R_vid_ct, dim_order="HWC")
        plot_JOD_score_matrix[L_b_index, contrast_index] = JOD
json_plot_data['L_b_matrix'].append(plot_L_b_matrix.tolist())
json_plot_data['contrast_matrix'].append(plot_contrast_matrix.tolist())
json_plot_data['JOD_score_matrix'].append(plot_JOD_score_matrix.tolist())
with open(os.path.join(save_root_path, f'cvvdp_test_on_gabors_different_L_b_contour_plot_ppd_{default_ppd}_final.json'), 'w') as fp:
    json.dump(json_plot_data, fp)

