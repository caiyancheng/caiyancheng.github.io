import sys

sys.path.append('E:\Py_codes\LVM_Comparision')
import pycvvdp
cvvdp = pycvvdp.cvvdp(display_name='standard_4k')
import numpy as np
import torch
from Band_limit_noise_generator.generate_plot_band_lim_noise import generate_band_lim_noise, \
    generate_band_lim_noise_fix_random_seed
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
# input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/cvvdp/different_rho'
os.makedirs(save_root_path, exist_ok=True)

default_W = 224
default_H = 224
rho_list = np.logspace(np.log10(0.5), np.log10(32), 20)
default_L_b = 100
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
default_ppd = 60

json_plot_data = {}
json_plot_data['rho_matrix'] = []
json_plot_data['contrast_matrix'] = []
json_plot_data['JOD_score_matrix'] = []
json_plot_data['JOD_score_matrix_fix_random_seed'] = []

plot_rho_matrix = np.zeros([len(rho_list), len(contrast_list)])
plot_contrast_matrix = np.zeros([len(rho_list), len(contrast_list)])
plot_JOD_score_matrix = np.zeros([len(rho_list), len(contrast_list)])
plot_JOD_score_matrix_f = np.zeros([len(rho_list), len(contrast_list)])

for rho_index in tqdm(range(len(rho_list))):
    rho_value = rho_list[rho_index]
    for contrast_index in range(len(contrast_list)):
        contrast_value = contrast_list[contrast_index]
        plot_rho_matrix[rho_index, contrast_index] = rho_value
        plot_contrast_matrix[rho_index, contrast_index] = contrast_value
        T_vid, R_vid = generate_band_lim_noise(W=default_W, H=default_H, freq_band=rho_value,
                                               L_b=default_L_b, contrast=contrast_value, ppd=default_ppd)
        T_vid_f, R_vid_f = generate_band_lim_noise_fix_random_seed(W=default_W, H=default_H, freq_band=rho_value,
                                                                   L_b=default_L_b, contrast=contrast_value,
                                                                   ppd=default_ppd)
        T_vid = np.stack([T_vid] * 3, axis=-1)
        R_vid = np.stack([R_vid] * 3, axis=-1)
        T_vid_f = np.stack([T_vid_f] * 3, axis=-1)
        R_vid_f = np.stack([R_vid_f] * 3, axis=-1)
        T_vid_c = display_encode_tool.L2C_sRGB(T_vid) * 255
        R_vid_c = display_encode_tool.L2C_sRGB(R_vid) * 255
        T_vid_c_f = display_encode_tool.L2C_sRGB(T_vid_f) * 255
        R_vid_c_f = display_encode_tool.L2C_sRGB(R_vid_f) * 255
        T_vid_ct = torch.tensor(T_vid_c, dtype=torch.uint8)
        R_vid_ct = torch.tensor(R_vid_c, dtype=torch.uint8)
        T_vid_ct_f = torch.tensor(T_vid_c_f, dtype=torch.uint8)
        R_vid_ct_f = torch.tensor(R_vid_c_f, dtype=torch.uint8)
        JOD, m_stats = cvvdp.predict(T_vid_ct, R_vid_ct, dim_order="HWC")
        JOD_f, m_stats_f = cvvdp.predict(T_vid_ct_f, R_vid_ct_f, dim_order="HWC")
        plot_JOD_score_matrix[rho_index, contrast_index] = JOD
        plot_JOD_score_matrix_f[rho_index, contrast_index] = JOD_f
json_plot_data['rho_matrix'].append(plot_rho_matrix.tolist())
json_plot_data['contrast_matrix'].append(plot_contrast_matrix.tolist())
json_plot_data['JOD_score_matrix'].append(plot_JOD_score_matrix.tolist())
json_plot_data['JOD_score_matrix_fix_random_seed'].append(plot_JOD_score_matrix_f.tolist())

with open(os.path.join(save_root_path,
                       f'cvvdp_test_on_band_lim_noise_different_rho_band_contour_plot_ppd_{default_ppd}_final.json'),
          'w') as fp:
    json.dump(json_plot_data, fp)
