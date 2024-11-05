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
rho_list[0] = 0.5
rho_list[-1] = 32
default_L_b = 100
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
default_ppd = 60
multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)

castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_rho_sensitivity_data.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_rho_list = castleCSF_result_data['rho_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']
middle_rho = 10 ** ((np.log10(rho_list[0]) + np.log10(rho_list[-1])) / 2)
middle_rho_S = np.interp(middle_rho, castleCSF_result_rho_list, castleCSF_result_sensitivity_list)

T_vid, R_vid = generate_band_lim_noise(W=default_W, H=default_H, freq_band=middle_rho,
                                       L_b=default_L_b, contrast=1 / middle_rho_S, ppd=default_ppd)
T_vid_f, R_vid_f = generate_band_lim_noise_fix_random_seed(W=default_W, H=default_H, freq_band=middle_rho,
                                                           L_b=default_L_b, contrast=1 / middle_rho_S,
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
aim_JOD, m_stats = cvvdp.predict(T_vid_ct, R_vid_ct, dim_order="HWC")
aim_JOD_f, m_stats_f = cvvdp.predict(T_vid_ct_f, R_vid_ct_f, dim_order="HWC")

Spearman_matrix_JOD = np.zeros([len(rho_list), len(multiplier_list)])
Spearman_matrix_JOD_f = np.zeros([len(rho_list), len(multiplier_list)])
for rho_index, rho_value in tqdm(enumerate(rho_list)):
    S_gt = np.interp(rho_value, castleCSF_result_rho_list, castleCSF_result_sensitivity_list)
    for multiplier_index, multiplier_value in enumerate(multiplier_list):
        S_test = multiplier_value * S_gt
        T_vid, R_vid = generate_band_lim_noise(W=default_W, H=default_H, freq_band=rho_value,
                                               L_b=default_L_b, contrast=1 / S_test, ppd=default_ppd)
        T_vid_f, R_vid_f = generate_band_lim_noise_fix_random_seed(W=default_W, H=default_H, freq_band=rho_value,
                                                                   L_b=default_L_b, contrast=1 / S_test,
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
        Spearman_matrix_JOD[rho_index, multiplier_index] = JOD
        Spearman_matrix_JOD_f[rho_index, multiplier_index] = JOD_f

json_file_name = r'new_data_logs\cvvdp\different_rho/cvvdp_test_on_band_lim_noise_different_rho_band_contour_plot_ppd_60_final.json'
save_json_file_name = r'new_data_logs\cvvdp\different_rho/cvvdp_test_on_band_lim_noise_different_rho_band_contour_plot_ppd_60_final_aim.json'
with open(json_file_name, 'r') as fp:
    json_data = json.load(fp)
json_data['aim_JOD'] = float(aim_JOD)
json_data['aim_JOD_f'] = float(aim_JOD_f)
json_data['rho_list'] = rho_list.tolist()
json_data['multiplier_list'] = multiplier_list.tolist()
json_data['Spearman_matrix_JOD'] = Spearman_matrix_JOD.tolist()
json_data['Spearman_matrix_JOD_f'] = Spearman_matrix_JOD_f.tolist()

with open(save_json_file_name, 'w') as fp:
    json.dump(json_data, fp)
