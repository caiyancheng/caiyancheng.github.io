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
from display_encoding import display_encode

display_encode_tool = display_encode(400)

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt

# Only test cpd right now
save_root_path = 'new_data_logs/cvvdp/different_rho_RG'
os.makedirs(save_root_path, exist_ok=True)

default_W = 224
default_H = 224
default_R = 1
rho_list = np.logspace(np.log10(0.5), np.log10(32), 20)
rho_list[0] = 0.5
rho_list[-1] = 32
contrast_list = np.logspace(np.log10(0.001), np.log10(0.2), 20)
default_O = 0
default_L_b = 100
default_ppd = 60
multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)

castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_rho_sensitivity_data_RG.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_rho_list = castleCSF_result_data['rho_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']
middle_rho = 10 ** ((np.log10(rho_list[0]) + np.log10(rho_list[-1])) / 2)
middle_rho_S = np.interp(middle_rho, castleCSF_result_rho_list, castleCSF_result_sensitivity_list)
T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=default_R, rho=middle_rho,
                                    O=default_O, L_b=default_L_b, contrast=1 / middle_rho_S,
                                    ppd=default_ppd, color_direction='rg')
T_vid_c = display_encode_tool.L2C_sRGB(T_vid) * 255
R_vid_c = display_encode_tool.L2C_sRGB(R_vid) * 255
T_vid_ct = torch.tensor(T_vid_c, dtype=torch.uint8)
R_vid_ct = torch.tensor(R_vid_c, dtype=torch.uint8)
aim_JOD, m_stats = cvvdp.predict(T_vid_ct, R_vid_ct, dim_order="HWC")

# 下面是计算Spearman Correlation部分
Spearman_matrix_JOD = np.zeros([len(rho_list), len(multiplier_list)])
for rho_index, rho_value in tqdm(enumerate(rho_list)):
    S_gt = np.interp(rho_value, castleCSF_result_rho_list, castleCSF_result_sensitivity_list)
    for multiplier_index, multiplier_value in enumerate(multiplier_list):
        S_test = multiplier_value * S_gt
        T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=default_R, rho=rho_value,
                                            O=default_O,
                                            L_b=default_L_b, contrast=1 / S_test, ppd=default_ppd,
                                            color_direction='rg')
        T_vid_c = display_encode_tool.L2C_sRGB(T_vid) * 255
        R_vid_c = display_encode_tool.L2C_sRGB(R_vid) * 255
        T_vid_ct = torch.tensor(T_vid_c, dtype=torch.uint8)
        R_vid_ct = torch.tensor(R_vid_c, dtype=torch.uint8)
        JOD, m_stats = cvvdp.predict(T_vid_ct, R_vid_ct, dim_order="HWC")
        Spearman_matrix_JOD[rho_index, multiplier_index] = float(JOD)

json_file_name = r'new_data_logs\cvvdp\different_rho_RG/cvvdp_test_on_gabors_different_rho_contour_plot_ppd_60_RG_final.json'
save_json_file_name = r'new_data_logs\cvvdp\different_rho_RG/cvvdp_test_on_gabors_different_rho_contour_plot_ppd_60_RG_final_aim.json'
with open(json_file_name, 'r') as fp:
    json_data = json.load(fp)
json_data['aim_JOD'] = float(aim_JOD)
json_data['rho_RG_list'] = rho_list.tolist()
json_data['multiplier_list'] = multiplier_list.tolist()
json_data['Spearman_matrix_JOD'] = Spearman_matrix_JOD.tolist()

with open(save_json_file_name, 'w') as fp:
    json.dump(json_data, fp)

