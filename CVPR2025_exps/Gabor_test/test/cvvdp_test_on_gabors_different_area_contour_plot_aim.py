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
import math

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt
from display_encoding import display_encode

display_encode_tool = display_encode(400)

# Only test cpd right now
# Dino input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/cvvdp/different_area'
os.makedirs(save_root_path, exist_ok=True)

default_W = 224
default_H = 224
R_min = 0.1
R_max = 1
Area_list = np.logspace(np.log10(math.pi * R_min ** 2), np.log10(math.pi * R_max ** 2), 20)
Area_list[0] = math.pi * R_min ** 2
Area_list[-1] = math.pi * R_max ** 2
R_list = (Area_list / math.pi) ** 0.5
# rho_list = [0.5, 1, 2, 4, 8, 16, 32]
default_rho = 8
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
default_O = 0
default_contrast = 1
default_L_b = 100
default_ppd = 60
multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)

## 如果要refit需要寻找到的目标值
castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_area_sensitivity_data.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_area_list = castleCSF_result_data['area_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']
middle_Area = 10 ** ((np.log10(Area_list[0]) + np.log10(Area_list[-1])) / 2)
middle_Area_S = np.interp(middle_Area, castleCSF_result_area_list, castleCSF_result_sensitivity_list)
T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=(middle_Area / math.pi) ** 0.5, rho=default_rho,
                                    O=default_O,
                                    L_b=default_L_b, contrast=1 / middle_Area_S, ppd=default_ppd,
                                    color_direction='ach')
T_vid_c = display_encode_tool.L2C_sRGB(T_vid) * 255
R_vid_c = display_encode_tool.L2C_sRGB(R_vid) * 255
T_vid_ct = torch.tensor(T_vid_c, dtype=torch.uint8)
R_vid_ct = torch.tensor(R_vid_c, dtype=torch.uint8)
aim_JOD, m_stats = cvvdp.predict(T_vid_ct, R_vid_ct, dim_order="HWC")

# 下面是计算Spearman Correlation部分
Spearman_matrix_JOD = np.zeros([len(Area_list), len(multiplier_list)])
for Area_index, Area_value in tqdm(enumerate(Area_list)):
    S_gt = np.interp(Area_value, castleCSF_result_area_list, castleCSF_result_sensitivity_list)
    for multiplier_index, multiplier_value in enumerate(multiplier_list):
        S_test = multiplier_value * S_gt
        T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=(Area_value / math.pi) ** 0.5, rho=default_rho,
                                            O=default_O,
                                            L_b=default_L_b, contrast=1 / S_test, ppd=default_ppd,
                                            color_direction='ach')
        T_vid_c = display_encode_tool.L2C_sRGB(T_vid) * 255
        R_vid_c = display_encode_tool.L2C_sRGB(R_vid) * 255
        T_vid_ct = torch.tensor(T_vid_c, dtype=torch.uint8)
        R_vid_ct = torch.tensor(R_vid_c, dtype=torch.uint8)
        JOD, m_stats = cvvdp.predict(T_vid_ct, R_vid_ct, dim_order="HWC")
        Spearman_matrix_JOD[Area_index, multiplier_index] = float(JOD)

json_file_name = r'new_data_logs\cvvdp\different_area/cvvdp_test_on_gabors_different_area_contour_plot_ppd_60_final.json'
save_json_file_name = r'new_data_logs\cvvdp\different_area/cvvdp_test_on_gabors_different_area_contour_plot_ppd_60_final_aim.json'
with open(json_file_name, 'r') as fp:
    json_data = json.load(fp)
json_data['aim_JOD'] = float(aim_JOD)
json_data['area_list'] = Area_list.tolist()
json_data['multiplier_list'] = multiplier_list.tolist()
json_data['Spearman_matrix_JOD'] = Spearman_matrix_JOD.tolist()

with open(save_json_file_name, 'w') as fp:
    json.dump(json_data, fp)
