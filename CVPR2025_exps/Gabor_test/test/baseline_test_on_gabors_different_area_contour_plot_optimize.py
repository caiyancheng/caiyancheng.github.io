import sys

sys.path.append('E:\Py_codes\LVM_Comparision')
import numpy as np
import torch
from Gabor_test_stimulus_generator.generate_plot_gabor_functions_new import generate_gabor_patch
import torch.nn.functional as F
import pandas as pd
import json
from tqdm import tqdm
import os
from scipy.optimize import minimize_scalar, root_scalar, fsolve
import math

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt
from display_encoding import display_encode

display_encode_tool = display_encode(400)

# Only test cpd right now
# Dino input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/baseline/different_area'
os.makedirs(save_root_path, exist_ok=True)

default_W = 224
default_H = 224
R_min = 0.1
R_max = 1
Area_list = np.logspace(np.log10(math.pi * R_min ** 2), np.log10(math.pi * R_max ** 2), 20)
R_list = (Area_list / math.pi) ** 0.5
# rho_list = [0.5, 1, 2, 4, 8, 16, 32]
default_rho = 8
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
default_O = 0
default_contrast = 1
default_L_b = 100
default_ppd = 60

## 如果要refit需要寻找到的目标值
castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_area_sensitivity_data.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_area_list = castleCSF_result_data['area_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']
middle_Area = 10**((np.log10(Area_list[0])+np.log10(Area_list[-1]))/2)
middle_Area_S = np.interp(middle_Area, castleCSF_result_area_list, castleCSF_result_sensitivity_list)
T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=(middle_Area / math.pi) ** 0.5, rho=default_rho, O=default_O,
                                    L_b=default_L_b, contrast=1/middle_Area_S, ppd=default_ppd,
                                    color_direction='ach')
T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
test_feature = T_vid_ct
reference_feature = R_vid_ct
aim_cos_similarity = float(F.cosine_similarity(test_feature.reshape(1,-1), reference_feature.reshape(1,-1)).cpu())

def T_optimize_target(sensitivity, R_value, aim_score):
    T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=R_value, rho=default_rho, O=default_O,
                                        L_b=default_L_b, contrast=1 / sensitivity, ppd=default_ppd,
                                        color_direction='ach')
    T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
    R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
    T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
    R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
    test_feature = T_vid_ct
    reference_feature = R_vid_ct
    cos_similarity = float(F.cosine_similarity(test_feature.reshape(1, -1), reference_feature.reshape(1, -1)))
    return cos_similarity - aim_score

json_file_name = r'new_data_logs\baseline\different_area/baseline_test_on_gabors_different_area_contour_plot_ppd_60_final.json'
with open(json_file_name, 'r') as fp:
    json_data = json.load(fp)
json_data['optimize_x_area'] = []
json_data['optimize_y_sensitivity'] = []
json_data['gt_y_sensitivity'] = []

for Area_index, Area_value in tqdm(enumerate(Area_list)):
    R_value = (Area_value / np.pi) ** 0.5
    # if not Area_index % 10 == 0:
    #     continue
    # if abs(Area_value - middle_Area) < 0.1:
    #     X = 1
    S_bounds = (1, 1000)
    target_function = lambda test_sensitivity: T_optimize_target(
        sensitivity=test_sensitivity, R_value=R_value, aim_score=aim_cos_similarity
    )
    S_try_values = np.logspace(np.log10(S_bounds[0]), np.log10(S_bounds[1]), num=100)
    S_try_score_list = []
    S_optimize_result = -1
    for S_value in S_try_values:
        score = target_function(S_value)
        S_try_score_list.append(score)
        if abs(score) < 10e-4:
            print('OK!')
            S_optimize_result = S_value
            break
    if S_optimize_result == -1:
        min_score_index = min(enumerate(S_try_score_list), key=lambda x: abs(x[1]))[0]
        S_optimize_result = S_try_values[min_score_index]

    json_data['optimize_x_area'].append(Area_value)
    json_data['optimize_y_sensitivity'].append(S_optimize_result)

with open(json_file_name, 'w') as fp:
    json.dump(json_data, fp)
