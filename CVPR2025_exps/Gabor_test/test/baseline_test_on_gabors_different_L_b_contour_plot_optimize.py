import sys

sys.path.append('E:\Py_codes\LVM_Comparision')
import numpy as np
import torch
from Gabor_test_stimulus_generator.generate_plot_gabor_functions_new import generate_gabor_patch
import torch.nn.functional as F
import pandas as pd
import json
from tqdm import tqdm
from scipy.optimize import minimize_scalar, root_scalar
import os

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt
from display_encoding import display_encode

display_encode_tool = display_encode(400)
# Only test cpd right now
# Dinov2 input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/baseline/different_luminance'
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

castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_luminance_sensitivity_data.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_luminance_list = castleCSF_result_data['luminance_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']
middle_Luminance = 10 ** ((np.log10(L_b_list[0]) + np.log10(L_b_list[-1])) / 2)
middle_Luminance_S = np.interp(middle_Luminance, castleCSF_result_luminance_list, castleCSF_result_sensitivity_list)
T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=default_R, rho=default_rho, O=default_O,
                                    L_b=middle_Luminance, contrast=1/middle_Luminance_S, ppd=default_ppd,
                                    color_direction='ach')
T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
test_feature = T_vid_ct
reference_feature = R_vid_ct
aim_cos_similarity = float(F.cosine_similarity(test_feature.reshape(1, -1), reference_feature.reshape(1, -1)).cpu())

json_plot_data = {}
json_plot_data['L_b_matrix'] = []
json_plot_data['contrast_matrix'] = []
json_plot_data['L1_similarity_matrix'] = []
json_plot_data['L2_similarity_matrix'] = []
json_plot_data['cos_similarity_matrix'] = []

plot_L_b_matrix = np.zeros([len(L_b_list), len(contrast_list)])
plot_contrast_matrix = np.zeros([len(L_b_list), len(contrast_list)])
plot_L1_similarity_matrix = np.zeros([len(L_b_list), len(contrast_list)])
plot_L2_similarity_matrix = np.zeros([len(L_b_list), len(contrast_list)])
plot_cos_similarity_matrix = np.zeros([len(L_b_list), len(contrast_list)])

def T_optimize_target(sensitivity, L_value, aim_score):
    T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=default_R, rho=default_rho, O=default_O,
                                        L_b=L_value, contrast=1 / sensitivity, ppd=default_ppd,
                                        color_direction='ach')
    T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
    R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
    T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
    R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
    test_feature = T_vid_ct
    reference_feature = R_vid_ct
    cos_similarity = float(F.cosine_similarity(test_feature.reshape(1, -1), reference_feature.reshape(1, -1)))
    return cos_similarity - aim_score

json_file_name = r'new_data_logs\baseline\different_luminance/baseline_test_on_gabors_different_L_b_contour_plot_ppd_60_final.json'
with open(json_file_name, 'r') as fp:
    json_data = json.load(fp)
json_data['optimize_x_luminance'] = []
json_data['optimize_y_sensitivity'] = []

for L_index, L_value in tqdm(enumerate(L_b_list)):
    S_bounds = (1, 1000)
    target_function = lambda test_sensitivity: T_optimize_target(
        sensitivity=test_sensitivity, L_value=L_value, aim_score=aim_cos_similarity
    )
    S_try_values = np.logspace(np.log10(S_bounds[0]), np.log10(S_bounds[1]), num=100)
    S_try_score_list = []
    S_optimize_result = -1
    for S_value in S_try_values:
        score = target_function(S_value)
        S_try_score_list.append(score)
        if abs(score) < 10e-5:
            print('OK!')
            S_optimize_result = S_value
            break
    if S_optimize_result == -1:
        min_score_index = min(enumerate(S_try_score_list), key=lambda x: abs(x[1]))[0]
        S_optimize_result = S_try_values[min_score_index]

    json_data['optimize_x_luminance'].append(L_value)
    json_data['optimize_y_sensitivity'].append(S_optimize_result)

with open(json_file_name, 'w') as fp:
    json.dump(json_data, fp)
