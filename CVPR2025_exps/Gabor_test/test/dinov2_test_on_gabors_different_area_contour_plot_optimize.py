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
# Dinov2 input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/dinov2/different_area'
os.makedirs(save_root_path, exist_ok=True)
all_backbone_list = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14',
                     'dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg']
# all_backbone_list = ['dinov2_vits14']

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

castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_area_sensitivity_data.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_area_list = castleCSF_result_data['area_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']
middle_Area = 10 ** ((np.log10(Area_list[0]) + np.log10(Area_list[-1])) / 2)
middle_Area_S = np.interp(middle_Area, castleCSF_result_area_list, castleCSF_result_sensitivity_list)

def T_optimize_target(backbone_model, sensitivity, R_value, aim_score):
    T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=R_value, rho=default_rho, O=default_O,
                                        L_b=default_L_b, contrast=1 / sensitivity, ppd=default_ppd,
                                        color_direction='ach')
    T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
    R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
    T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
    R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
    test_feature = backbone_model(T_vid_ct)
    reference_feature = backbone_model(R_vid_ct)
    cos_similarity = float(F.cosine_similarity(test_feature.reshape(1, -1), reference_feature.reshape(1, -1)))
    return cos_similarity - aim_score

json_file_name = r'new_data_logs\dinov2\different_area/dinov2_test_on_gabors_different_area_contour_plot_ppd_60_final.json'
with open(json_file_name, 'r') as fp:
    json_data = json.load(fp)
json_data['optimize_x_area_model_list'] = []
json_data['optimize_y_sensitivity_model_list'] = []

for backbone_name in tqdm(all_backbone_list):
    backbone_model = torch.hub.load('facebookresearch/dinov2', backbone_name)
    backbone_model.eval()
    backbone_model.cuda()

    T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=(middle_Area / math.pi) ** 0.5, rho=default_rho,
                                        O=default_O, L_b=default_L_b, contrast=1 / middle_Area_S, ppd=default_ppd,
                                        color_direction='ach')
    T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
    R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
    T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
    R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
    test_feature = backbone_model(T_vid_ct)
    reference_feature = backbone_model(R_vid_ct)
    aim_cos_similarity = float(F.cosine_similarity(test_feature.reshape(1, -1), reference_feature.reshape(1, -1)).cpu())

    optimize_x_area = []
    optimize_y_sensitivity = []

    for Area_index, Area_value in tqdm(enumerate(Area_list)):
        R_value = (Area_value / np.pi) ** 0.5
        S_bounds = (10, 1000)
        target_function = lambda test_sensitivity: T_optimize_target(
            backbone_model=backbone_model, sensitivity=test_sensitivity, R_value=R_value, aim_score=aim_cos_similarity
        ) * 1e6
        result = root_scalar(target_function, x0=float(middle_Area_S), method='newton')
        S_optimize_result = result.root
        if not S_optimize_result > 1:
            S_optimize_result = 1
        elif not S_optimize_result < 1000:
            S_optimize_result = 1000
        optimize_x_area.append(Area_value)
        optimize_y_sensitivity.append(S_optimize_result)
    json_data['optimize_x_area_model_list'].append(optimize_x_area)
    json_data['optimize_y_sensitivity_model_list'].append(optimize_y_sensitivity)

with open(json_file_name, 'w') as fp:
    json.dump(json_data, fp)
