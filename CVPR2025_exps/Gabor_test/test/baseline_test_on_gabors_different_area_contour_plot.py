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

json_plot_data = {}
json_plot_data['radius_matrix'] = []
json_plot_data['area_matrix'] = []
json_plot_data['contrast_matrix'] = []
json_plot_data['L1_similarity_matrix'] = []
json_plot_data['L2_similarity_matrix'] = []
json_plot_data['cos_similarity_matrix'] = []

plot_radius_matrix = np.zeros([len(R_list), len(contrast_list)])
plot_area_matrix = np.zeros([len(R_list), len(contrast_list)])
plot_contrast_matrix = np.zeros([len(R_list), len(contrast_list)])
plot_L1_similarity_matrix = np.zeros([len(R_list), len(contrast_list)])
plot_L2_similarity_matrix = np.zeros([len(R_list), len(contrast_list)])
plot_cos_similarity_matrix = np.zeros([len(R_list), len(contrast_list)])

for R_index in tqdm(range(len(R_list))):
    R_value = R_list[R_index]
    A_value = Area_list[R_index]
    for contrast_index in range(len(contrast_list)):
        contrast_value = contrast_list[contrast_index]
        plot_radius_matrix[R_index, contrast_index] = R_value
        plot_area_matrix[R_index, contrast_index] = A_value
        plot_contrast_matrix[R_index, contrast_index] = contrast_value
        T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=R_value, rho=default_rho, O=default_O,
                                            L_b=default_L_b, contrast=contrast_value, ppd=default_ppd,
                                            color_direction='ach')
        T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
        R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
        T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
        R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
        test_feature = T_vid_ct
        reference_feature = R_vid_ct
        L1_similarity = float(torch.norm(test_feature - reference_feature, p=1).cpu())
        L2_similarity = float(torch.norm(test_feature - reference_feature, p=2).cpu())
        cos_similarity = float(F.cosine_similarity(test_feature.reshape(1,-1), reference_feature.reshape(1,-1)).cpu())
        plot_L1_similarity_matrix[R_index, contrast_index] = L1_similarity
        plot_L2_similarity_matrix[R_index, contrast_index] = L2_similarity
        plot_cos_similarity_matrix[R_index, contrast_index] = cos_similarity
json_plot_data['radius_matrix'].append(plot_radius_matrix.tolist())
json_plot_data['area_matrix'].append(plot_area_matrix.tolist())
json_plot_data['contrast_matrix'].append(plot_contrast_matrix.tolist())
json_plot_data['L1_similarity_matrix'].append(plot_L1_similarity_matrix.tolist())
json_plot_data['L2_similarity_matrix'].append(plot_L2_similarity_matrix.tolist())
json_plot_data['cos_similarity_matrix'].append(plot_cos_similarity_matrix.tolist())
with open(os.path.join(save_root_path, f'baseline_test_on_gabors_different_area_contour_plot_ppd_{default_ppd}_final.json'), 'w') as fp:
    json.dump(json_plot_data, fp)
