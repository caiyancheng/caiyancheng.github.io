import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import numpy as np
import torch
from Contrast_masking_generator.generate_plot_contrast_masking_gabor_on_noise import generate_contrast_masking_gabor_on_noise
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
save_root_path = 'new_data_logs/baseline/different_c'
os.makedirs(save_root_path, exist_ok=True)

default_W = 224
default_H = 224
default_Mask_upper_frequency = 12
default_L_b = 100
contrast_mask_list = np.logspace(np.log10(0.005), np.log10(0.5), 20)
contrast_test_list = np.logspace(np.log10(0.01), np.log10(0.5), 20)
default_ppd = 60
default_radius_test = 0.8
default_rho_test = 1.2

json_plot_data = {}
json_plot_data['contrast_mask_matrix'] = []
json_plot_data['contrast_test_matrix'] = []
json_plot_data['L1_similarity_matrix'] = []
json_plot_data['L2_similarity_matrix'] = []
json_plot_data['cos_similarity_matrix'] = []

plot_contrast_mask_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
plot_contrast_test_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
plot_L1_similarity_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
plot_L2_similarity_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
plot_cos_similarity_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])

for contrast_mask_index in tqdm(range(len(contrast_mask_list))):
    contrast_mask_value = contrast_mask_list[contrast_mask_index]
    for contrast_test_index in range(len(contrast_test_list)):
        contrast_test_value = contrast_test_list[contrast_test_index]
        plot_contrast_mask_matrix[contrast_mask_index, contrast_test_index] = contrast_mask_value
        plot_contrast_test_matrix[contrast_mask_index, contrast_test_index] = contrast_test_value
        T_vid, R_vid = generate_contrast_masking_gabor_on_noise(W=default_W, H=default_H,
                                                                sigma=default_radius_test,
                                                                rho=default_rho_test,
                                                                Mask_upper_frequency=default_Mask_upper_frequency,
                                                                L_b=default_L_b,
                                                                contrast_mask=contrast_mask_value,
                                                                contrast_test=contrast_test_value,
                                                                ppd=default_ppd)
        T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
        R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
        T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32)[None, None, ...]
        R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32)[None, None, ...]
        T_vid_ct = T_vid_ct.expand(-1, 3, -1, -1).cuda()
        R_vid_ct = R_vid_ct.expand(-1, 3, -1, -1).cuda()
        test_feature = T_vid_ct
        reference_feature = R_vid_ct
        L1_similarity = float(torch.norm(test_feature - reference_feature, p=1).cpu())
        L2_similarity = float(torch.norm(test_feature - reference_feature, p=2).cpu())
        cos_similarity = float(F.cosine_similarity(test_feature.reshape(1, -1), reference_feature.reshape(1, -1)).cpu())
        plot_L1_similarity_matrix[contrast_mask_index, contrast_test_index] = L1_similarity
        plot_L2_similarity_matrix[contrast_mask_index, contrast_test_index] = L2_similarity
        plot_cos_similarity_matrix[contrast_mask_index, contrast_test_index] = cos_similarity
json_plot_data['contrast_test_matrix'].append(plot_contrast_test_matrix.tolist())
json_plot_data['contrast_mask_matrix'].append(plot_contrast_mask_matrix.tolist())
json_plot_data['L1_similarity_matrix'].append(plot_L1_similarity_matrix.tolist())
json_plot_data['L2_similarity_matrix'].append(plot_L2_similarity_matrix.tolist())
json_plot_data['cos_similarity_matrix'].append(plot_cos_similarity_matrix.tolist())

with open(os.path.join(save_root_path,
                       f'baseline_test_on_contrast_masking_different_c_contour_plot_ppd_{default_ppd}_final.json'),
          'w') as fp:
    json.dump(json_plot_data, fp)
