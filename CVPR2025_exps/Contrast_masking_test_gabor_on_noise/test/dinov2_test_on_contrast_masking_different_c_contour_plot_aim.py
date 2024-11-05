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
# Dinov2 input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/dinov2/different_c'
os.makedirs(save_root_path, exist_ok=True)
all_backbone_list = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14',
                     'dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg']
# all_backbone_list = ['dinov2_vits14']

default_W = 224
default_H = 224
default_Mask_upper_frequency = 12
default_L_b = 32
contrast_mask_list = np.logspace(np.log10(0.005), np.log10(0.5), 20)
contrast_test_list = np.logspace(np.log10(0.01), np.log10(0.5), 20)
default_ppd = 60
default_radius_test = 0.8
default_rho_test = 1.2
multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)

foley_result_json = r'../contrast_masking_data_gabor_on_noise.json'
with open(foley_result_json, 'r') as fp:
    foley_result_data = json.load(fp)
foley_result_x_mask_contrast_list = np.array(foley_result_data['mask_contrast_list'])
foley_result_y_test_contrast_list = np.array(foley_result_data['test_contrast_list'])
valid_gt_indices = [index for index, value in enumerate(foley_result_x_mask_contrast_list) if
                    value > contrast_mask_list.min() and value < 0.25]
gt_x_mask_C = foley_result_x_mask_contrast_list[valid_gt_indices]
gt_y_test_C = foley_result_y_test_contrast_list[valid_gt_indices]
middle_X_mask_C = gt_x_mask_C[round(len(gt_x_mask_C) / 2)]
middle_Y_test_C = gt_y_test_C[round(len(gt_y_test_C) / 2)]
T_vid, R_vid = generate_contrast_masking_gabor_on_noise(W=default_W, H=default_H,
                                                                sigma=default_radius_test,
                                                                rho=default_rho_test,
                                                                Mask_upper_frequency=default_Mask_upper_frequency,
                                                                L_b=default_L_b,
                                                                contrast_mask=middle_X_mask_C,
                                                                contrast_test=middle_Y_test_C,
                                                                ppd=default_ppd)
T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32)[None, None, ...]
R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32)[None, None, ...]
T_vid_ct_aim = T_vid_ct.expand(-1, 3, -1, -1).cuda()
R_vid_ct_aim = R_vid_ct.expand(-1, 3, -1, -1).cuda()

aim_cos_similarity_list = []
Spearman_matrix_cos_list = []
for backbone_name in tqdm(all_backbone_list):
    backbone_model = torch.hub.load('facebookresearch/dinov2', backbone_name)
    backbone_model.eval()
    backbone_model.cuda()
    test_feature = backbone_model(T_vid_ct_aim)
    reference_feature = backbone_model(R_vid_ct_aim)
    cos_similarity = float(F.cosine_similarity(test_feature, reference_feature).cpu())
    aim_cos_similarity_list.append(cos_similarity)

    Spearman_matrix_cos = np.zeros([len(gt_x_mask_C), len(multiplier_list)])
    for contrast_mask_index in tqdm(range(len(gt_x_mask_C))):
        contrast_mask_value = gt_x_mask_C[contrast_mask_index]
        contrast_test_value = gt_y_test_C[contrast_mask_index]
        for multiplier_index, multiplier_value in enumerate(multiplier_list):
            C_test = contrast_test_value * multiplier_value
            T_vid, R_vid = generate_contrast_masking_gabor_on_noise(W=default_W, H=default_H,
                                                                    sigma=default_radius_test,
                                                                    rho=default_rho_test,
                                                                    Mask_upper_frequency=default_Mask_upper_frequency,
                                                                    L_b=default_L_b,
                                                                    contrast_mask=contrast_mask_value,
                                                                    contrast_test=C_test,
                                                                    ppd=default_ppd)
            T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
            R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
            T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32)[None, None, ...]
            R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32)[None, None, ...]
            T_vid_ct = T_vid_ct.expand(-1, 3, -1, -1).cuda()
            R_vid_ct = R_vid_ct.expand(-1, 3, -1, -1).cuda()
            test_feature = backbone_model(T_vid_ct)
            reference_feature = backbone_model(R_vid_ct)
            cos_similarity = float(F.cosine_similarity(test_feature, reference_feature).cpu())
            Spearman_matrix_cos[contrast_mask_index, multiplier_index] = cos_similarity
    Spearman_matrix_cos_list.append(Spearman_matrix_cos.tolist())
json_file_name = r'new_data_logs\dinov2\different_c/dinov2_test_on_contrast_masking_different_c_contour_plot_ppd_60_final.json'
save_json_file_name = r'new_data_logs\dinov2\different_c/dinov2_test_on_contrast_masking_different_c_contour_plot_ppd_60_final_aim.json'
with open(json_file_name, 'r') as fp:
    json_data = json.load(fp)
json_data['aim_cos_similarity_list'] = aim_cos_similarity_list
json_data['valid_gt_indices'] = valid_gt_indices
json_data['gt_x_mask_C'] = gt_x_mask_C.tolist()
json_data['gt_y_test_C'] = gt_y_test_C.tolist()
json_data['multiplier_list'] = multiplier_list.tolist()
json_data['Spearman_matrix_cos_list'] = Spearman_matrix_cos_list

with open(save_json_file_name, 'w') as fp:
    json.dump(json_data, fp)
