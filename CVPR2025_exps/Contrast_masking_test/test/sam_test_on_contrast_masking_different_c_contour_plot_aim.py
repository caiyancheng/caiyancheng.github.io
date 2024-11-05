import sys
sys.path.append(r'E:\Py_codes\LVM_Comparision')
import numpy as np
import torch
from Contrast_masking_generator.generate_plot_contrast_masking import generate_contrast_masking
import torch.nn.functional as F
import pandas as pd
import json
from tqdm import tqdm
import os
import math
from PIL import Image
from SAM_repo.SAM import SAMFeatureExtractor

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt
from display_encoding import display_encode

display_encode_tool = display_encode(400)
save_root_path = 'new_data_logs/sam/different_c'
os.makedirs(save_root_path, exist_ok=True)
all_sam_model_list = ['sam_vit_b_01ec64', 'sam_vit_l_0b3195', 'sam_vit_h_4b8939']
sam_vit_list = ['vit_b', 'vit_l', 'vit_h']

default_W = 224
default_H = 224
default_rho = 2
default_O = 0
default_L_b = 32
contrast_mask_list = np.logspace(np.log10(0.005), np.log10(0.5), 20)
contrast_test_list = np.logspace(np.log10(0.01), np.log10(0.5), 20)
default_gabor_radius = 0.5
ppd_list = [60]
multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)

for ppd_index in tqdm(range(len(ppd_list))):
    ppd_number = ppd_list[ppd_index]

    foley_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/foley_contrast_masking_data_gabor.json'
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
    T_vid, R_vid = generate_contrast_masking(W=default_W, H=default_H, rho=default_rho, O=default_O,
                                             L_b=default_L_b, contrast_mask=middle_X_mask_C,
                                             contrast_test=middle_Y_test_C, ppd=ppd_number,
                                             gabor_radius=default_gabor_radius)
    T_vid = np.stack([T_vid] * 3, axis=-1)
    R_vid = np.stack([R_vid] * 3, axis=-1)
    T_vid_c_aim = (display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8)
    R_vid_c_aim = (display_encode_tool.L2C_sRGB(R_vid) * 255).astype(np.uint8)

    aim_cos_similarity_list = []
    Spearman_matrix_cos_list = []
    for sam_model_index in tqdm(range(len(all_sam_model_list))):
        sam_model_name = all_sam_model_list[sam_model_index]
        feature_extractor = SAMFeatureExtractor(
            model_type=sam_vit_list[sam_model_index],
            checkpoint_path=rf"E:\Py_codes\LVM_Comparision\SAM_repo/{sam_model_name}.pth",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        T_features = feature_extractor.extract_features_from_numpy(T_vid_c_aim)
        R_features = feature_extractor.extract_features_from_numpy(R_vid_c_aim)
        cos_similarity = float(F.cosine_similarity(T_features.view(1, -1), R_features.view(1, -1)).cpu())
        aim_cos_similarity_list.append(cos_similarity)

        Spearman_matrix_cos = np.zeros([len(gt_x_mask_C), len(multiplier_list)])
        for contrast_mask_index in tqdm(range(len(gt_x_mask_C))):
            contrast_mask_value = gt_x_mask_C[contrast_mask_index]
            contrast_test_value = gt_y_test_C[contrast_mask_index]
            for multiplier_index, multiplier_value in enumerate(multiplier_list):
                C_test = contrast_test_value * multiplier_value
                T_vid, R_vid = generate_contrast_masking(W=default_W, H=default_H, rho=default_rho, O=default_O,
                                                         L_b=default_L_b, contrast_mask=contrast_mask_value,
                                                         contrast_test=C_test, ppd=ppd_number,
                                                         gabor_radius=default_gabor_radius)
                T_vid = np.stack([T_vid] * 3, axis=-1)
                R_vid = np.stack([R_vid] * 3, axis=-1)
                with torch.no_grad():
                    T_vid_c = (display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8)
                    R_vid_c = (display_encode_tool.L2C_sRGB(R_vid) * 255).astype(np.uint8)
                    T_features = feature_extractor.extract_features_from_numpy(T_vid_c)
                    R_features = feature_extractor.extract_features_from_numpy(R_vid_c)
                    cos_similarity = float(F.cosine_similarity(T_features.view(1, -1), R_features.view(1, -1)).cpu())
                    Spearman_matrix_cos[contrast_mask_index, multiplier_index] = cos_similarity
        Spearman_matrix_cos_list.append(Spearman_matrix_cos.tolist())
    json_file_name = rf'new_data_logs\sam\different_c/sam_test_on_contrast_masking_different_c_contour_plot_ppd_{ppd_number}_final.json'
    save_json_file_name = rf'new_data_logs\sam\different_c/sam_test_on_contrast_masking_different_c_contour_plot_ppd_{ppd_number}_final_aim.json'
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

