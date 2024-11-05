import sys

sys.path.append('E:\Py_codes\LVM_Comparision')
import numpy as np
import torch
from Band_limit_noise_generator.generate_plot_band_lim_noise import generate_band_lim_noise, generate_band_lim_noise_fix_random_seed
import torch.nn.functional as F
import pandas as pd
import json
from tqdm import tqdm
import os
from display_encoding import display_encode
from PIL import Image
from SAM_repo.SAM import SAMFeatureExtractor
display_encode_tool = display_encode(400)

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt

# Only test cpd right now
# input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/sam/different_rho'
os.makedirs(save_root_path, exist_ok=True)
all_sam_model_list = ['sam_vit_b_01ec64', 'sam_vit_l_0b3195', 'sam_vit_h_4b8939']
sam_vit_list = ['vit_b', 'vit_l', 'vit_h']

default_W = 224
default_H = 224
rho_list = np.logspace(np.log10(0.5), np.log10(32), 20)
default_L_b = 100
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
ppd_list = [30, 35, 40, 45, 50, 55, 60]

for ppd_index in tqdm(range(len(ppd_list))):
    ppd_number = ppd_list[ppd_index]
    json_plot_data = {}
    json_plot_data['sam_model_name'] = []
    json_plot_data['rho_matrix'] = []
    json_plot_data['contrast_matrix'] = []
    json_plot_data['final_feature_L1_similarity_matrix'] = []
    json_plot_data['final_feature_L2_similarity_matrix'] = []
    json_plot_data['final_feature_cos_similarity_matrix'] = []
    json_plot_data['final_feature_L1_similarity_matrix_fix_random_seed'] = []
    json_plot_data['final_feature_L2_similarity_matrix_fix_random_seed'] = []
    json_plot_data['final_feature_cos_similarity_matrix_fix_random_seed'] = []

    for sam_model_index in tqdm(range(len(all_sam_model_list))):
        sam_model_name = all_sam_model_list[sam_model_index]
        plot_rho_matrix = np.zeros([len(rho_list), len(contrast_list)])
        plot_contrast_matrix = np.zeros([len(rho_list), len(contrast_list)])
        plot_final_feature_L1_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
        plot_final_feature_L2_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
        plot_final_feature_cos_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
        plot_final_feature_L1_similarity_matrix_f = np.zeros([len(rho_list), len(contrast_list)])
        plot_final_feature_L2_similarity_matrix_f = np.zeros([len(rho_list), len(contrast_list)])
        plot_final_feature_cos_similarity_matrix_f = np.zeros([len(rho_list), len(contrast_list)])

        feature_extractor = SAMFeatureExtractor(
            model_type=sam_vit_list[sam_model_index],
            checkpoint_path=rf"E:\Py_codes\LVM_Comparision\SAM_repo/{sam_model_name}.pth",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        for rho_index in range(len(rho_list)):
            rho_value = rho_list[rho_index]
            for contrast_index in range(len(contrast_list)):
                contrast_value = contrast_list[contrast_index]
                plot_rho_matrix[rho_index, contrast_index] = rho_value
                plot_contrast_matrix[rho_index, contrast_index] = contrast_value
                T_vid, R_vid = generate_band_lim_noise(W=default_W, H=default_H, freq_band=rho_value,
                                                       L_b=default_L_b, contrast=contrast_value, ppd=ppd_number)
                T_vid_f, R_vid_f = generate_band_lim_noise_fix_random_seed(W=default_W, H=default_H,
                                                                           freq_band=rho_value,
                                                                           L_b=default_L_b, contrast=contrast_value,
                                                                           ppd=ppd_number)
                T_vid = np.stack([T_vid] * 3, axis=-1)
                R_vid = np.stack([R_vid] * 3, axis=-1)
                T_vid_f = np.stack([T_vid_f] * 3, axis=-1)
                R_vid_f = np.stack([R_vid_f] * 3, axis=-1)
                with torch.no_grad():
                    T_vid_c = (display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8)
                    R_vid_c = (display_encode_tool.L2C_sRGB(R_vid) * 255).astype(np.uint8)
                    T_features = feature_extractor.extract_features_from_numpy(T_vid_c)
                    R_features = feature_extractor.extract_features_from_numpy(R_vid_c)
                    T_vid_c_f = (display_encode_tool.L2C_sRGB(T_vid_f) * 255).astype(np.uint8)
                    R_vid_c_f = (display_encode_tool.L2C_sRGB(R_vid_f) * 255).astype(np.uint8)
                    T_features_f = feature_extractor.extract_features_from_numpy(T_vid_c_f)
                    R_features_f = feature_extractor.extract_features_from_numpy(R_vid_c_f)

                    L1_similarity = float(torch.norm(T_features - R_features, p=1).cpu())
                    L2_similarity = float(torch.norm(T_features - R_features, p=2).cpu())
                    cos_similarity = float(F.cosine_similarity(T_features.view(1, -1), R_features.view(1, -1)).cpu())
                    L1_similarity_f = float(torch.norm(T_features_f - R_features_f, p=1).cpu())
                    L2_similarity_f = float(torch.norm(T_features_f - R_features_f, p=2).cpu())
                    cos_similarity_f = float(
                        F.cosine_similarity(T_features_f.view(1, -1), R_features_f.view(1, -1)).cpu())

                plot_final_feature_L1_similarity_matrix[rho_index, contrast_index] = L1_similarity
                plot_final_feature_L2_similarity_matrix[rho_index, contrast_index] = L2_similarity
                plot_final_feature_cos_similarity_matrix[rho_index, contrast_index] = cos_similarity
                plot_final_feature_L1_similarity_matrix_f[rho_index, contrast_index] = L1_similarity_f
                plot_final_feature_L2_similarity_matrix_f[rho_index, contrast_index] = L2_similarity_f
                plot_final_feature_cos_similarity_matrix_f[rho_index, contrast_index] = cos_similarity_f
        json_plot_data['sam_model_name'].append(sam_model_name)
        json_plot_data['rho_matrix'].append(plot_rho_matrix.tolist())
        json_plot_data['contrast_matrix'].append(plot_contrast_matrix.tolist())
        json_plot_data['final_feature_L1_similarity_matrix'].append(plot_final_feature_L1_similarity_matrix.tolist())
        json_plot_data['final_feature_L2_similarity_matrix'].append(plot_final_feature_L2_similarity_matrix.tolist())
        json_plot_data['final_feature_cos_similarity_matrix'].append(plot_final_feature_cos_similarity_matrix.tolist())
        json_plot_data['final_feature_L1_similarity_matrix_fix_random_seed'].append(
            plot_final_feature_L1_similarity_matrix_f.tolist())
        json_plot_data['final_feature_L2_similarity_matrix_fix_random_seed'].append(
            plot_final_feature_L2_similarity_matrix_f.tolist())
        json_plot_data['final_feature_cos_similarity_matrix_fix_random_seed'].append(
            plot_final_feature_cos_similarity_matrix_f.tolist())

    with open(os.path.join(save_root_path,
                           f'sam_test_on_band_lim_noise_different_rho_band_contour_plot_ppd_{ppd_number}_final.json'),
              'w') as fp:
        json.dump(json_plot_data, fp)
