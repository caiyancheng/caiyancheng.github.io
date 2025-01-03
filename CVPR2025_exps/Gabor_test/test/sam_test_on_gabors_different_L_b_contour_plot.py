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
from PIL import Image
from SAM_repo.SAM import SAMFeatureExtractor

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt
from display_encoding import display_encode

display_encode_tool = display_encode(400)

# Only test cpd right now
# Dino input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/sam/different_luminance'
os.makedirs(save_root_path, exist_ok=True)
all_sam_model_list = ['sam_vit_b_01ec64', 'sam_vit_l_0b3195', 'sam_vit_h_4b8939']
sam_vit_list = ['vit_b', 'vit_l', 'vit_h']

default_W = 224
default_H = 224
default_R = 1
default_rho = 2
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
default_O = 0
default_contrast = 1
L_b_list = np.logspace(np.log10(0.1), np.log10(200), 20)
ppd_list = [60]

for ppd_index in tqdm(range(len(ppd_list))):
    ppd_number = ppd_list[ppd_index]
    csv_data = {}
    csv_data['sam_model_name'] = []
    csv_data['L_b'] = []
    csv_data['contrast'] = []
    csv_data['final_feature_L1_similarity'] = []
    csv_data['final_feature_L2_similarity'] = []
    csv_data['final_feature_cos_similarity'] = []

    json_plot_data = {}
    json_plot_data['sam_model_name'] = []
    json_plot_data['L_b_matrix'] = []
    json_plot_data['contrast_matrix'] = []
    json_plot_data['final_feature_L1_similarity_matrix'] = []
    json_plot_data['final_feature_L2_similarity_matrix'] = []
    json_plot_data['final_feature_cos_similarity_matrix'] = []

    for sam_model_index in tqdm(range(len(all_sam_model_list))):
        sam_model_name = all_sam_model_list[sam_model_index]
        plot_L_b_matrix = np.zeros([len(L_b_list), len(contrast_list)])
        plot_contrast_matrix = np.zeros([len(L_b_list), len(contrast_list)])
        plot_final_feature_L1_similarity_matrix = np.zeros([len(L_b_list), len(contrast_list)])
        plot_final_feature_L2_similarity_matrix = np.zeros([len(L_b_list), len(contrast_list)])
        plot_final_feature_cos_similarity_matrix = np.zeros([len(L_b_list), len(contrast_list)])

        feature_extractor = SAMFeatureExtractor(
            model_type=sam_vit_list[sam_model_index],
            checkpoint_path=rf"E:\Py_codes\LVM_Comparision\SAM_repo/{sam_model_name}.pth",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        for L_b_index in range(len(L_b_list)):
            L_b_value = L_b_list[L_b_index]
            for contrast_index in range(len(contrast_list)):
                contrast_value = contrast_list[contrast_index]
                csv_data['sam_model_name'].append(sam_model_name)
                csv_data['L_b'].append(L_b_value)
                csv_data['contrast'].append(contrast_value)
                plot_L_b_matrix[L_b_index, contrast_index] = L_b_value
                plot_contrast_matrix[L_b_index, contrast_index] = contrast_value
                T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=default_R, rho=default_rho, O=default_O,
                                                    L_b=L_b_value, contrast=contrast_value, ppd=ppd_number,
                                                    color_direction='ach')
                with torch.no_grad():
                    T_vid_c = (display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8)
                    R_vid_c = (display_encode_tool.L2C_sRGB(R_vid) * 255).astype(np.uint8)
                    T_features = feature_extractor.extract_features_from_numpy(T_vid_c)
                    R_features = feature_extractor.extract_features_from_numpy(R_vid_c)
                    L1_similarity = float(torch.norm(T_features - R_features, p=1).cpu())
                    L2_similarity = float(torch.norm(T_features - R_features, p=2).cpu())
                    cos_similarity = float(F.cosine_similarity(T_features.view(1, -1), R_features.view(1, -1)).cpu())

                csv_data['final_feature_L1_similarity'].append(L1_similarity)
                plot_final_feature_L1_similarity_matrix[L_b_index, contrast_index] = L1_similarity
                csv_data['final_feature_L2_similarity'].append(L2_similarity)
                plot_final_feature_L2_similarity_matrix[L_b_index, contrast_index] = L2_similarity
                csv_data['final_feature_cos_similarity'].append(cos_similarity)
                plot_final_feature_cos_similarity_matrix[L_b_index, contrast_index] = cos_similarity

                df = pd.DataFrame(csv_data)
                df.to_csv(os.path.join(save_root_path,
                                       f'sam_test_on_gabors_different_luminance_contour_plot_ppd_{ppd_number}_temporary.csv'),
                          index=False)
        json_plot_data['sam_model_name'].append(sam_model_name)
        json_plot_data['L_b_matrix'].append(plot_L_b_matrix.tolist())
        json_plot_data['contrast_matrix'].append(plot_contrast_matrix.tolist())
        json_plot_data['final_feature_L1_similarity_matrix'].append(plot_final_feature_L1_similarity_matrix.tolist())
        json_plot_data['final_feature_L2_similarity_matrix'].append(plot_final_feature_L2_similarity_matrix.tolist())
        json_plot_data['final_feature_cos_similarity_matrix'].append(plot_final_feature_cos_similarity_matrix.tolist())

    df = pd.DataFrame(csv_data)
    df.to_csv(
        os.path.join(save_root_path,
                     f'sam_test_on_gabors_different_luminance_contour_plot_ppd_{ppd_number}_final.csv'),
        index=False)
    with open(
            os.path.join(save_root_path,
                         f'sam_test_on_gabors_different_luminance_contour_plot_ppd_{ppd_number}_final.json'),
            'w') as fp:
        json.dump(json_plot_data, fp)
