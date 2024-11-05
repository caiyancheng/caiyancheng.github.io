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
import math
from PIL import Image
from sam2.sam2.build_sam import build_sam2
from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt
from display_encoding import display_encode

display_encode_tool = display_encode(400)
save_root_path = 'new_data_logs/sam2/different_c'
os.makedirs(save_root_path, exist_ok=True)
all_sam_model_list = ['sam2.1_hiera_tiny', 'sam2.1_hiera_small', 'sam2.1_hiera_base_plus', 'sam2.1_hiera_large']
sam_config_list = ['sam2.1_hiera_t', 'sam2.1_hiera_s', 'sam2.1_hiera_b+', 'sam2.1_hiera_l']

default_W = 224
default_H = 224
default_Mask_upper_frequency = 12
default_L_b = 37
contrast_mask_list = np.logspace(np.log10(0.005), np.log10(0.5), 20)
contrast_test_list = np.logspace(np.log10(0.01), np.log10(0.5), 20)
default_radius_test = 0.8
default_rho_test = 1.2
ppd_list = [60]

for ppd_index in tqdm(range(len(ppd_list))):
    ppd_number = ppd_list[ppd_index]
    csv_data = {}
    csv_data['sam_model_name'] = []
    csv_data['contrast_mask'] = []
    csv_data['contrast_test'] = []
    csv_data['final_feature_L1_similarity'] = []
    csv_data['final_feature_L2_similarity'] = []
    csv_data['final_feature_cos_similarity'] = []

    json_plot_data = {}
    json_plot_data['sam_model_name'] = []
    json_plot_data['contrast_mask_matrix'] = []
    json_plot_data['contrast_test_matrix'] = []
    json_plot_data['final_feature_L1_similarity_matrix'] = []
    json_plot_data['final_feature_L2_similarity_matrix'] = []
    json_plot_data['final_feature_cos_similarity_matrix'] = []

    for sam_model_index in tqdm(range(len(all_sam_model_list))):
        sam_model_name = all_sam_model_list[sam_model_index]
        sam_config_name = sam_config_list[sam_model_index]
        plot_contrast_mask_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
        plot_contrast_test_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
        plot_final_feature_L1_similarity_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
        plot_final_feature_L2_similarity_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
        plot_final_feature_cos_similarity_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])

        checkpoint = fr"E:\Py_codes\LVM_Comparision\SAM_repo\{sam_model_name}.pt"
        model_cfg = fr"E:\Py_codes\LVM_Comparision\sam2\sam2\configs/sam2.1/{sam_config_name}.yaml"
        predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

        for contrast_mask_index in range(len(contrast_mask_list)):
            contrast_mask_value = contrast_mask_list[contrast_mask_index]
            for contrast_test_index in range(len(contrast_test_list)):
                contrast_test_value = contrast_test_list[contrast_test_index]
                csv_data['sam_model_name'].append(sam_model_name)
                csv_data['contrast_mask'].append(contrast_mask_value)
                csv_data['contrast_test'].append(contrast_test_value)
                plot_contrast_mask_matrix[contrast_mask_index, contrast_test_index] = contrast_mask_value
                plot_contrast_test_matrix[contrast_mask_index, contrast_test_index] = contrast_test_value
                T_vid, R_vid = generate_contrast_masking_gabor_on_noise(W=default_W, H=default_H,
                                                                        sigma=default_radius_test,
                                                                        rho=default_rho_test,
                                                                        Mask_upper_frequency=default_Mask_upper_frequency,
                                                                        L_b=default_L_b,
                                                                        contrast_mask=contrast_mask_value,
                                                                        contrast_test=contrast_test_value,
                                                                        ppd=ppd_number)
                T_vid = np.stack([T_vid] * 3, axis=-1)
                R_vid = np.stack([R_vid] * 3, axis=-1)
                T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
                R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
                with torch.no_grad():
                    T_vid_c = (display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8)
                    R_vid_c = (display_encode_tool.L2C_sRGB(R_vid) * 255).astype(np.uint8)
                    predictor.set_image(T_vid_c)
                    T_features = predictor.get_image_embedding()
                    predictor.set_image(R_vid_c)
                    R_features = predictor.get_image_embedding()
                    L1_similarity = float(torch.norm(T_features - R_features, p=1).cpu())
                    L2_similarity = float(torch.norm(T_features - R_features, p=2).cpu())
                    cos_similarity = float(F.cosine_similarity(T_features.reshape(1, -1), R_features.reshape(1, -1)).cpu())

                csv_data['final_feature_L1_similarity'].append(L1_similarity)
                plot_final_feature_L1_similarity_matrix[contrast_mask_index, contrast_test_index] = L1_similarity

                csv_data['final_feature_L2_similarity'].append(L2_similarity)
                plot_final_feature_L2_similarity_matrix[contrast_mask_index, contrast_test_index] = L2_similarity

                csv_data['final_feature_cos_similarity'].append(cos_similarity)
                plot_final_feature_cos_similarity_matrix[contrast_mask_index, contrast_test_index] = cos_similarity

                df = pd.DataFrame(csv_data)
                df.to_csv(os.path.join(save_root_path,
                                       f'sam2_test_on_contrast_masking_different_c_contour_plot_ppd_{ppd_number}_temporary.csv'),
                          index=False)
        json_plot_data['sam_model_name'].append(sam_model_name)
        json_plot_data['contrast_test_matrix'].append(plot_contrast_test_matrix.tolist())
        json_plot_data['contrast_mask_matrix'].append(plot_contrast_mask_matrix.tolist())
        json_plot_data['final_feature_L1_similarity_matrix'].append(plot_final_feature_L1_similarity_matrix.tolist())
        json_plot_data['final_feature_L2_similarity_matrix'].append(plot_final_feature_L2_similarity_matrix.tolist())
        json_plot_data['final_feature_cos_similarity_matrix'].append(plot_final_feature_cos_similarity_matrix.tolist())

    df = pd.DataFrame(csv_data)
    df.to_csv(
        os.path.join(save_root_path,
                     f'sam2_test_on_contrast_masking_different_c_contour_plot_ppd_{ppd_number}_final.csv'),
        index=False)
    with open(os.path.join(save_root_path,
                           f'sam2_test_on_contrast_masking_different_c_contour_plot_ppd_{ppd_number}_final.json'),
              'w') as fp:
        json.dump(json_plot_data, fp)
