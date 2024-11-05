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
from sam2.sam2.build_sam import build_sam2
from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt
from display_encoding import display_encode

display_encode_tool = display_encode(400)
save_root_path = 'new_data_logs/sam2/different_area'
os.makedirs(save_root_path, exist_ok=True)
all_sam_model_list = ['sam2.1_hiera_tiny', 'sam2.1_hiera_small', 'sam2.1_hiera_base_plus', 'sam2.1_hiera_large']
sam_config_list = ['sam2.1_hiera_t', 'sam2.1_hiera_s', 'sam2.1_hiera_b+', 'sam2.1_hiera_l']

default_W = 224
default_H = 224
R_min = 0.1
R_max = 1
Area_list = np.logspace(np.log10(math.pi * R_min ** 2), np.log10(math.pi * R_max ** 2), 20)
R_list = (Area_list / math.pi) ** 0.5
default_rho = 8
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
default_O = 0
default_contrast = 1
default_L_b = 100
ppd_list = [60]

for ppd_index in tqdm(range(len(ppd_list))):
    ppd_number = ppd_list[ppd_index]
    csv_data = {}
    csv_data['sam_model_name'] = []
    csv_data['Radius'] = []
    csv_data['Area'] = []
    csv_data['contrast'] = []
    csv_data['final_feature_L1_similarity'] = []
    csv_data['final_feature_L2_similarity'] = []
    csv_data['final_feature_cos_similarity'] = []

    json_plot_data = {}
    json_plot_data['sam_model_name'] = []
    json_plot_data['radius_matrix'] = []
    json_plot_data['area_matrix'] = []
    json_plot_data['contrast_matrix'] = []
    json_plot_data['final_feature_L1_similarity_matrix'] = []
    json_plot_data['final_feature_L2_similarity_matrix'] = []
    json_plot_data['final_feature_cos_similarity_matrix'] = []

    for sam_model_index in tqdm(range(len(all_sam_model_list))):
        sam_model_name = all_sam_model_list[sam_model_index]
        sam_config_name = sam_config_list[sam_model_index]
        plot_radius_matrix = np.zeros([len(R_list), len(contrast_list)])
        plot_area_matrix = np.zeros([len(R_list), len(contrast_list)])
        plot_contrast_matrix = np.zeros([len(R_list), len(contrast_list)])
        plot_final_feature_L1_similarity_matrix = np.zeros([len(R_list), len(contrast_list)])
        plot_final_feature_L2_similarity_matrix = np.zeros([len(R_list), len(contrast_list)])
        plot_final_feature_cos_similarity_matrix = np.zeros([len(R_list), len(contrast_list)])

        checkpoint = fr"E:\Py_codes\LVM_Comparision\SAM_repo\{sam_model_name}.pt"
        model_cfg = fr"E:\Py_codes\LVM_Comparision\sam2\sam2\configs/sam2.1/{sam_config_name}.yaml"
        predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

        for R_index in range(len(R_list)):
            R_value = R_list[R_index]
            A_value = Area_list[R_index]
            for contrast_index in range(len(contrast_list)):
                contrast_value = contrast_list[contrast_index]
                csv_data['sam_model_name'].append(sam_model_name)
                csv_data['Radius'].append(R_value)
                csv_data['Area'].append(A_value)
                csv_data['contrast'].append(contrast_value)
                plot_radius_matrix[R_index, contrast_index] = R_value
                plot_area_matrix[R_index, contrast_index] = A_value
                plot_contrast_matrix[R_index, contrast_index] = contrast_value
                T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=R_value, rho=default_rho, O=default_O,
                                                    L_b=default_L_b, contrast=contrast_value, ppd=ppd_number,
                                                    color_direction='ach')
                with torch.no_grad():
                    T_vid_c = (display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8)
                    R_vid_c = (display_encode_tool.L2C_sRGB(R_vid) * 255).astype(np.uint8)
                    predictor.set_image(T_vid_c)
                    T_features = predictor.get_image_embedding()
                    predictor.set_image(R_vid_c)
                    R_features = predictor.get_image_embedding()
                    L1_similarity = float(torch.norm(T_features - R_features, p=1).cpu())
                    L2_similarity = float(torch.norm(T_features - R_features, p=2).cpu())
                    cos_similarity = float(
                        F.cosine_similarity(T_features.reshape(1, -1), R_features.reshape(1, -1)).cpu())
                csv_data['final_feature_L1_similarity'].append(L1_similarity)
                plot_final_feature_L1_similarity_matrix[R_index, contrast_index] = L1_similarity

                csv_data['final_feature_L2_similarity'].append(L2_similarity)
                plot_final_feature_L2_similarity_matrix[R_index, contrast_index] = L2_similarity

                csv_data['final_feature_cos_similarity'].append(cos_similarity)
                plot_final_feature_cos_similarity_matrix[R_index, contrast_index] = cos_similarity

                df = pd.DataFrame(csv_data)
                df.to_csv(os.path.join(save_root_path,
                                       f'sam2_test_on_gabors_different_area_contour_plot_ppd_{ppd_number}_temporary.csv'),
                          index=False)
        json_plot_data['sam_model_name'].append(sam_model_name)
        json_plot_data['radius_matrix'].append(plot_radius_matrix.tolist())
        json_plot_data['area_matrix'].append(plot_area_matrix.tolist())
        json_plot_data['contrast_matrix'].append(plot_contrast_matrix.tolist())
        json_plot_data['final_feature_L1_similarity_matrix'].append(plot_final_feature_L1_similarity_matrix.tolist())
        json_plot_data['final_feature_L2_similarity_matrix'].append(plot_final_feature_L2_similarity_matrix.tolist())
        json_plot_data['final_feature_cos_similarity_matrix'].append(plot_final_feature_cos_similarity_matrix.tolist())

    df = pd.DataFrame(csv_data)
    df.to_csv(
        os.path.join(save_root_path, f'sam2_test_on_gabors_different_area_contour_plot_ppd_{ppd_number}_final.csv'),
        index=False)
    with open(os.path.join(save_root_path, f'sam2_test_on_gabors_different_area_contour_plot_ppd_{ppd_number}_final.json'),
            'w') as fp:
        json.dump(json_plot_data, fp)


