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

# Only test cpd right now
# Dino input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/sam2/different_rho'
os.makedirs(save_root_path, exist_ok=True)
all_sam_model_list = ['sam2.1_hiera_tiny', 'sam2.1_hiera_small', 'sam2.1_hiera_base_plus', 'sam2.1_hiera_large']
sam_config_list = ['sam2.1_hiera_t', 'sam2.1_hiera_s', 'sam2.1_hiera_b+', 'sam2.1_hiera_l']

default_W = 224
default_H = 224
default_R = 1
rho_list = np.logspace(np.log10(0.5), np.log10(32), 20)
rho_list[0] = 0.5
rho_list[-1] = 32
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
default_O = 0
default_contrast = 1
default_L_b = 100
ppd_list = [60]
multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)

castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_rho_sensitivity_data.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_rho_list = castleCSF_result_data['rho_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']
middle_rho = 10 ** ((np.log10(rho_list[0]) + np.log10(rho_list[-1])) / 2)
middle_rho_S = np.interp(middle_rho, castleCSF_result_rho_list, castleCSF_result_sensitivity_list)

for ppd_index in tqdm(range(len(ppd_list))):
    ppd_number = ppd_list[ppd_index]
    T_vid_aim, R_vid_aim = generate_gabor_patch(W=default_W, H=default_H, R=default_R, rho=middle_rho,
                                        O=default_O, L_b=default_L_b, contrast=1 / middle_rho_S,
                                        ppd=ppd_number, color_direction='ach')
    T_vid_c_aim = (display_encode_tool.L2C_sRGB(T_vid_aim) * 255).astype(np.uint8)
    R_vid_c_aim = (display_encode_tool.L2C_sRGB(R_vid_aim) * 255).astype(np.uint8)
    aim_cos_similarity_list = []
    Spearman_matrix_cos_list = []
    for sam_model_index in tqdm(range(len(all_sam_model_list))):
        sam_model_name = all_sam_model_list[sam_model_index]
        sam_config_name = sam_config_list[sam_model_index]

        checkpoint = fr"E:\Py_codes\LVM_Comparision\SAM_repo\{sam_model_name}.pt"
        model_cfg = fr"E:\Py_codes\LVM_Comparision\sam2\sam2\configs/sam2.1/{sam_config_name}.yaml"
        predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

        predictor.set_image(T_vid_c_aim)
        T_features_aim = predictor.get_image_embedding()
        predictor.set_image(R_vid_c_aim)
        R_features_aim = predictor.get_image_embedding()
        cos_similarity = float(
            F.cosine_similarity(T_features_aim.reshape(1, -1), R_features_aim.reshape(1, -1)).cpu())
        aim_cos_similarity_list.append(cos_similarity)

        Spearman_matrix_cos = np.zeros([len(rho_list), len(multiplier_list)])
        for rho_index, rho_value in enumerate(rho_list):
            S_gt = np.interp(rho_value, castleCSF_result_rho_list, castleCSF_result_sensitivity_list)
            for multiplier_index, multiplier_value in enumerate(multiplier_list):
                S_test = multiplier_value * S_gt
                T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=default_R, rho=rho_value,
                                                    O=default_O,
                                                    L_b=default_L_b, contrast=1 / S_test, ppd=ppd_number,
                                                    color_direction='ach')
                T_vid_c = (display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8)
                R_vid_c = (display_encode_tool.L2C_sRGB(R_vid) * 255).astype(np.uint8)
                predictor.set_image(T_vid_c)
                T_features = predictor.get_image_embedding()
                predictor.set_image(R_vid_c)
                R_features = predictor.get_image_embedding()
                cos_similarity = float(
                    F.cosine_similarity(T_features.reshape(1, -1), R_features.reshape(1, -1)).cpu())
                Spearman_matrix_cos[rho_index, multiplier_index] = cos_similarity
        Spearman_matrix_cos_list.append(Spearman_matrix_cos.tolist())
    json_file_name = rf'new_data_logs\sam2\different_rho/sam2_test_on_gabors_different_rho_contour_plot_ppd_{ppd_number}_final.json'
    save_json_file_name = rf'new_data_logs\sam2\different_rho/sam2_test_on_gabors_different_rho_contour_plot_ppd_{ppd_number}_final_aim.json'
    with open(json_file_name, 'r') as fp:
        json_data = json.load(fp)
    json_data['aim_cos_similarity_list'] = aim_cos_similarity_list
    json_data['rho_list'] = rho_list.tolist()
    json_data['multiplier_list'] = multiplier_list.tolist()
    json_data['Spearman_matrix_cos_list'] = Spearman_matrix_cos_list

    with open(save_json_file_name, 'w') as fp:
        json.dump(json_data, fp)
