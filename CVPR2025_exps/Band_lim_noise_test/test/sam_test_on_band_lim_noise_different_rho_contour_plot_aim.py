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
rho_list[0] = 0.5
rho_list[-1] = 32
default_L_b = 100
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
ppd_list = [60]
multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)

for ppd_index in tqdm(range(len(ppd_list))):
    ppd_number = ppd_list[ppd_index]

    castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_rho_sensitivity_data.json'
    with open(castleCSF_result_json, 'r') as fp:
        castleCSF_result_data = json.load(fp)
    castleCSF_result_rho_list = castleCSF_result_data['rho_list']
    castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']
    middle_rho = 10 ** ((np.log10(rho_list[0]) + np.log10(rho_list[-1])) / 2)
    middle_rho_S = np.interp(middle_rho, castleCSF_result_rho_list, castleCSF_result_sensitivity_list)
    T_vid, R_vid = generate_band_lim_noise(W=default_W, H=default_H, freq_band=middle_rho,
                                           L_b=default_L_b, contrast=1 / middle_rho_S, ppd=ppd_number)
    T_vid_f, R_vid_f = generate_band_lim_noise_fix_random_seed(W=default_W, H=default_H, freq_band=middle_rho,
                                                               L_b=default_L_b, contrast=1 / middle_rho_S,
                                                               ppd=ppd_number)
    T_vid = np.stack([T_vid] * 3, axis=-1)
    R_vid = np.stack([R_vid] * 3, axis=-1)
    T_vid_f = np.stack([T_vid_f] * 3, axis=-1)
    R_vid_f = np.stack([R_vid_f] * 3, axis=-1)
    T_vid_c_aim = (display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8)
    R_vid_c_aim = (display_encode_tool.L2C_sRGB(R_vid) * 255).astype(np.uint8)
    T_vid_c_f_aim = (display_encode_tool.L2C_sRGB(T_vid_f) * 255).astype(np.uint8)
    R_vid_c_f_aim = (display_encode_tool.L2C_sRGB(R_vid_f) * 255).astype(np.uint8)

    aim_cos_similarity_list = []
    Spearman_matrix_cos_list = []
    aim_cos_similarity_f_list = []
    Spearman_matrix_cos_f_list = []

    for sam_model_index in tqdm(range(len(all_sam_model_list))):
        sam_model_name = all_sam_model_list[sam_model_index]
        feature_extractor = SAMFeatureExtractor(
            model_type=sam_vit_list[sam_model_index],
            checkpoint_path=rf"E:\Py_codes\LVM_Comparision\SAM_repo/{sam_model_name}.pth",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        T_features = feature_extractor.extract_features_from_numpy(T_vid_c_aim)
        R_features = feature_extractor.extract_features_from_numpy(R_vid_c_aim)
        T_features_f = feature_extractor.extract_features_from_numpy(T_vid_c_f_aim)
        R_features_f = feature_extractor.extract_features_from_numpy(R_vid_c_f_aim)

        cos_similarity = float(F.cosine_similarity(T_features.view(1, -1), R_features.view(1, -1)).cpu())
        aim_cos_similarity_list.append(cos_similarity)
        cos_similarity_f = float(
            F.cosine_similarity(T_features_f.view(1, -1), R_features_f.view(1, -1)).cpu())
        aim_cos_similarity_f_list.append(cos_similarity_f)

        # 下面是计算Spearman Correlation部分
        Spearman_matrix_cos = np.zeros([len(rho_list), len(multiplier_list)])
        Spearman_matrix_cos_f = np.zeros([len(rho_list), len(multiplier_list)])
        for rho_index, rho_value in tqdm(enumerate(rho_list)):
            S_gt = np.interp(rho_value, castleCSF_result_rho_list, castleCSF_result_sensitivity_list)
            for multiplier_index, multiplier_value in enumerate(multiplier_list):
                S_test = multiplier_value * S_gt
                T_vid, R_vid = generate_band_lim_noise(W=default_W, H=default_H, freq_band=rho_value,
                                                       L_b=default_L_b, contrast=1 / S_test, ppd=ppd_number)
                T_vid_f, R_vid_f = generate_band_lim_noise_fix_random_seed(W=default_W, H=default_H,
                                                                           freq_band=rho_value,
                                                                           L_b=default_L_b, contrast=1 / S_test,
                                                                           ppd=ppd_number)
                T_vid = np.stack([T_vid] * 3, axis=-1)
                R_vid = np.stack([R_vid] * 3, axis=-1)
                T_vid_f = np.stack([T_vid_f] * 3, axis=-1)
                R_vid_f = np.stack([R_vid_f] * 3, axis=-1)
                T_vid_c = (display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8)
                R_vid_c = (display_encode_tool.L2C_sRGB(R_vid) * 255).astype(np.uint8)
                T_vid_c_f = (display_encode_tool.L2C_sRGB(T_vid_f) * 255).astype(np.uint8)
                R_vid_c_f = (display_encode_tool.L2C_sRGB(R_vid_f) * 255).astype(np.uint8)
                T_features = feature_extractor.extract_features_from_numpy(T_vid_c)
                R_features = feature_extractor.extract_features_from_numpy(R_vid_c)
                T_features_f = feature_extractor.extract_features_from_numpy(T_vid_c_f)
                R_features_f = feature_extractor.extract_features_from_numpy(R_vid_c_f)

                cos_similarity = float(F.cosine_similarity(T_features.view(1, -1), R_features.view(1, -1)).cpu())
                cos_similarity_f = float(
                    F.cosine_similarity(T_features_f.view(1, -1), R_features_f.view(1, -1)).cpu())

                Spearman_matrix_cos[rho_index, multiplier_index] = cos_similarity
                Spearman_matrix_cos_f[rho_index, multiplier_index] = cos_similarity_f
        Spearman_matrix_cos_list.append(Spearman_matrix_cos.tolist())
        Spearman_matrix_cos_f_list.append(Spearman_matrix_cos_f.tolist())
    json_file_name = r'new_data_logs\sam\different_rho/sam_test_on_band_lim_noise_different_rho_band_contour_plot_ppd_60_final.json'
    save_json_file_name = r'new_data_logs\sam\different_rho/sam_test_on_band_lim_noise_different_rho_band_contour_plot_ppd_60_final_aim.json'
    with open(json_file_name, 'r') as fp:
        json_data = json.load(fp)
    json_data['aim_cos_similarity_list'] = aim_cos_similarity_list
    json_data['aim_cos_similarity_f_list'] = aim_cos_similarity_f_list
    json_data['rho_list'] = rho_list.tolist()
    json_data['multiplier_list'] = multiplier_list.tolist()
    json_data['Spearman_matrix_cos_list'] = Spearman_matrix_cos_list
    json_data['Spearman_matrix_cos_f_list'] = Spearman_matrix_cos_f_list

    with open(save_json_file_name, 'w') as fp:
        json.dump(json_data, fp)