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
save_root_path = 'new_data_logs/sam/different_area'
os.makedirs(save_root_path, exist_ok=True)
all_sam_model_list = ['sam_vit_b_01ec64', 'sam_vit_l_0b3195', 'sam_vit_h_4b8939']
sam_vit_list = ['vit_b', 'vit_l', 'vit_h']

default_W = 224
default_H = 224
R_min = 0.1
R_max = 1
Area_list = np.logspace(np.log10(math.pi * R_min ** 2), np.log10(math.pi * R_max ** 2), 20)
Area_list[0] = math.pi * R_min ** 2
Area_list[-1] = math.pi * R_max ** 2
R_list = (Area_list / math.pi) ** 0.5
default_rho = 8
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
default_O = 0
default_contrast = 1
default_L_b = 100
ppd_list = [60]
multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)

castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_area_sensitivity_data.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_area_list = castleCSF_result_data['area_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']
middle_Area = 10 ** ((np.log10(Area_list[0]) + np.log10(Area_list[-1])) / 2)
middle_Area_S = np.interp(middle_Area, castleCSF_result_area_list, castleCSF_result_sensitivity_list)

for ppd_index in tqdm(range(len(ppd_list))):
    ppd_number = ppd_list[ppd_index]
    T_vid_aim, R_vid_aim = generate_gabor_patch(W=default_W, H=default_H, R=(middle_Area / math.pi) ** 0.5, rho=default_rho,
                                        O=default_O,
                                        L_b=default_L_b, contrast=1 / middle_Area_S, ppd=ppd_number,
                                        color_direction='ach')
    T_vid_c_aim = (display_encode_tool.L2C_sRGB(T_vid_aim) * 255).astype(np.uint8)
    R_vid_c_aim = (display_encode_tool.L2C_sRGB(R_vid_aim) * 255).astype(np.uint8)

    aim_cos_similarity_list = []
    Spearman_matrix_cos_list = []
    for sam_model_index in tqdm(range(len(all_sam_model_list))):
        sam_model_name = all_sam_model_list[sam_model_index]
        feature_extractor = SAMFeatureExtractor(
            model_type=sam_vit_list[sam_model_index],
            checkpoint_path=rf"E:\Py_codes\LVM_Comparision\SAM_repo/{sam_model_name}.pth",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        with torch.no_grad():
            T_features_aim = feature_extractor.extract_features_from_numpy(T_vid_c_aim)
            R_features_aim = feature_extractor.extract_features_from_numpy(R_vid_c_aim)
            cos_similarity = float(F.cosine_similarity(T_features_aim.view(1, -1), R_features_aim.view(1, -1)).cpu())
            aim_cos_similarity_list.append(cos_similarity)

        Spearman_matrix_cos = np.zeros([len(Area_list), len(multiplier_list)])
        for Area_index, Area_value in enumerate(Area_list):
            S_gt = np.interp(Area_value, castleCSF_result_area_list, castleCSF_result_sensitivity_list)
            for multiplier_index, multiplier_value in enumerate(multiplier_list):
                S_test = multiplier_value * S_gt
                T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=(Area_value / math.pi) ** 0.5,
                                                    rho=default_rho,
                                                    O=default_O,
                                                    L_b=default_L_b, contrast=1 / S_test, ppd=ppd_number,
                                                    color_direction='ach')
                T_vid_c = (display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8)
                R_vid_c = (display_encode_tool.L2C_sRGB(R_vid) * 255).astype(np.uint8)
                T_features = feature_extractor.extract_features_from_numpy(T_vid_c)
                R_features = feature_extractor.extract_features_from_numpy(R_vid_c)
                cos_similarity = float(F.cosine_similarity(T_features.view(1, -1), R_features.view(1, -1)).cpu())
                Spearman_matrix_cos[Area_index, multiplier_index] = cos_similarity
        Spearman_matrix_cos_list.append(Spearman_matrix_cos.tolist())

    json_file_name = rf'new_data_logs\sam\different_area/sam_test_on_gabors_different_area_contour_plot_ppd_{ppd_number}_final.json'
    save_json_file_name = rf'new_data_logs\sam\different_area/sam_test_on_gabors_different_area_contour_plot_ppd_{ppd_number}_final_aim.json'
    with open(json_file_name, 'r') as fp:
        json_data = json.load(fp)
    json_data['aim_cos_similarity_list'] = aim_cos_similarity_list
    json_data['area_list'] = Area_list.tolist()
    json_data['multiplier_list'] = multiplier_list.tolist()
    json_data['Spearman_matrix_cos_list'] = Spearman_matrix_cos_list

    with open(save_json_file_name, 'w') as fp:
        json.dump(json_data, fp)
