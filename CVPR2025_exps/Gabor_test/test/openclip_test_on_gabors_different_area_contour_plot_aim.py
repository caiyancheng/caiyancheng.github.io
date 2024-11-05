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
import open_clip
from PIL import Image

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt
from display_encoding import display_encode

display_encode_tool = display_encode(400)

# Only test cpd right now
# Dino input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/openclip/different_area'
os.makedirs(save_root_path, exist_ok=True)
# all_clip_model_list = open_clip.list_pretrained()
all_clip_model_list = [('RN50', 'openai'), ('RN50', 'yfcc15m'), ('RN101', 'openai'), ('RN101', 'yfcc15m'),
                       ('ViT-B-32', 'openai'), ('ViT-B-32', 'laion2b_s34b_b79k'), ('ViT-B-16', 'openai'), ('ViT-B-16', 'laion2b_s34b_b88k'),
                       ('ViT-L-14', 'openai'), ('ViT-L-14', 'laion2b_s32b_b82k'), ('convnext_base_w', 'laion2b_s13b_b82k'), ('convnext_base_w', 'laion2b_s13b_b82k_augreg'),
                       ('convnext_large_d', 'laion2b_s26b_b102k_augreg'), ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg')]
test_continue = False
continue_least_model_epoch = 98

default_W = 224
default_H = 224
R_min = 0.1
R_max = 1
Area_list = np.logspace(np.log10(math.pi * R_min ** 2), np.log10(math.pi * R_max ** 2), 20)
Area_list[0] = math.pi * R_min ** 2
Area_list[-1] = math.pi * R_max ** 2
R_list = (Area_list / math.pi) ** 0.5
# rho_list = [0.5, 1, 2, 4, 8, 16, 32]
default_rho = 8
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
default_O = 0
default_contrast = 1
default_L_b = 100
default_ppd = 60
multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)

castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_area_sensitivity_data.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_area_list = castleCSF_result_data['area_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']
middle_Area = 10 ** ((np.log10(Area_list[0]) + np.log10(Area_list[-1])) / 2)
middle_Area_S = np.interp(middle_Area, castleCSF_result_area_list, castleCSF_result_sensitivity_list)
T_vid_aim, R_vid_aim = generate_gabor_patch(W=default_W, H=default_H, R=(middle_Area / math.pi) ** 0.5, rho=default_rho,
                                    O=default_O,
                                    L_b=default_L_b, contrast=1 / middle_Area_S, ppd=default_ppd,
                                    color_direction='ach')
T_vid_c_aim = Image.fromarray((display_encode_tool.L2C_sRGB(T_vid_aim) * 255).astype(np.uint8))
R_vid_c_aim = Image.fromarray((display_encode_tool.L2C_sRGB(R_vid_aim) * 255).astype(np.uint8))

aim_cos_similarity_list = []
Spearman_matrix_cos_list = []
for clip_model_index in tqdm(range(len(all_clip_model_list))):
    clip_model_cell = all_clip_model_list[clip_model_index]
    clip_model_name = clip_model_cell[0]
    clip_model_trainset = clip_model_cell[1]

    model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_model_trainset,
                                                                 cache_dir=r'E:\Openclip_cache')  # preprocess的input图片大小应该是H,W,3
    model.eval()
    model.cuda()

    T_vid_ct_aim = preprocess(T_vid_c_aim).unsqueeze(0).cuda()
    R_vid_ct_aim = preprocess(R_vid_c_aim).unsqueeze(0).cuda()
    T_features_aim = model.encode_image(T_vid_ct_aim)
    R_features_aim = model.encode_image(R_vid_ct_aim)
    cos_similarity = float(F.cosine_similarity(T_features_aim, R_features_aim).cpu())
    aim_cos_similarity_list.append(cos_similarity)

    Spearman_matrix_cos = np.zeros([len(Area_list), len(multiplier_list)])
    for Area_index, Area_value in enumerate(Area_list):
        S_gt = np.interp(Area_value, castleCSF_result_area_list, castleCSF_result_sensitivity_list)
        for multiplier_index, multiplier_value in enumerate(multiplier_list):
            S_test = multiplier_value * S_gt
            T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=(Area_value / math.pi) ** 0.5,
                                                rho=default_rho,
                                                O=default_O,
                                                L_b=default_L_b, contrast=1 / S_test, ppd=default_ppd,
                                                color_direction='ach')
            T_vid_c = Image.fromarray((display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8))
            R_vid_c = Image.fromarray((display_encode_tool.L2C_sRGB(R_vid) * 255).astype(np.uint8))
            T_vid_ct= preprocess(T_vid_c).unsqueeze(0).cuda()
            R_vid_ct = preprocess(R_vid_c).unsqueeze(0).cuda()
            T_features = model.encode_image(T_vid_ct)
            R_features = model.encode_image(R_vid_ct)
            cos_similarity = float(F.cosine_similarity(T_features, R_features).cpu())
            Spearman_matrix_cos[Area_index, multiplier_index] = cos_similarity
    Spearman_matrix_cos_list.append(Spearman_matrix_cos.tolist())

json_file_name = r'new_data_logs\openclip\different_area/openclip_test_on_gabors_different_area_contour_plot_ppd_60_final.json'
save_json_file_name = r'new_data_logs\openclip\different_area/openclip_test_on_gabors_different_area_contour_plot_ppd_60_final_aim.json'
with open(json_file_name, 'r') as fp:
    json_data = json.load(fp)
json_data['aim_cos_similarity_list'] = aim_cos_similarity_list
json_data['area_list'] = Area_list.tolist()
json_data['multiplier_list'] = multiplier_list.tolist()
json_data['Spearman_matrix_cos_list'] = Spearman_matrix_cos_list

with open(save_json_file_name, 'w') as fp:
    json.dump(json_data, fp)
