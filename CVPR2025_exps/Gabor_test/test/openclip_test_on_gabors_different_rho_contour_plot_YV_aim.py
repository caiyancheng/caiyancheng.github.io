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
save_root_path = 'new_data_logs/openclip/different_rho_YV'
os.makedirs(save_root_path, exist_ok=True)
all_clip_model_list = [('RN50', 'openai'), ('RN50', 'yfcc15m'), ('RN101', 'openai'), ('RN101', 'yfcc15m'),
                       ('ViT-B-32', 'openai'), ('ViT-B-32', 'laion2b_s34b_b79k'), ('ViT-B-16', 'openai'), ('ViT-B-16', 'laion2b_s34b_b88k'),
                       ('ViT-L-14', 'openai'), ('ViT-L-14', 'laion2b_s32b_b82k'), ('convnext_base_w', 'laion2b_s13b_b82k'), ('convnext_base_w', 'laion2b_s13b_b82k_augreg'),
                       ('convnext_large_d', 'laion2b_s26b_b102k_augreg'), ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg')]

default_W = 224
default_H = 224
default_R = 1
rho_list = np.logspace(np.log10(0.5), np.log10(32), 20)
rho_list[0] = 0.5
rho_list[-1] = 32
contrast_list = np.logspace(np.log10(0.001), np.log10(0.2), 20)
default_O = 0
default_contrast = 1
default_L_b = 100
default_ppd = 60
multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)

castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_rho_sensitivity_data_YV.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_rho_list = castleCSF_result_data['rho_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']
middle_rho = 10 ** ((np.log10(rho_list[0]) + np.log10(rho_list[-1])) / 2)
middle_rho_S = np.interp(middle_rho, castleCSF_result_rho_list, castleCSF_result_sensitivity_list)
T_vid_aim, R_vid_aim = generate_gabor_patch(W=default_W, H=default_H, R=default_R, rho=middle_rho,
                                    O=default_O, L_b=default_L_b, contrast=1 / middle_rho_S,
                                    ppd=default_ppd, color_direction='yv')
T_vid_c_aim = Image.fromarray((display_encode_tool.L2C_sRGB(T_vid_aim) * 255).astype(np.uint8))
R_vid_c_aim = Image.fromarray((display_encode_tool.L2C_sRGB(R_vid_aim) * 255).astype(np.uint8))

aim_cos_similarity_list = []
Spearman_matrix_cos_list = []
rho_change_list = rho_list[rho_list<16]
for clip_model_cell in tqdm(all_clip_model_list):
    clip_model_name = clip_model_cell[0]
    clip_model_trainset = clip_model_cell[1]

    model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_model_trainset, cache_dir=r'E:\Openclip_cache') #preprocess的input图片大小应该是H,W,3
    model.eval()
    model.cuda()

    T_vid_ct_aim = preprocess(T_vid_c_aim).unsqueeze(0).cuda()
    R_vid_ct_aim = preprocess(R_vid_c_aim).unsqueeze(0).cuda()
    T_features_aim = model.encode_image(T_vid_ct_aim)
    R_features_aim = model.encode_image(R_vid_ct_aim)
    cos_similarity = float(F.cosine_similarity(T_features_aim, R_features_aim).cpu())
    aim_cos_similarity_list.append(cos_similarity)

    Spearman_matrix_cos = np.zeros([len(rho_change_list), len(multiplier_list)])
    for rho_index, rho_value in enumerate(rho_change_list):
        S_gt = np.interp(rho_value, castleCSF_result_rho_list, castleCSF_result_sensitivity_list)
        for multiplier_index, multiplier_value in enumerate(multiplier_list):
            S_test = multiplier_value * S_gt
            T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=default_R, rho=rho_value,
                                                O=default_O,
                                                L_b=default_L_b, contrast=1 / S_test, ppd=default_ppd,
                                                color_direction='yv')
            T_vid_c = Image.fromarray((display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8))
            R_vid_c = Image.fromarray((display_encode_tool.L2C_sRGB(R_vid) * 255).astype(np.uint8))
            T_vid_ct = preprocess(T_vid_c).unsqueeze(0).cuda()
            R_vid_ct = preprocess(R_vid_c).unsqueeze(0).cuda()
            T_features = model.encode_image(T_vid_ct)
            R_features = model.encode_image(R_vid_ct)
            cos_similarity = float(F.cosine_similarity(T_features, R_features).cpu())
            Spearman_matrix_cos[rho_index, multiplier_index] = cos_similarity
    Spearman_matrix_cos_list.append(Spearman_matrix_cos.tolist())

json_file_name = r'new_data_logs\openclip\different_rho_YV/openclip_test_on_gabors_different_rho_contour_plot_ppd_60_YV_final.json'
save_json_file_name = r'new_data_logs\openclip\different_rho_YV/openclip_test_on_gabors_different_rho_contour_plot_ppd_60_YV_final_aim.json'
with open(json_file_name, 'r') as fp:
    json_data = json.load(fp)
json_data['aim_cos_similarity_list'] = aim_cos_similarity_list
json_data['rho_YV_list'] = rho_list.tolist()
json_data['rho_change_list'] = rho_change_list.tolist()
json_data['multiplier_list'] = multiplier_list.tolist()
json_data['Spearman_matrix_cos_list'] = Spearman_matrix_cos_list
with open(save_json_file_name, 'w') as fp:
    json.dump(json_data, fp)
