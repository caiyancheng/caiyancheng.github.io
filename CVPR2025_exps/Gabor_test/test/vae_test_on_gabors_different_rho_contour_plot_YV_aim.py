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
from diffusers import AutoencoderKL

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt
from display_encoding import display_encode

display_encode_tool = display_encode(400)

# Only test cpd right now
# Dino input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/vae/different_rho_YV'
os.makedirs(save_root_path, exist_ok=True)
all_vae_model_list = ['stable-diffusion-v1-5/stable-diffusion-v1-5', 'stabilityai/stable-diffusion-xl-base-1.0']

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
T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=default_R, rho=middle_rho,
                                    O=default_O, L_b=default_L_b, contrast=1 / middle_rho_S,
                                    ppd=default_ppd, color_direction='yv')
T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
norm_T_vid_ct = (T_vid_ct - 0.5) * 2
norm_R_vid_ct = (R_vid_ct - 0.5) * 2
aim_cos_similarity_list = []
Spearman_matrix_cos_list = []
rho_change_list = rho_list[rho_list < 16]
for vae_model_index in tqdm(range(len(all_vae_model_list))):
    vae_model_name = all_vae_model_list[vae_model_index]
    vae_model = AutoencoderKL.from_pretrained(vae_model_name, subfolder="vae")
    T_latent = vae_model.encode(norm_T_vid_ct)
    R_latent = vae_model.encode(norm_R_vid_ct)
    T_features = T_latent.latent_dist.sample()
    R_features = R_latent.latent_dist.sample()
    cos_similarity = float(F.cosine_similarity(T_features.view(1, -1), R_features.view(1, -1)).cpu())
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
            T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
            R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
            T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
            R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
            norm_T_vid_ct = (T_vid_ct - 0.5) * 2
            norm_R_vid_ct = (R_vid_ct - 0.5) * 2
            T_latent = vae_model.encode(norm_T_vid_ct)
            R_latent = vae_model.encode(norm_R_vid_ct)
            T_features = T_latent.latent_dist.sample()
            R_features = R_latent.latent_dist.sample()
            cos_similarity = float(F.cosine_similarity(T_features.view(1, -1), R_features.view(1, -1)).cpu())
            Spearman_matrix_cos[rho_index, multiplier_index] = cos_similarity
    Spearman_matrix_cos_list.append(Spearman_matrix_cos.tolist())

json_file_name = r'new_data_logs\vae\different_rho_YV/vae_test_on_gabors_different_rho_contour_plot_ppd_60_YV_final.json'
save_json_file_name = r'new_data_logs\vae\different_rho_YV/vae_test_on_gabors_different_rho_contour_plot_ppd_60_YV_final_aim.json'
with open(json_file_name, 'r') as fp:
    json_data = json.load(fp)
json_data['aim_cos_similarity_list'] = aim_cos_similarity_list
json_data['rho_YV_list'] = rho_list.tolist()
json_data['rho_change_list'] = rho_change_list.tolist()
json_data['multiplier_list'] = multiplier_list.tolist()
json_data['Spearman_matrix_cos_list'] = Spearman_matrix_cos_list

with open(save_json_file_name, 'w') as fp:
    json.dump(json_data, fp)