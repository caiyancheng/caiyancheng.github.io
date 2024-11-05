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
from diffusers import AutoencoderKL
display_encode_tool = display_encode(400)

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt

# Only test cpd right now
# input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/vae/different_rho'
os.makedirs(save_root_path, exist_ok=True)
all_vae_model_list = ['stable-diffusion-v1-5/stable-diffusion-v1-5', 'stabilityai/stable-diffusion-xl-base-1.0']

default_W = 224
default_H = 224
rho_list = np.logspace(np.log10(0.5), np.log10(32), 20)
default_L_b = 100
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
default_ppd = 60

json_plot_data = {}
json_plot_data['vae_model_name'] = []
json_plot_data['rho_matrix'] = []
json_plot_data['contrast_matrix'] = []
json_plot_data['final_feature_L1_similarity_matrix'] = []
json_plot_data['final_feature_L2_similarity_matrix'] = []
json_plot_data['final_feature_cos_similarity_matrix'] = []
json_plot_data['final_feature_L1_similarity_matrix_fix_random_seed'] = []
json_plot_data['final_feature_L2_similarity_matrix_fix_random_seed'] = []
json_plot_data['final_feature_cos_similarity_matrix_fix_random_seed'] = []

for vae_model_index in tqdm(range(len(all_vae_model_list))):
    vae_model_name = all_vae_model_list[vae_model_index]
    plot_rho_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_contrast_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_L1_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_L2_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_cos_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_L1_similarity_matrix_f = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_L2_similarity_matrix_f = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_cos_similarity_matrix_f = np.zeros([len(rho_list), len(contrast_list)])

    vae_model = AutoencoderKL.from_pretrained(vae_model_name, subfolder="vae")

    for rho_index in range(len(rho_list)):
        rho_value = rho_list[rho_index]
        for contrast_index in range(len(contrast_list)):
            contrast_value = contrast_list[contrast_index]
            plot_rho_matrix[rho_index, contrast_index] = rho_value
            plot_contrast_matrix[rho_index, contrast_index] = contrast_value
            T_vid, R_vid = generate_band_lim_noise(W=default_W, H=default_H, freq_band=rho_value,
                                                   L_b=default_L_b, contrast=contrast_value, ppd=default_ppd)
            T_vid_f, R_vid_f = generate_band_lim_noise_fix_random_seed(W=default_W, H=default_H, freq_band=rho_value,
                                                                       L_b=default_L_b, contrast=contrast_value,
                                                                       ppd=default_ppd)
            T_vid = np.stack([T_vid] * 3, axis=-1)
            R_vid = np.stack([R_vid] * 3, axis=-1)
            T_vid_f = np.stack([T_vid_f] * 3, axis=-1)
            R_vid_f = np.stack([R_vid_f] * 3, axis=-1)

            T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
            R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
            T_vid_c_f = display_encode_tool.L2C_sRGB(T_vid_f)
            R_vid_c_f = display_encode_tool.L2C_sRGB(R_vid_f)
            with torch.no_grad():
                T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
                R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
                T_vid_ct_f = torch.tensor(T_vid_c_f, dtype=torch.float32).permute(2, 0, 1)[None, ...]
                R_vid_ct_f = torch.tensor(R_vid_c_f, dtype=torch.float32).permute(2, 0, 1)[None, ...]
                norm_T_vid_ct = (T_vid_ct - 0.5) * 2
                norm_R_vid_ct = (R_vid_ct - 0.5) * 2
                norm_T_vid_ct_f = (T_vid_ct_f - 0.5) * 2
                norm_R_vid_ct_f = (R_vid_ct_f - 0.5) * 2
                T_latent = vae_model.encode(norm_T_vid_ct)
                R_latent = vae_model.encode(norm_R_vid_ct)
                T_latent_f = vae_model.encode(norm_T_vid_ct_f)
                R_latent_f = vae_model.encode(norm_R_vid_ct_f)
                T_features = T_latent.latent_dist.sample()
                R_features = R_latent.latent_dist.sample()
                T_features_f = T_latent_f.latent_dist.sample()
                R_features_f = R_latent_f.latent_dist.sample()

                L1_similarity = float(torch.norm(T_features - R_features, p=1).cpu())
                L2_similarity = float(torch.norm(T_features - R_features, p=2).cpu())
                cos_similarity = float(F.cosine_similarity(T_features.view(1, -1), R_features.view(1, -1)).cpu())
                L1_similarity_f = float(torch.norm(T_features_f - R_features_f, p=1).cpu())
                L2_similarity_f = float(torch.norm(T_features_f - R_features_f, p=2).cpu())
                cos_similarity_f = float(F.cosine_similarity(T_features_f.view(1, -1), R_features_f.view(1, -1)).cpu())

            plot_final_feature_L1_similarity_matrix[rho_index, contrast_index] = L1_similarity
            plot_final_feature_L2_similarity_matrix[rho_index, contrast_index] = L2_similarity
            plot_final_feature_cos_similarity_matrix[rho_index, contrast_index] = cos_similarity
            plot_final_feature_L1_similarity_matrix_f[rho_index, contrast_index] = L1_similarity_f
            plot_final_feature_L2_similarity_matrix_f[rho_index, contrast_index] = L2_similarity_f
            plot_final_feature_cos_similarity_matrix_f[rho_index, contrast_index] = cos_similarity_f
    json_plot_data['vae_model_name'].append(vae_model_name)
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

with open(os.path.join(save_root_path, f'vae_test_on_band_lim_noise_different_rho_band_contour_plot_ppd_{default_ppd}_final.json'), 'w') as fp:
    json.dump(json_plot_data, fp)
