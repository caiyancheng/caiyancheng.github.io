import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import numpy as np
import torch
from Contrast_masking_generator.generate_plot_contrast_masking import generate_contrast_masking
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
save_root_path = 'new_data_logs/vae/different_c'
os.makedirs(save_root_path, exist_ok=True)
all_vae_model_list = ['stable-diffusion-v1-5/stable-diffusion-v1-5', 'stabilityai/stable-diffusion-xl-base-1.0']

default_W = 224
default_H = 224
default_rho = 2
default_O = 0
default_L_b = 32
contrast_mask_list = np.logspace(np.log10(0.005), np.log10(0.5), 20)
contrast_test_list = np.logspace(np.log10(0.01), np.log10(0.5), 20)
default_ppd = 60
default_gabor_radius = 0.5

csv_data = {}
csv_data['vae_model_name'] = []
csv_data['contrast_mask'] = []
csv_data['contrast_test'] = []
csv_data['final_feature_L1_similarity'] = []
csv_data['final_feature_L2_similarity'] = []
csv_data['final_feature_cos_similarity'] = []

json_plot_data = {}
json_plot_data['vae_model_name'] = []
json_plot_data['contrast_mask_matrix'] = []
json_plot_data['contrast_test_matrix'] = []
json_plot_data['final_feature_L1_similarity_matrix'] = []
json_plot_data['final_feature_L2_similarity_matrix'] = []
json_plot_data['final_feature_cos_similarity_matrix'] = []

for vae_model_index in tqdm(range(len(all_vae_model_list))):
    vae_model_name = all_vae_model_list[vae_model_index]
    plot_contrast_mask_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
    plot_contrast_test_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
    plot_final_feature_L1_similarity_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
    plot_final_feature_L2_similarity_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
    plot_final_feature_cos_similarity_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])

    vae_model = AutoencoderKL.from_pretrained(vae_model_name, subfolder="vae")

    for contrast_mask_index in range(len(contrast_mask_list)):
        contrast_mask_value = contrast_mask_list[contrast_mask_index]
        for contrast_test_index in range(len(contrast_test_list)):
            contrast_test_value = contrast_test_list[contrast_test_index]
            csv_data['vae_model_name'].append(vae_model_name)
            csv_data['contrast_mask'].append(contrast_mask_value)
            csv_data['contrast_test'].append(contrast_test_value)
            plot_contrast_mask_matrix[contrast_mask_index, contrast_test_index] = contrast_mask_value
            plot_contrast_test_matrix[contrast_mask_index, contrast_test_index] = contrast_test_value
            T_vid, R_vid = generate_contrast_masking(W=default_W, H=default_H, rho=default_rho, O=default_O,
                                                     L_b=default_L_b, contrast_mask=contrast_mask_value,
                                                     contrast_test=contrast_test_value, ppd=default_ppd,
                                                     gabor_radius=default_gabor_radius)
            T_vid = np.stack([T_vid] * 3, axis=-1)
            R_vid = np.stack([R_vid] * 3, axis=-1)
            T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
            R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
            with torch.no_grad():
                T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
                R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
                norm_T_vid_ct = (T_vid_ct - 0.5) * 2
                norm_R_vid_ct = (R_vid_ct - 0.5) * 2
                T_latent = vae_model.encode(norm_T_vid_ct)
                R_latent = vae_model.encode(norm_R_vid_ct)
                T_features = T_latent.latent_dist.sample()
                R_features = R_latent.latent_dist.sample()
                L1_similarity = float(torch.norm(T_features - R_features, p=1).cpu())
                L2_similarity = float(torch.norm(T_features - R_features, p=2).cpu())
                cos_similarity = float(F.cosine_similarity(T_features.view(1, -1), R_features.view(1, -1)).cpu())

            csv_data['final_feature_L1_similarity'].append(L1_similarity)
            plot_final_feature_L1_similarity_matrix[contrast_mask_index, contrast_test_index] = L1_similarity

            csv_data['final_feature_L2_similarity'].append(L2_similarity)
            plot_final_feature_L2_similarity_matrix[contrast_mask_index, contrast_test_index] = L2_similarity

            csv_data['final_feature_cos_similarity'].append(cos_similarity)
            plot_final_feature_cos_similarity_matrix[contrast_mask_index, contrast_test_index] = cos_similarity

            df = pd.DataFrame(csv_data)
            df.to_csv(os.path.join(save_root_path, f'vae_test_on_contrast_masking_different_c_contour_plot_ppd_{default_ppd}_temporary.csv'), index=False)
    json_plot_data['vae_model_name'].append(vae_model_name)
    json_plot_data['contrast_test_matrix'].append(plot_contrast_test_matrix.tolist())
    json_plot_data['contrast_mask_matrix'].append(plot_contrast_mask_matrix.tolist())
    json_plot_data['final_feature_L1_similarity_matrix'].append(plot_final_feature_L1_similarity_matrix.tolist())
    json_plot_data['final_feature_L2_similarity_matrix'].append(plot_final_feature_L2_similarity_matrix.tolist())
    json_plot_data['final_feature_cos_similarity_matrix'].append(plot_final_feature_cos_similarity_matrix.tolist())

df = pd.DataFrame(csv_data)
df.to_csv(os.path.join(save_root_path, f'vae_test_on_contrast_masking_different_c_contour_plot_ppd_{default_ppd}_final.csv'), index=False)
with open(os.path.join(save_root_path, f'vae_test_on_contrast_masking_different_c_contour_plot_ppd_{default_ppd}_final.json'), 'w') as fp:
    json.dump(json_plot_data, fp)
