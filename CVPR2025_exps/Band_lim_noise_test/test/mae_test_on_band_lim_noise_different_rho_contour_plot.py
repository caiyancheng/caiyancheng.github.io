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
from transformers import AutoImageProcessor, ViTMAEForPreTraining
display_encode_tool = display_encode(400)

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt

# Only test cpd right now
# input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/mae/different_rho'
os.makedirs(save_root_path, exist_ok=True)
all_mae_model_list = ['vit-mae-base', 'vit-mae-large', 'vit-mae-huge']

torch.manual_seed(8)
default_noise = torch.rand(1, 196)
default_W = 224
default_H = 224
rho_list = np.logspace(np.log10(0.5), np.log10(32), 20)
default_L_b = 100
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
default_ppd = 60

json_plot_data = {}
json_plot_data['mae_model_name'] = []
json_plot_data['rho_matrix'] = []
json_plot_data['contrast_matrix'] = []
json_plot_data['final_feature_L1_similarity_matrix'] = []
json_plot_data['final_feature_L2_similarity_matrix'] = []
json_plot_data['final_feature_cos_similarity_matrix'] = []
json_plot_data['final_feature_L1_similarity_matrix_fix_random_seed'] = []
json_plot_data['final_feature_L2_similarity_matrix_fix_random_seed'] = []
json_plot_data['final_feature_cos_similarity_matrix_fix_random_seed'] = []

for mae_model_index in tqdm(range(len(all_mae_model_list))):
    mae_model_name = all_mae_model_list[mae_model_index]
    plot_rho_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_contrast_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_L1_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_L2_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_cos_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_L1_similarity_matrix_f = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_L2_similarity_matrix_f = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_cos_similarity_matrix_f = np.zeros([len(rho_list), len(contrast_list)])

    processor = AutoImageProcessor.from_pretrained(f'facebook/{mae_model_name}')
    model = ViTMAEForPreTraining.from_pretrained(f'facebook/{mae_model_name}')
    model.eval()

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
            T_vid_c = Image.fromarray((display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8))
            R_vid_c = Image.fromarray((display_encode_tool.L2C_sRGB(R_vid) * 255).astype(np.uint8))
            T_vid_f_c = Image.fromarray((display_encode_tool.L2C_sRGB(T_vid_f) * 255).astype(np.uint8))
            R_vid_f_c = Image.fromarray((display_encode_tool.L2C_sRGB(R_vid_f) * 255).astype(np.uint8))
            T_vid_ct_inputs = processor(images=T_vid_c, return_tensors="pt")
            R_vid_ct_inputs = processor(images=R_vid_c, return_tensors="pt")
            T_vid_f_ct_inputs = processor(images=T_vid_f_c, return_tensors="pt")
            R_vid_f_ct_inputs = processor(images=R_vid_f_c, return_tensors="pt")
            T_output = model.vit(**T_vid_ct_inputs)
            R_output = model.vit(**R_vid_ct_inputs)
            T_output_f = model.vit(**T_vid_f_ct_inputs)
            R_output_f = model.vit(**R_vid_f_ct_inputs)
            T_features = T_output.last_hidden_state
            R_features = R_output.last_hidden_state
            T_features_f = T_output_f.last_hidden_state
            R_features_f = R_output_f.last_hidden_state

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
    json_plot_data['mae_model_name'].append(mae_model_name)
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

with open(os.path.join(save_root_path, f'mae_test_on_band_lim_noise_different_rho_band_contour_plot_ppd_{default_ppd}_final.json'), 'w') as fp:
    json.dump(json_plot_data, fp)
