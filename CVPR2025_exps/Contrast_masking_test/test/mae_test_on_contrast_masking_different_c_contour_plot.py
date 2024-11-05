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
from transformers import AutoImageProcessor, ViTMAEForPreTraining

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt
from display_encoding import display_encode

display_encode_tool = display_encode(400)
save_root_path = 'new_data_logs/mae/different_c'
os.makedirs(save_root_path, exist_ok=True)
all_mae_model_list = ['vit-mae-base', 'vit-mae-large', 'vit-mae-huge']

torch.manual_seed(8)
default_noise = torch.rand(1, 196)
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
csv_data['mae_model_name'] = []
csv_data['contrast_mask'] = []
csv_data['contrast_test'] = []
csv_data['final_feature_L1_similarity'] = []
csv_data['final_feature_L2_similarity'] = []
csv_data['final_feature_cos_similarity'] = []

json_plot_data = {}
json_plot_data['mae_model_name'] = []
json_plot_data['contrast_mask_matrix'] = []
json_plot_data['contrast_test_matrix'] = []
json_plot_data['final_feature_L1_similarity_matrix'] = []
json_plot_data['final_feature_L2_similarity_matrix'] = []
json_plot_data['final_feature_cos_similarity_matrix'] = []

for mae_model_index in tqdm(range(len(all_mae_model_list))):
    mae_model_name = all_mae_model_list[mae_model_index]
    plot_contrast_mask_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
    plot_contrast_test_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
    plot_final_feature_L1_similarity_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
    plot_final_feature_L2_similarity_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
    plot_final_feature_cos_similarity_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])

    processor = AutoImageProcessor.from_pretrained(f'facebook/{mae_model_name}')
    model = ViTMAEForPreTraining.from_pretrained(f'facebook/{mae_model_name}')
    model.eval()

    for contrast_mask_index in range(len(contrast_mask_list)):
        contrast_mask_value = contrast_mask_list[contrast_mask_index]
        for contrast_test_index in range(len(contrast_test_list)):
            contrast_test_value = contrast_test_list[contrast_test_index]
            csv_data['mae_model_name'].append(mae_model_name)
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
            T_vid_c = Image.fromarray((display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8))
            R_vid_c = Image.fromarray((display_encode_tool.L2C_sRGB(R_vid) * 255).astype(np.uint8))
            T_vid_ct_inputs = processor(images=T_vid_c, return_tensors="pt")
            R_vid_ct_inputs = processor(images=R_vid_c, return_tensors="pt")
            T_vid_ct_inputs["noise"] = default_noise
            R_vid_ct_inputs["noise"] = default_noise
            T_output = model.vit(**T_vid_ct_inputs)
            R_output = model.vit(**R_vid_ct_inputs)
            T_features = T_output.last_hidden_state
            R_features = R_output.last_hidden_state
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
            df.to_csv(os.path.join(save_root_path, f'mae_test_on_contrast_masking_different_c_contour_plot_ppd_{default_ppd}_temporary.csv'), index=False)
    json_plot_data['mae_model_name'].append(mae_model_name)
    json_plot_data['contrast_test_matrix'].append(plot_contrast_test_matrix.tolist())
    json_plot_data['contrast_mask_matrix'].append(plot_contrast_mask_matrix.tolist())
    json_plot_data['final_feature_L1_similarity_matrix'].append(plot_final_feature_L1_similarity_matrix.tolist())
    json_plot_data['final_feature_L2_similarity_matrix'].append(plot_final_feature_L2_similarity_matrix.tolist())
    json_plot_data['final_feature_cos_similarity_matrix'].append(plot_final_feature_cos_similarity_matrix.tolist())

df = pd.DataFrame(csv_data)
df.to_csv(os.path.join(save_root_path, f'mae_test_on_contrast_masking_different_c_contour_plot_ppd_{default_ppd}_final.csv'), index=False)
with open(os.path.join(save_root_path, f'mae_test_on_contrast_masking_different_c_contour_plot_ppd_{default_ppd}_final.json'), 'w') as fp:
    json.dump(json_plot_data, fp)
