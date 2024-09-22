import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import numpy as np
import torch
from Contrast_masking_generator.generate_plot_contrast_masking_band_limit_noise import generate_contrast_masking_band_limit_noise
import torch.nn.functional as F
import pandas as pd
import json
from tqdm import tqdm
import os
from display_encoding import display_encode
import open_clip
from PIL import Image
display_encode_tool = display_encode(400)

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt

# Only test cpd right now
# Dinov2 input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/openclip/different_c'
os.makedirs(save_root_path, exist_ok=True)
all_clip_model_list = [('RN50', 'openai'), ('RN50', 'yfcc15m'), ('RN101', 'openai'), ('RN101', 'yfcc15m'),
                       ('ViT-B-32', 'openai'), ('ViT-B-32', 'laion2b_s34b_b79k'), ('ViT-B-16', 'openai'), ('ViT-B-16', 'laion2b_s34b_b88k'),
                       ('ViT-L-14', 'openai'), ('ViT-L-14', 'laion2b_s32b_b82k'), ('convnext_base_w', 'laion2b_s13b_b82k'), ('convnext_base_w', 'laion2b_s13b_b82k_augreg'),
                       ('convnext_large_d', 'laion2b_s26b_b102k_augreg'), ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg')]

default_W = 224
default_H = 224
default_T_freq_band = 4
default_R_freq_band = 4
default_L_b = 100
contrast_mask_list = np.logspace(np.log10(0.005), np.log10(0.5), 20)
contrast_test_list = np.logspace(np.log10(0.01), np.log10(0.5), 20)
default_ppd = 60

csv_data = {}
csv_data['clip_model_name'] = []
csv_data['clip_model_trainset'] = []
csv_data['contrast_mask'] = []
csv_data['contrast_test'] = []
csv_data['final_feature_L1_similarity'] = []
csv_data['final_feature_L2_similarity'] = []
csv_data['final_feature_cos_similarity'] = []

json_plot_data = {}
json_plot_data['clip_model_name'] = []
json_plot_data['clip_model_trainset'] = []
json_plot_data['contrast_mask_matrix'] = []
json_plot_data['contrast_test_matrix'] = []
json_plot_data['final_feature_L1_similarity_matrix'] = []
json_plot_data['final_feature_L2_similarity_matrix'] = []
json_plot_data['final_feature_cos_similarity_matrix'] = []

for clip_model_cell in tqdm(all_clip_model_list):
    clip_model_name = clip_model_cell[0]
    clip_model_trainset = clip_model_cell[1]
    plot_contrast_mask_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
    plot_contrast_test_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
    plot_final_feature_L1_similarity_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
    plot_final_feature_L2_similarity_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
    plot_final_feature_cos_similarity_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])

    model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_model_trainset,
                                                                 cache_dir=r'E:\Openclip_cache')
    model.eval()
    model.cuda()

    for contrast_mask_index in range(len(contrast_mask_list)):
        contrast_mask_value = contrast_mask_list[contrast_mask_index]
        for contrast_test_index in range(len(contrast_test_list)):
            contrast_test_value = contrast_test_list[contrast_test_index]
            csv_data['clip_model_name'].append(clip_model_name)
            csv_data['clip_model_trainset'].append(clip_model_trainset)
            csv_data['contrast_mask'].append(contrast_mask_value)
            csv_data['contrast_test'].append(contrast_test_value)
            plot_contrast_mask_matrix[contrast_mask_index, contrast_test_index] = contrast_mask_value
            plot_contrast_test_matrix[contrast_mask_index, contrast_test_index] = contrast_test_value
            T_vid, R_vid = generate_contrast_masking_band_limit_noise(W=default_W, H=default_H,
                                                                      T_freq_band=default_T_freq_band,
                                                                      R_freq_band=default_R_freq_band,
                                                                      L_b=default_L_b,
                                                                      contrast_mask=contrast_mask_value,
                                                                      contrast_test=contrast_test_value,
                                                                      ppd=default_ppd)
            T_vid = np.stack([T_vid] * 3, axis=-1)
            R_vid = np.stack([R_vid] * 3, axis=-1)
            T_vid_c = Image.fromarray((display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8))
            R_vid_c = Image.fromarray((display_encode_tool.L2C_sRGB(R_vid) * 255).astype(np.uint8))
            T_vid_ct = preprocess(T_vid_c).unsqueeze(0).cuda()
            R_vid_ct = preprocess(R_vid_c).unsqueeze(0).cuda()
            T_features = model.encode_image(T_vid_ct)
            R_features = model.encode_image(R_vid_ct)
            L1_similarity = float(torch.norm(T_features - R_features, p=1).cpu())
            L2_similarity = float(torch.norm(T_features - R_features, p=2).cpu())
            cos_similarity = float(F.cosine_similarity(T_features, R_features).cpu())

            csv_data['final_feature_L1_similarity'].append(L1_similarity)
            plot_final_feature_L1_similarity_matrix[contrast_mask_index, contrast_test_index] = L1_similarity
            csv_data['final_feature_L2_similarity'].append(L2_similarity)
            plot_final_feature_L2_similarity_matrix[contrast_mask_index, contrast_test_index] = L2_similarity
            csv_data['final_feature_cos_similarity'].append(cos_similarity)
            plot_final_feature_cos_similarity_matrix[contrast_mask_index, contrast_test_index] = cos_similarity
            df = pd.DataFrame(csv_data)
            df.to_csv(os.path.join(save_root_path, f'openclip_test_on_contrast_masking_different_c_contour_plot_ppd_{default_ppd}_temporary.csv'), index=False)
    json_plot_data['clip_model_name'].append(clip_model_name)
    json_plot_data['clip_model_trainset'].append(clip_model_trainset)
    json_plot_data['contrast_test_matrix'].append(plot_contrast_test_matrix.tolist())
    json_plot_data['contrast_mask_matrix'].append(plot_contrast_mask_matrix.tolist())
    json_plot_data['final_feature_L1_similarity_matrix'].append(plot_final_feature_L1_similarity_matrix.tolist())
    json_plot_data['final_feature_L2_similarity_matrix'].append(plot_final_feature_L2_similarity_matrix.tolist())
    json_plot_data['final_feature_cos_similarity_matrix'].append(plot_final_feature_cos_similarity_matrix.tolist())
df = pd.DataFrame(csv_data)
df.to_csv(os.path.join(save_root_path, f'openclip_test_on_contrast_masking_different_c_contour_plot_ppd_{default_ppd}_final.csv'), index=False)
with open(os.path.join(save_root_path, f'openclip_test_on_contrast_masking_different_c_contour_plot_ppd_{default_ppd}_final.json'), 'w') as fp:
    json.dump(json_plot_data, fp)
