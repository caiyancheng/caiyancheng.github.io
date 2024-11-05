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
save_root_path = 'new_data_logs/openclip/different_rho'
os.makedirs(save_root_path, exist_ok=True)
all_clip_model_list = [('RN50', 'openai'), ('RN50', 'yfcc15m'), ('RN101', 'openai'), ('RN101', 'yfcc15m'),
                       ('ViT-B-32', 'openai'), ('ViT-B-32', 'laion2b_s34b_b79k'), ('ViT-B-16', 'openai'), ('ViT-B-16', 'laion2b_s34b_b88k'),
                       ('ViT-L-14', 'openai'), ('ViT-L-14', 'laion2b_s32b_b82k'), ('convnext_base_w', 'laion2b_s13b_b82k'), ('convnext_base_w', 'laion2b_s13b_b82k_augreg'),
                       ('convnext_large_d', 'laion2b_s26b_b102k_augreg'), ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg')]
test_continue = False

default_W = 224
default_H = 224
default_R = 1
rho_list = np.logspace(np.log10(0.5), np.log10(32), 20)
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
default_O = 0
default_contrast = 1
default_L_b = 100
default_ppd = 60

if not test_continue:
    csv_data = {}
    csv_data['clip_model_name'] = []
    csv_data['clip_model_trainset'] = []
    csv_data['rho'] = []
    csv_data['contrast'] = []
    csv_data['final_feature_L1_similarity'] = []
    csv_data['final_feature_L2_similarity'] = []
    csv_data['final_feature_cos_similarity'] = []
else:
    df = pd.read_csv(os.path.join(save_root_path, f'openclip_test_on_gabors_different_rho_contour_plot_ppd_{default_ppd}_temporary.csv'))
    csv_data = df.to_dict(orient='list')

json_plot_data = {}
json_plot_data['clip_model_name'] = []
json_plot_data['clip_model_trainset'] = []
json_plot_data['rho_matrix'] = []
json_plot_data['contrast_matrix'] = []
json_plot_data['final_feature_L1_similarity_matrix'] = []
json_plot_data['final_feature_L2_similarity_matrix'] = []
json_plot_data['final_feature_cos_similarity_matrix'] = []

for clip_model_cell in tqdm(all_clip_model_list):
    clip_model_name = clip_model_cell[0]
    clip_model_trainset = clip_model_cell[1]
    plot_rho_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_contrast_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_L1_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_L2_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_cos_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])

    model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_model_trainset, cache_dir=r'E:\Openclip_cache') #preprocess的input图片大小应该是H,W,3
    model.eval()
    model.cuda()

    for rho_index in range(len(rho_list)):
        rho_value = rho_list[rho_index]
        for contrast_index in range(len(contrast_list)):
            contrast_value = contrast_list[contrast_index]
            if test_continue:
                check_df = df[(df['clip_model_name'] == clip_model_name) & (df['clip_model_trainset'] == clip_model_trainset)
                              & (abs(df['rho'] - rho_value) < 1e-5) & (abs(df['contrast'] - contrast_value) < 1e-5)]
                if len(check_df) == 1:
                    plot_rho_matrix[rho_index, contrast_index] = rho_value
                    plot_contrast_matrix[rho_index, contrast_index] = contrast_value
                    plot_final_feature_L1_similarity_matrix[rho_index, contrast_index] = check_df['final_feature_L1_similarity']
                    plot_final_feature_L2_similarity_matrix[rho_index, contrast_index] = check_df['final_feature_L2_similarity']
                    plot_final_feature_cos_similarity_matrix[rho_index, contrast_index] = check_df['final_feature_cos_similarity']
                    continue
                elif len(check_df) > 1:
                    raise EOFError('CSV Data length find rrepeats')
            csv_data['clip_model_name'].append(clip_model_name)
            csv_data['clip_model_trainset'].append(clip_model_trainset)
            csv_data['rho'].append(rho_value)
            csv_data['contrast'].append(contrast_value)
            plot_rho_matrix[rho_index, contrast_index] = rho_value
            plot_contrast_matrix[rho_index, contrast_index] = contrast_value
            T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=default_R, rho=rho_value, O=default_O,
                                                L_b=default_L_b, contrast=contrast_value, ppd=default_ppd,
                                                color_direction='ach')
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
            plot_final_feature_L1_similarity_matrix[rho_index, contrast_index] = L1_similarity
            csv_data['final_feature_L2_similarity'].append(L2_similarity)
            plot_final_feature_L2_similarity_matrix[rho_index, contrast_index] = L2_similarity
            csv_data['final_feature_cos_similarity'].append(cos_similarity)
            plot_final_feature_cos_similarity_matrix[rho_index, contrast_index] = cos_similarity

            df = pd.DataFrame(csv_data)
            df.to_csv(os.path.join(save_root_path,
                                   f'openclip_test_on_gabors_different_rho_contour_plot_ppd_{default_ppd}_temporary.csv'),
                      index=False)
    json_plot_data['clip_model_name'].append(clip_model_name)
    json_plot_data['clip_model_trainset'].append(clip_model_trainset)
    json_plot_data['rho_matrix'].append(plot_rho_matrix.tolist())
    json_plot_data['contrast_matrix'].append(plot_contrast_matrix.tolist())
    json_plot_data['final_feature_L1_similarity_matrix'].append(plot_final_feature_L1_similarity_matrix.tolist())
    json_plot_data['final_feature_L2_similarity_matrix'].append(plot_final_feature_L2_similarity_matrix.tolist())
    json_plot_data['final_feature_cos_similarity_matrix'].append(plot_final_feature_cos_similarity_matrix.tolist())

df = pd.DataFrame(csv_data)
df.to_csv(
    os.path.join(save_root_path, f'openclip_test_on_gabors_different_rho_contour_plot_ppd_{default_ppd}_final.csv'),
    index=False)
with open(
        os.path.join(save_root_path, f'openclip_test_on_gabors_different_rho_contour_plot_ppd_{default_ppd}_final.json'),
        'w') as fp:
    json.dump(json_plot_data, fp)
