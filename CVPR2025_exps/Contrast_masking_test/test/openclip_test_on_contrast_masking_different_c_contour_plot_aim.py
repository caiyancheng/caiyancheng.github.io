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
                       ('ViT-B-32', 'openai'), ('ViT-B-32', 'laion2b_s34b_b79k'), ('ViT-B-16', 'openai'),
                       ('ViT-B-16', 'laion2b_s34b_b88k'),
                       ('ViT-L-14', 'openai'), ('ViT-L-14', 'laion2b_s32b_b82k'),
                       ('convnext_base_w', 'laion2b_s13b_b82k'), ('convnext_base_w', 'laion2b_s13b_b82k_augreg'),
                       ('convnext_large_d', 'laion2b_s26b_b102k_augreg'),
                       ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg')]

default_W = 224
default_H = 224
default_rho = 2
default_O = 0
default_L_b = 32
contrast_mask_list = np.logspace(np.log10(0.005), np.log10(0.5), 20)
contrast_test_list = np.logspace(np.log10(0.01), np.log10(0.5), 20)
default_ppd = 60
default_gabor_radius = 0.5
multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)

foley_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/foley_contrast_masking_data_gabor.json'
with open(foley_result_json, 'r') as fp:
    foley_result_data = json.load(fp)
foley_result_x_mask_contrast_list = np.array(foley_result_data['mask_contrast_list'])
foley_result_y_test_contrast_list = np.array(foley_result_data['test_contrast_list'])
valid_gt_indices = [index for index, value in enumerate(foley_result_x_mask_contrast_list) if
                    value > contrast_mask_list.min() and value < 0.25]
gt_x_mask_C = foley_result_x_mask_contrast_list[valid_gt_indices]
gt_y_test_C = foley_result_y_test_contrast_list[valid_gt_indices]
middle_X_mask_C = gt_x_mask_C[round(len(gt_x_mask_C) / 2)]
middle_Y_test_C = gt_y_test_C[round(len(gt_y_test_C) / 2)]
T_vid, R_vid = generate_contrast_masking(W=default_W, H=default_H, rho=default_rho, O=default_O,
                                         L_b=default_L_b, contrast_mask=middle_X_mask_C,
                                         contrast_test=middle_Y_test_C, ppd=default_ppd,
                                         gabor_radius=default_gabor_radius)
T_vid = np.stack([T_vid] * 3, axis=-1)
R_vid = np.stack([R_vid] * 3, axis=-1)
T_vid_c_aim = Image.fromarray((display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8))
R_vid_c_aim = Image.fromarray((display_encode_tool.L2C_sRGB(R_vid) * 255).astype(np.uint8))

aim_cos_similarity_list = []
Spearman_matrix_cos_list = []
for clip_model_cell in tqdm(all_clip_model_list):
    clip_model_name = clip_model_cell[0]
    clip_model_trainset = clip_model_cell[1]
    model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_model_trainset,
                                                                 cache_dir=r'E:\Openclip_cache')
    model.eval()
    model.cuda()

    T_vid_ct = preprocess(T_vid_c_aim).unsqueeze(0).cuda()
    R_vid_ct = preprocess(R_vid_c_aim).unsqueeze(0).cuda()
    T_features = model.encode_image(T_vid_ct)
    R_features = model.encode_image(R_vid_ct)
    cos_similarity = float(F.cosine_similarity(T_features, R_features).cpu())
    aim_cos_similarity_list.append(cos_similarity)

    Spearman_matrix_cos = np.zeros([len(gt_x_mask_C), len(multiplier_list)])
    for contrast_mask_index in tqdm(range(len(gt_x_mask_C))):
        contrast_mask_value = gt_x_mask_C[contrast_mask_index]
        contrast_test_value = gt_y_test_C[contrast_mask_index]
        for multiplier_index, multiplier_value in enumerate(multiplier_list):
            C_test = contrast_test_value * multiplier_value
            T_vid, R_vid = generate_contrast_masking(W=default_W, H=default_H, rho=default_rho, O=default_O,
                                                     L_b=default_L_b, contrast_mask=contrast_mask_value,
                                                     contrast_test=C_test, ppd=default_ppd,
                                                     gabor_radius=default_gabor_radius)
            T_vid = np.stack([T_vid] * 3, axis=-1)
            R_vid = np.stack([R_vid] * 3, axis=-1)
            T_vid_c = Image.fromarray((display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8))
            R_vid_c = Image.fromarray((display_encode_tool.L2C_sRGB(R_vid) * 255).astype(np.uint8))
            T_vid_ct = preprocess(T_vid_c).unsqueeze(0).cuda()
            R_vid_ct = preprocess(R_vid_c).unsqueeze(0).cuda()
            T_features = model.encode_image(T_vid_ct)
            R_features = model.encode_image(R_vid_ct)
            cos_similarity = float(F.cosine_similarity(T_features, R_features).cpu())
            Spearman_matrix_cos[contrast_mask_index, multiplier_index] = cos_similarity
    Spearman_matrix_cos_list.append(Spearman_matrix_cos.tolist())
json_file_name = r'new_data_logs\openclip\different_c/openclip_test_on_contrast_masking_different_c_contour_plot_ppd_60_final.json'
save_json_file_name = r'new_data_logs\openclip\different_c/openclip_test_on_contrast_masking_different_c_contour_plot_ppd_60_final_aim.json'
with open(json_file_name, 'r') as fp:
    json_data = json.load(fp)
json_data['aim_cos_similarity_list'] = aim_cos_similarity_list
json_data['valid_gt_indices'] = valid_gt_indices
json_data['gt_x_mask_C'] = gt_x_mask_C.tolist()
json_data['gt_y_test_C'] = gt_y_test_C.tolist()
json_data['multiplier_list'] = multiplier_list.tolist()
json_data['Spearman_matrix_cos_list'] = Spearman_matrix_cos_list

with open(save_json_file_name, 'w') as fp:
    json.dump(json_data, fp)