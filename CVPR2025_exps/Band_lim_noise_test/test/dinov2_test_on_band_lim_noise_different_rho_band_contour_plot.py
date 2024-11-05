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
display_encode_tool = display_encode(400)

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt

# Only test cpd right now
# Dinov2 input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/dinov2/different_rho'
os.makedirs(save_root_path, exist_ok=True)
all_backbone_list = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14',
                     'dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg']
# all_backbone_list = ['dinov2_vits14']

default_W = 224
default_H = 224
rho_list = np.logspace(np.log10(0.5), np.log10(32), 20)
default_L_b = 100
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
default_ppd = 60

# csv_data = {}
# csv_data['backbone_name'] = []
# csv_data['rho'] = []
# csv_data['contrast'] = []
# csv_data['final_feature_L1_similarity'] = []
# csv_data['final_feature_L2_similarity'] = []
# csv_data['final_feature_cos_similarity'] = []
# csv_data['intermediate_feature_L1_similarity'] = []
# csv_data['intermediate_feature_L2_similarity'] = []
# csv_data['intermediate_feature_cos_similarity'] = []
# csv_data['final_feature_L1_similarity_fix_random_seed'] = []
# csv_data['final_feature_L2_similarity_fix_random_seed'] = []
# csv_data['final_feature_cos_similarity_fix_random_seed'] = []
# csv_data['intermediate_feature_L1_similarity_fix_random_seed'] = []
# csv_data['intermediate_feature_L2_similarity_fix_random_seed'] = []
# csv_data['intermediate_feature_cos_similarity_fix_random_seed'] = []

json_plot_data = {}
json_plot_data['backbone_name'] = []
json_plot_data['rho_matrix'] = []
json_plot_data['contrast_matrix'] = []
json_plot_data['final_feature_L1_similarity_matrix'] = []
json_plot_data['final_feature_L2_similarity_matrix'] = []
json_plot_data['final_feature_cos_similarity_matrix'] = []
json_plot_data['intermediate_feature_L1_similarity_matrix'] = []
json_plot_data['intermediate_feature_L2_similarity_matrix'] = []
json_plot_data['intermediate_feature_cos_similarity_matrix'] = []
json_plot_data['final_feature_L1_similarity_matrix_fix_random_seed'] = []
json_plot_data['final_feature_L2_similarity_matrix_fix_random_seed'] = []
json_plot_data['final_feature_cos_similarity_matrix_fix_random_seed'] = []
json_plot_data['intermediate_feature_L1_similarity_matrix_fix_random_seed'] = []
json_plot_data['intermediate_feature_L2_similarity_matrix_fix_random_seed'] = []
json_plot_data['intermediate_feature_cos_similarity_matrix_fix_random_seed'] = []

for backbone_name in tqdm(all_backbone_list):
    json_plot_data['backbone_name'].append(backbone_name)
    plot_rho_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_contrast_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_L1_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_L2_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_cos_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_intermediate_feature_L1_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_intermediate_feature_L2_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_intermediate_feature_cos_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_L1_similarity_matrix_f = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_L2_similarity_matrix_f = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_cos_similarity_matrix_f = np.zeros([len(rho_list), len(contrast_list)])
    plot_intermediate_feature_L1_similarity_matrix_f = np.zeros([len(rho_list), len(contrast_list)])
    plot_intermediate_feature_L2_similarity_matrix_f = np.zeros([len(rho_list), len(contrast_list)])
    plot_intermediate_feature_cos_similarity_matrix_f = np.zeros([len(rho_list), len(contrast_list)])

    backbone_model = torch.hub.load('facebookresearch/dinov2', backbone_name)
    backbone_model.eval()
    backbone_model.cuda()

    for rho_index in tqdm(range(len(rho_list))):
        rho_value = rho_list[rho_index]
        for contrast_index in range(len(contrast_list)):
            contrast_value = contrast_list[contrast_index]
            # csv_data['backbone_name'].append(backbone_name)
            # csv_data['rho'].append(rho_value)
            # csv_data['contrast'].append(contrast_value)
            plot_rho_matrix[rho_index, contrast_index] = rho_value
            plot_contrast_matrix[rho_index, contrast_index] = contrast_value
            T_vid, R_vid = generate_band_lim_noise(W=default_W, H=default_H, freq_band=rho_value,
                                                   L_b=default_L_b, contrast=contrast_value, ppd=default_ppd)
            T_vid_f, R_vid_f = generate_band_lim_noise_fix_random_seed(W=default_W, H=default_H, freq_band=rho_value,
                                                                       L_b=default_L_b, contrast=contrast_value,
                                                                       ppd=default_ppd)
            T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
            R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
            T_vid_f_c = display_encode_tool.L2C_sRGB(T_vid_f)
            R_vid_f_c = display_encode_tool.L2C_sRGB(R_vid_f)
            T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32)[None, None, ...]
            R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32)[None, None, ...]
            T_vid_f_ct = torch.tensor(T_vid_f_c, dtype=torch.float32)[None, None, ...]
            R_vid_f_ct = torch.tensor(R_vid_f_c, dtype=torch.float32)[None, None, ...]
            T_vid_ct = T_vid_ct.expand(-1, 3, -1, -1).cuda()
            R_vid_ct = R_vid_ct.expand(-1, 3, -1, -1).cuda()
            test_feature = backbone_model(T_vid_ct)
            reference_feature = backbone_model(R_vid_ct)
            L1_similarity = float(torch.norm(test_feature - reference_feature, p=1).cpu())
            L2_similarity = float(torch.norm(test_feature - reference_feature, p=2).cpu())
            cos_similarity = float(F.cosine_similarity(test_feature, reference_feature).cpu())

            T_vid_f_ct = T_vid_f_ct.expand(-1, 3, -1, -1).cuda()
            R_vid_f_ct = R_vid_f_ct.expand(-1, 3, -1, -1).cuda()
            test_feature_f = backbone_model(T_vid_f_ct)
            reference_feature_f = backbone_model(R_vid_f_ct)
            L1_similarity_f = float(torch.norm(test_feature_f - reference_feature_f, p=1).cpu())
            L2_similarity_f = float(torch.norm(test_feature_f - reference_feature_f, p=2).cpu())
            cos_similarity_f = float(F.cosine_similarity(test_feature_f, reference_feature_f).cpu())

            test_feature_intermediate = backbone_model.get_intermediate_layers(T_vid_ct, n=4)
            test_feature_intermediate = torch.stack(test_feature_intermediate)
            reference_feature_intermediate = backbone_model.get_intermediate_layers(R_vid_ct, n=4)
            reference_feature_intermediate = torch.stack(reference_feature_intermediate)
            L1_similarity_intermediate = float(
                torch.norm(test_feature_intermediate - reference_feature_intermediate, p=1).cpu())
            L2_similarity_intermediate = float(
                torch.norm(test_feature_intermediate - reference_feature_intermediate, p=2).cpu())
            cos_similarity_intermediate = float(F.cosine_similarity(test_feature_intermediate.view(1, -1),
                                                                    reference_feature_intermediate.view(1,
                                                                                                        -1)).cpu())
            test_feature_intermediate_f = backbone_model.get_intermediate_layers(T_vid_f_ct, n=4)
            test_feature_intermediate_f = torch.stack(test_feature_intermediate_f)
            reference_feature_intermediate_f = backbone_model.get_intermediate_layers(R_vid_f_ct, n=4)
            reference_feature_intermediate_f = torch.stack(reference_feature_intermediate_f)
            L1_similarity_intermediate_f = float(
                torch.norm(test_feature_intermediate_f - reference_feature_intermediate_f, p=1).cpu())
            L2_similarity_intermediate_f = float(
                torch.norm(test_feature_intermediate_f - reference_feature_intermediate_f, p=2).cpu())
            cos_similarity_intermediate_f = float(F.cosine_similarity(test_feature_intermediate_f.view(1, -1),
                                                                      reference_feature_intermediate_f.view(1,
                                                                                                            -1)).cpu())

            # csv_data['final_feature_L1_similarity'].append(L1_similarity)
            # csv_data['intermediate_feature_L1_similarity'].append(L1_similarity_intermediate)
            plot_final_feature_L1_similarity_matrix[rho_index, contrast_index] = L1_similarity
            plot_intermediate_feature_L1_similarity_matrix[rho_index, contrast_index] = L1_similarity_intermediate

            # csv_data['final_feature_L2_similarity'].append(L2_similarity)
            # csv_data['intermediate_feature_L2_similarity'].append(L2_similarity_intermediate)
            plot_final_feature_L2_similarity_matrix[rho_index, contrast_index] = L2_similarity
            plot_intermediate_feature_L2_similarity_matrix[rho_index, contrast_index] = L2_similarity_intermediate

            # csv_data['final_feature_cos_similarity'].append(cos_similarity)
            # csv_data['intermediate_feature_cos_similarity'].append(cos_similarity_intermediate)
            plot_final_feature_cos_similarity_matrix[rho_index, contrast_index] = cos_similarity
            plot_intermediate_feature_cos_similarity_matrix[rho_index, contrast_index] = cos_similarity_intermediate

            # csv_data['final_feature_L1_similarity_fix_random_seed'].append(L1_similarity_f)
            # csv_data['intermediate_feature_L1_similarity_fix_random_seed'].append(L1_similarity_intermediate_f)
            plot_final_feature_L1_similarity_matrix_f[rho_index, contrast_index] = L1_similarity_f
            plot_intermediate_feature_L1_similarity_matrix_f[rho_index, contrast_index] = L1_similarity_intermediate_f

            # csv_data['final_feature_L2_similarity_fix_random_seed'].append(L2_similarity_f)
            # csv_data['intermediate_feature_L2_similarity_fix_random_seed'].append(L2_similarity_intermediate_f)
            plot_final_feature_L2_similarity_matrix_f[rho_index, contrast_index] = L2_similarity_f
            plot_intermediate_feature_L2_similarity_matrix_f[rho_index, contrast_index] = L2_similarity_intermediate_f

            # csv_data['final_feature_cos_similarity_fix_random_seed'].append(cos_similarity_f)
            # csv_data['intermediate_feature_cos_similarity_fix_random_seed'].append(cos_similarity_intermediate_f)
            plot_final_feature_cos_similarity_matrix_f[rho_index, contrast_index] = cos_similarity_f
            plot_intermediate_feature_cos_similarity_matrix_f[rho_index, contrast_index] = cos_similarity_intermediate_f

            # df = pd.DataFrame(csv_data)
            # df.to_csv(os.path.join(save_root_path, f'dinov2_test_on_band_lim_noise_different_rho_band_contour_plot_ppd_{default_ppd}_temporary.csv'), index=False)
    json_plot_data['rho_matrix'].append(plot_rho_matrix.tolist())
    json_plot_data['contrast_matrix'].append(plot_contrast_matrix.tolist())
    json_plot_data['final_feature_L1_similarity_matrix'].append(plot_final_feature_L1_similarity_matrix.tolist())
    json_plot_data['final_feature_L2_similarity_matrix'].append(plot_final_feature_L2_similarity_matrix.tolist())
    json_plot_data['final_feature_cos_similarity_matrix'].append(plot_final_feature_cos_similarity_matrix.tolist())
    json_plot_data['intermediate_feature_L1_similarity_matrix'].append(plot_intermediate_feature_L1_similarity_matrix.tolist())
    json_plot_data['intermediate_feature_L2_similarity_matrix'].append(plot_intermediate_feature_L2_similarity_matrix.tolist())
    json_plot_data['intermediate_feature_cos_similarity_matrix'].append(plot_intermediate_feature_cos_similarity_matrix.tolist())
    json_plot_data['final_feature_L1_similarity_matrix_fix_random_seed'].append(
        plot_final_feature_L1_similarity_matrix_f.tolist())
    json_plot_data['final_feature_L2_similarity_matrix_fix_random_seed'].append(
        plot_final_feature_L2_similarity_matrix_f.tolist())
    json_plot_data['final_feature_cos_similarity_matrix_fix_random_seed'].append(
        plot_final_feature_cos_similarity_matrix_f.tolist())
    json_plot_data['intermediate_feature_L1_similarity_matrix_fix_random_seed'].append(
        plot_intermediate_feature_L1_similarity_matrix_f.tolist())
    json_plot_data['intermediate_feature_L2_similarity_matrix_fix_random_seed'].append(
        plot_intermediate_feature_L2_similarity_matrix_f.tolist())
    json_plot_data['intermediate_feature_cos_similarity_matrix_fix_random_seed'].append(
        plot_intermediate_feature_cos_similarity_matrix_f.tolist())

# df = pd.DataFrame(csv_data)
# df.to_csv(os.path.join(save_root_path, f'dinov2_test_on_band_lim_noise_different_rho_band_contour_plot_ppd_{default_ppd}_final.csv'), index=False)
with open(os.path.join(save_root_path, f'dinov2_test_on_band_lim_noise_different_rho_band_contour_plot_ppd_{default_ppd}_final.json'), 'w') as fp:
    json.dump(json_plot_data, fp)
