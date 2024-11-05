import sys

sys.path.append('E:\Py_codes\LVM_Comparision')
import numpy as np
import torch
from Band_limit_noise_generator.generate_plot_band_lim_noise import generate_band_lim_noise, \
    generate_band_lim_noise_fix_random_seed
import torch.nn.functional as F
import pandas as pd
import json
from tqdm import tqdm
from stlpips_pytorch import stlpips
import os
from display_encoding import display_encode

display_encode_tool = display_encode(400)

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt

# Only test cpd right now
# input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/stlpips/different_rho'
os.makedirs(save_root_path, exist_ok=True)
loss_fn_alex = stlpips.LPIPS(net="alex", variant="shift_tolerant").eval()
loss_fn_vgg = stlpips.LPIPS(net="vgg", variant="shift_tolerant").eval()

default_W = 224
default_H = 224
rho_list = np.logspace(np.log10(0.5), np.log10(32), 20)
rho_list[0] = 0.5
rho_list[-1] = 32
default_L_b = 100
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
default_ppd = 60
multiplier_list = np.logspace(np.log10(0.5), np.log10(2), 10)

castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_rho_sensitivity_data.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_rho_list = castleCSF_result_data['rho_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']
middle_rho = 10 ** ((np.log10(rho_list[0]) + np.log10(rho_list[-1])) / 2)
middle_rho_S = np.interp(middle_rho, castleCSF_result_rho_list, castleCSF_result_sensitivity_list)
T_vid, R_vid = generate_band_lim_noise(W=default_W, H=default_H, freq_band=middle_rho,
                                       L_b=default_L_b, contrast=1 / middle_rho_S, ppd=default_ppd)
T_vid_f, R_vid_f = generate_band_lim_noise_fix_random_seed(W=default_W, H=default_H, freq_band=middle_rho,
                                                           L_b=default_L_b, contrast=1 / middle_rho_S, ppd=default_ppd)
T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
T_vid_f_c = display_encode_tool.L2C_sRGB(T_vid_f)
R_vid_f_c = display_encode_tool.L2C_sRGB(R_vid_f)
T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32)[None, None, ...]
R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32)[None, None, ...]
T_vid_f_ct = torch.tensor(T_vid_f_c, dtype=torch.float32)[None, None, ...]
R_vid_f_ct = torch.tensor(R_vid_f_c, dtype=torch.float32)[None, None, ...]
T_vid_ct = T_vid_ct.expand(-1, 3, -1, -1)
R_vid_ct = R_vid_ct.expand(-1, 3, -1, -1)
T_vid_f_ct = T_vid_f_ct.expand(-1, 3, -1, -1)
R_vid_f_ct = R_vid_f_ct.expand(-1, 3, -1, -1)
norm_T_vid_ct = (T_vid_ct - 0.5) * 2
norm_R_vid_ct = (R_vid_ct - 0.5) * 2
norm_T_vid_f_ct = (T_vid_f_ct - 0.5) * 2
norm_R_vid_f_ct = (R_vid_f_ct - 0.5) * 2

aim_loss_fn_alex_value = float(loss_fn_alex(norm_T_vid_ct, norm_R_vid_ct).cpu())
aim_loss_fn_vgg_value = float(loss_fn_vgg(norm_T_vid_ct, norm_R_vid_ct).cpu())
aim_loss_fn_alex_value_f = float(loss_fn_alex(norm_T_vid_f_ct, norm_R_vid_f_ct).cpu())
aim_loss_fn_vgg_value_f = float(loss_fn_vgg(norm_T_vid_f_ct, norm_R_vid_f_ct).cpu())

Spearman_matrix_alex = np.zeros([len(rho_list), len(multiplier_list)])
Spearman_matrix_vgg = np.zeros([len(rho_list), len(multiplier_list)])
Spearman_matrix_alex_f = np.zeros([len(rho_list), len(multiplier_list)])
Spearman_matrix_vgg_f = np.zeros([len(rho_list), len(multiplier_list)])

for rho_index, rho_value in tqdm(enumerate(rho_list)):
    S_gt = np.interp(rho_value, castleCSF_result_rho_list, castleCSF_result_sensitivity_list)
    for multiplier_index, multiplier_value in enumerate(multiplier_list):
        S_test = multiplier_value * S_gt
        T_vid, R_vid = generate_band_lim_noise(W=default_W, H=default_H, freq_band=rho_value,
                                               L_b=default_L_b, contrast=1 / S_test, ppd=default_ppd)
        T_vid_f, R_vid_f = generate_band_lim_noise_fix_random_seed(W=default_W, H=default_H, freq_band=rho_value,
                                                                   L_b=default_L_b, contrast=1 / S_test,
                                                                   ppd=default_ppd)
        T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
        R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
        T_vid_f_c = display_encode_tool.L2C_sRGB(T_vid_f)
        R_vid_f_c = display_encode_tool.L2C_sRGB(R_vid_f)
        T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32)[None, None, ...]
        R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32)[None, None, ...]
        T_vid_f_ct = torch.tensor(T_vid_f_c, dtype=torch.float32)[None, None, ...]
        R_vid_f_ct = torch.tensor(R_vid_f_c, dtype=torch.float32)[None, None, ...]
        T_vid_ct = T_vid_ct.expand(-1, 3, -1, -1)
        R_vid_ct = R_vid_ct.expand(-1, 3, -1, -1)
        T_vid_f_ct = T_vid_f_ct.expand(-1, 3, -1, -1)
        R_vid_f_ct = R_vid_f_ct.expand(-1, 3, -1, -1)
        norm_T_vid_ct = (T_vid_ct - 0.5) * 2
        norm_R_vid_ct = (R_vid_ct - 0.5) * 2
        norm_T_vid_f_ct = (T_vid_f_ct - 0.5) * 2
        norm_R_vid_f_ct = (R_vid_f_ct - 0.5) * 2
        loss_fn_alex_value = float(loss_fn_alex(norm_T_vid_ct, norm_R_vid_ct).cpu())
        loss_fn_vgg_value = float(loss_fn_vgg(norm_T_vid_ct, norm_R_vid_ct).cpu())
        loss_fn_alex_value_f = float(loss_fn_alex(norm_T_vid_f_ct, norm_R_vid_f_ct).cpu())
        loss_fn_vgg_value_f = float(loss_fn_vgg(norm_T_vid_f_ct, norm_R_vid_f_ct).cpu())
        Spearman_matrix_alex[rho_index, multiplier_index] = loss_fn_alex_value
        Spearman_matrix_vgg[rho_index, multiplier_index] = loss_fn_vgg_value
        Spearman_matrix_alex_f[rho_index, multiplier_index] = loss_fn_alex_value_f
        Spearman_matrix_vgg_f[rho_index, multiplier_index] = loss_fn_vgg_value_f

json_file_name = r'new_data_logs\stlpips\different_rho/stlpips_test_on_band_lim_noise_different_rho_contour_plot_ppd_60_final.json'
save_json_file_name = r'new_data_logs\stlpips\different_rho/stlpips_test_on_band_lim_noise_different_rho_contour_plot_ppd_60_final_aim.json'
with open(json_file_name, 'r') as fp:
    json_data = json.load(fp)
json_data['aim_loss_fn_alex_value'] = aim_loss_fn_alex_value
json_data['aim_loss_fn_vgg_value'] = aim_loss_fn_vgg_value
json_data['aim_loss_fn_alex_value_f'] = aim_loss_fn_alex_value_f
json_data['aim_loss_fn_vgg_value_f'] = aim_loss_fn_vgg_value_f
json_data['rho_list'] = rho_list.tolist()
json_data['multiplier_list'] = multiplier_list.tolist()
json_data['Spearman_matrix_alex'] = Spearman_matrix_alex.tolist()
json_data['Spearman_matrix_vgg'] = Spearman_matrix_vgg.tolist()
json_data['Spearman_matrix_alex_f'] = Spearman_matrix_alex_f.tolist()
json_data['Spearman_matrix_vgg_f'] = Spearman_matrix_vgg_f.tolist()

with open(save_json_file_name, 'w') as fp:
    json.dump(json_data, fp)
