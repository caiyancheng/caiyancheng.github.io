import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import numpy as np
import torch
from Gabor_test_stimulus_generator.generate_plot_gabor_functions_new import generate_gabor_patch
import torch.nn.functional as F
import pandas as pd
import json
from tqdm import tqdm
import lpips
import os

torch.hub.set_dir(r'E:\Torch_hub')
from display_encoding import display_encode
display_encode_tool = display_encode(400)
import matplotlib.pyplot as plt

# Only test cpd right now
# input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/lpips/different_rho_YV'
os.makedirs(save_root_path, exist_ok=True)
loss_fn_alex = lpips.LPIPS(net='alex').eval()
loss_fn_vgg = lpips.LPIPS(net='vgg').eval()
loss_fn_squeeze = lpips.LPIPS(net='squeeze').eval()

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

rho_change_list = rho_list[rho_list < 16]
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

aim_loss_fn_alex_value = float(loss_fn_alex(norm_T_vid_ct, norm_R_vid_ct).cpu())
aim_loss_fn_vgg_value = float(loss_fn_vgg(norm_T_vid_ct, norm_R_vid_ct).cpu())

# 下面是计算Spearman Correlation部分
Spearman_matrix_lpips_alex = np.zeros([len(rho_change_list), len(multiplier_list)])
Spearman_matrix_lpips_vgg = np.zeros([len(rho_change_list), len(multiplier_list)])
for rho_index, rho_value in tqdm(enumerate(rho_change_list)):
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

        loss_fn_alex_value = float(loss_fn_alex(norm_T_vid_ct, norm_R_vid_ct).cpu())
        loss_fn_vgg_value = float(loss_fn_vgg(norm_T_vid_ct, norm_R_vid_ct).cpu())
        Spearman_matrix_lpips_alex[rho_index, multiplier_index] = loss_fn_alex_value
        Spearman_matrix_lpips_vgg[rho_index, multiplier_index] = loss_fn_vgg_value

json_file_name = r'new_data_logs\lpips\different_rho_YV/lpips_test_on_gabors_different_rho_contour_plot_ppd_60_YV_final.json'
save_json_file_name = r'new_data_logs\lpips\different_rho_YV/lpips_test_on_gabors_different_rho_contour_plot_ppd_60_YV_final_aim.json'
with open(json_file_name, 'r') as fp:
    json_data = json.load(fp)
json_data['aim_loss_fn_alex_value'] = aim_loss_fn_alex_value
json_data['aim_loss_fn_vgg_value'] = aim_loss_fn_vgg_value
json_data['rho_YV_list'] = rho_list.tolist()
json_data['rho_change_list'] = rho_change_list.tolist()
json_data['multiplier_list'] = multiplier_list.tolist()
json_data['Spearman_matrix_lpips_alex'] = Spearman_matrix_lpips_alex.tolist()
json_data['Spearman_matrix_lpips_vgg'] = Spearman_matrix_lpips_vgg.tolist()

with open(save_json_file_name, 'w') as fp:
    json.dump(json_data, fp)
