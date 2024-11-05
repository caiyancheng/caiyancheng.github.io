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
save_root_path = 'new_data_logs/lpips/different_rho_RG'
os.makedirs(save_root_path, exist_ok=True)
loss_fn_alex = lpips.LPIPS(net='alex').eval()
loss_fn_vgg = lpips.LPIPS(net='vgg').eval()
loss_fn_squeeze = lpips.LPIPS(net='squeeze').eval()

default_W = 224
default_H = 224
default_R = 1
rho_list = np.logspace(np.log10(0.5), np.log10(32), 20)
contrast_list = np.logspace(np.log10(0.001), np.log10(0.2), 20)
default_O = 0
default_contrast = 1
default_L_b = 100
default_ppd = 60

csv_data = {}
csv_data['rho'] = []
csv_data['contrast'] = []
csv_data['loss_fn_alex'] = []
csv_data['loss_fn_vgg'] = []
csv_data['loss_fn_squeeze'] = []

json_plot_data = {}
json_plot_data['rho_matrix'] = []
json_plot_data['contrast_matrix'] = []
json_plot_data['loss_fn_alex_matrix'] = []
json_plot_data['loss_fn_vgg_matrix'] = []
json_plot_data['loss_fn_squeeze_matrix'] = []

plot_rho_matrix = np.zeros([len(rho_list), len(contrast_list)])
plot_contrast_matrix = np.zeros([len(rho_list), len(contrast_list)])
plot_loss_fn_alex_matrix = np.zeros([len(rho_list), len(contrast_list)])
plot_loss_fn_vgg_matrix = np.zeros([len(rho_list), len(contrast_list)])
plot_loss_fn_squeeze_matrix = np.zeros([len(rho_list), len(contrast_list)])

for rho_index in range(len(rho_list)):
    rho_value = rho_list[rho_index]
    for contrast_index in range(len(contrast_list)):
        contrast_value = contrast_list[contrast_index]
        csv_data['rho'].append(rho_value)
        csv_data['contrast'].append(contrast_value)
        plot_rho_matrix[rho_index, contrast_index] = rho_value
        plot_contrast_matrix[rho_index, contrast_index] = contrast_value
        T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=default_R, rho=rho_value, O=default_O,
                                          L_b=default_L_b, contrast=contrast_value, ppd=default_ppd, color_direction='rg')
        T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
        R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
        T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
        R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
        norm_T_vid_ct = (T_vid_ct - 0.5) * 2
        norm_R_vid_ct = (R_vid_ct - 0.5) * 2

        loss_fn_alex_value = float(loss_fn_alex(norm_T_vid_ct, norm_R_vid_ct).cpu())
        loss_fn_vgg_value = float(loss_fn_vgg(norm_T_vid_ct, norm_R_vid_ct).cpu())
        loss_fn_squeeze_value = float(loss_fn_squeeze(norm_T_vid_ct, norm_R_vid_ct).cpu())

        csv_data['loss_fn_alex'].append(loss_fn_alex_value)
        plot_loss_fn_alex_matrix[rho_index, contrast_index] = loss_fn_alex_value
        csv_data['loss_fn_vgg'].append(loss_fn_vgg_value)
        plot_loss_fn_vgg_matrix[rho_index, contrast_index] = loss_fn_vgg_value
        csv_data['loss_fn_squeeze'].append(loss_fn_squeeze_value)
        plot_loss_fn_squeeze_matrix[rho_index, contrast_index] = loss_fn_squeeze_value

        df = pd.DataFrame(csv_data)
        df.to_csv(os.path.join(save_root_path, 'lpips_test_on_gabors_different_rho_contour_plot_ppd_{default_ppd}_RG_temporary.csv'),
                  index=False)
json_plot_data['rho_matrix'].append(plot_rho_matrix.tolist())
json_plot_data['contrast_matrix'].append(plot_contrast_matrix.tolist())
json_plot_data['loss_fn_alex_matrix'].append(plot_loss_fn_alex_matrix.tolist())
json_plot_data['loss_fn_vgg_matrix'].append(plot_loss_fn_vgg_matrix.tolist())
json_plot_data['loss_fn_squeeze_matrix'].append(plot_loss_fn_squeeze_matrix.tolist())

df = pd.DataFrame(csv_data)
df.to_csv(os.path.join(save_root_path, f'lpips_test_on_gabors_different_rho_contour_plot_ppd_{default_ppd}_RG_final.csv'), index=False)
with open(os.path.join(save_root_path, f'lpips_test_on_gabors_different_rho_contour_plot_ppd_{default_ppd}_RG_final.json'), 'w') as fp:
    json.dump(json_plot_data, fp)
