import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import numpy as np
import torch
from Gabor_test_stimulus_generator.generate_plot_gabor_functions_new import generate_gabor_patch
import torch.nn.functional as F
import pandas as pd
import json
from tqdm import tqdm
from stlpips_pytorch import stlpips
import os

torch.hub.set_dir(r'E:\Torch_hub')
from display_encoding import display_encode
display_encode_tool = display_encode(400)
import matplotlib.pyplot as plt

# Only test cpd right now
# input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/stlpips/different_luminance'
os.makedirs(save_root_path, exist_ok=True)
loss_fn_alex = stlpips.LPIPS(net="alex", variant="shift_tolerant").eval()
loss_fn_vgg = stlpips.LPIPS(net="vgg", variant="shift_tolerant").eval()

display_encoded_a = 400
display_encoded_gamma = 2.2

default_W = 224
default_H = 224
default_R = 1
default_rho = 2
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
default_O = 0
default_contrast = 1
L_b_list = np.logspace(np.log10(0.1), np.log10(200), 20)
default_ppd = 60

csv_data = {}
csv_data['L_b'] = []
csv_data['contrast'] = []
csv_data['loss_fn_alex'] = []
csv_data['loss_fn_vgg'] = []

json_plot_data = {}
json_plot_data['L_b_matrix'] = []
json_plot_data['contrast_matrix'] = []
json_plot_data['loss_fn_alex_matrix'] = []
json_plot_data['loss_fn_vgg_matrix'] = []


plot_L_b_matrix = np.zeros([len(L_b_list), len(contrast_list)])
plot_contrast_matrix = np.zeros([len(L_b_list), len(contrast_list)])
plot_loss_fn_alex_matrix = np.zeros([len(L_b_list), len(contrast_list)])
plot_loss_fn_vgg_matrix = np.zeros([len(L_b_list), len(contrast_list)])

for L_b_index in tqdm(range(len(L_b_list))):
    L_b_value = L_b_list[L_b_index]
    for contrast_index in range(len(contrast_list)):
        contrast_value = contrast_list[contrast_index]
        csv_data['L_b'].append(L_b_value)
        csv_data['contrast'].append(contrast_value)
        plot_L_b_matrix[L_b_index, contrast_index] = L_b_value
        plot_contrast_matrix[L_b_index, contrast_index] = contrast_value
        T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=default_R, rho=default_rho, O=default_O,
                                          L_b=L_b_value, contrast=contrast_value, ppd=default_ppd, color_direction='ach')
        T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
        R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
        T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
        R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
        norm_T_vid_ct = (T_vid_ct - 0.5) * 2
        norm_R_vid_ct = (R_vid_ct - 0.5) * 2

        loss_fn_alex_value = float(loss_fn_alex(norm_T_vid_ct, norm_R_vid_ct).cpu())
        loss_fn_vgg_value = float(loss_fn_vgg(norm_T_vid_ct, norm_R_vid_ct).cpu())

        csv_data['loss_fn_alex'].append(loss_fn_alex_value)
        plot_loss_fn_alex_matrix[L_b_index, contrast_index] = loss_fn_alex_value
        csv_data['loss_fn_vgg'].append(loss_fn_vgg_value)
        plot_loss_fn_vgg_matrix[L_b_index, contrast_index] = loss_fn_vgg_value

        df = pd.DataFrame(csv_data)
        df.to_csv(os.path.join(save_root_path, f'stlpips_test_on_gabors_different_L_b_contour_plot_ppd_{default_ppd}_temporary.csv'), index=False)
json_plot_data['L_b_matrix'].append(plot_L_b_matrix.tolist())
json_plot_data['contrast_matrix'].append(plot_contrast_matrix.tolist())
json_plot_data['loss_fn_alex_matrix'].append(plot_loss_fn_alex_matrix.tolist())
json_plot_data['loss_fn_vgg_matrix'].append(plot_loss_fn_vgg_matrix.tolist())

df = pd.DataFrame(csv_data)
df.to_csv(os.path.join(save_root_path, f'stlpips_test_on_gabors_different_L_b_contour_plot_ppd_{default_ppd}_final.csv'), index=False)
with open(os.path.join(save_root_path, f'stlpips_test_on_gabors_different_L_b_contour_plot_ppd_{default_ppd}_final.json'), 'w') as fp:
    json.dump(json_plot_data, fp)
