import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import numpy as np
import torch
from Contrast_masking_generator.generate_plot_contrast_masking_gabor_on_noise import generate_contrast_masking_gabor_on_noise
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
save_root_path = 'new_data_logs/stlpips/different_c'
os.makedirs(save_root_path, exist_ok=True)
loss_fn_alex = stlpips.LPIPS(net="alex", variant="shift_tolerant").eval()
loss_fn_vgg = stlpips.LPIPS(net="vgg", variant="shift_tolerant").eval()

default_W = 224
default_H = 224
default_Mask_upper_frequency = 12
default_L_b = 37
contrast_mask_list = np.logspace(np.log10(0.005), np.log10(0.5), 20)
contrast_test_list = np.logspace(np.log10(0.01), np.log10(0.5), 20)
default_ppd = 60
default_radius_test = 0.8
default_rho_test = 1.2

csv_data = {}
csv_data['contrast_mask'] = []
csv_data['contrast_test'] = []
csv_data['loss_fn_alex'] = []
csv_data['loss_fn_vgg'] = []

json_plot_data = {}
json_plot_data['contrast_mask_matrix'] = []
json_plot_data['contrast_test_matrix'] = []
json_plot_data['loss_fn_alex_matrix'] = []
json_plot_data['loss_fn_vgg_matrix'] = []

# reference_pattern = default_L_b * torch.ones([1, 3, default_H, default_W])
# norm_reference_pattern = (reference_pattern - 0.5) * 2

plot_contrast_mask_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
plot_contrast_test_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
plot_loss_fn_alex_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])
plot_loss_fn_vgg_matrix = np.zeros([len(contrast_mask_list), len(contrast_test_list)])

for contrast_mask_index in tqdm(range(len(contrast_mask_list))):
    contrast_mask_value = contrast_mask_list[contrast_mask_index]
    for contrast_test_index in range(len(contrast_test_list)):
        contrast_test_value = contrast_test_list[contrast_test_index]
        csv_data['contrast_mask'].append(contrast_mask_value)
        csv_data['contrast_test'].append(contrast_test_value)
        plot_contrast_mask_matrix[contrast_mask_index, contrast_test_index] = contrast_mask_value
        plot_contrast_test_matrix[contrast_mask_index, contrast_test_index] = contrast_test_value
        T_vid, R_vid = generate_contrast_masking_gabor_on_noise(W=default_W, H=default_H,
                                                                sigma=default_radius_test,
                                                                rho=default_rho_test,
                                                                Mask_upper_frequency=default_Mask_upper_frequency,
                                                                L_b=default_L_b,
                                                                contrast_mask=contrast_mask_value,
                                                                contrast_test=contrast_test_value,
                                                                ppd=default_ppd)
        T_vid_c = display_encode_tool.L2C_sRGB(T_vid)
        R_vid_c = display_encode_tool.L2C_sRGB(R_vid)
        T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32)[None, None, ...]
        R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32)[None, None, ...]
        T_vid_ct = T_vid_ct.expand(-1, 3, -1, -1)
        R_vid_ct = R_vid_ct.expand(-1, 3, -1, -1)
        norm_T_vid_ct = (T_vid_ct - 0.5) * 2
        norm_R_vid_ct = (R_vid_ct - 0.5) * 2

        loss_fn_alex_value = float(loss_fn_alex(norm_T_vid_ct, norm_R_vid_ct).cpu())
        loss_fn_vgg_value = float(loss_fn_vgg(norm_T_vid_ct, norm_R_vid_ct).cpu())

        csv_data['loss_fn_alex'].append(loss_fn_alex_value)
        plot_loss_fn_alex_matrix[contrast_mask_index, contrast_test_index] = loss_fn_alex_value
        csv_data['loss_fn_vgg'].append(loss_fn_vgg_value)
        plot_loss_fn_vgg_matrix[contrast_mask_index, contrast_test_index] = loss_fn_vgg_value

        df = pd.DataFrame(csv_data)
        df.to_csv(os.path.join(save_root_path, f'stlpips_test_on_contrast_masking_different_c_contour_plot_ppd_{default_ppd}_temporary.csv'),
                  index=False)
json_plot_data['contrast_mask_matrix'].append(plot_contrast_mask_matrix.tolist())
json_plot_data['contrast_test_matrix'].append(plot_contrast_test_matrix.tolist())
json_plot_data['loss_fn_alex_matrix'].append(plot_loss_fn_alex_matrix.tolist())
json_plot_data['loss_fn_vgg_matrix'].append(plot_loss_fn_vgg_matrix.tolist())

df = pd.DataFrame(csv_data)
df.to_csv(os.path.join(save_root_path, f'stlpips_test_on_contrast_masking_different_c_contour_plot_ppd_{default_ppd}_final.csv'), index=False)
with open(os.path.join(save_root_path, f'stlpips_test_on_contrast_masking_different_c_contour_plot_ppd_{default_ppd}_final.json'), 'w') as fp:
    json.dump(json_plot_data, fp)

