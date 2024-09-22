import numpy as np
import torch
from Contrast_masking_generator.generate_plot_contrast_masking_band_limit_noise import generate_contrast_masking_band_limit_noise
import torch.nn.functional as F
import pandas as pd
import json
from tqdm import tqdm
import lpips
import os
from display_encoding import display_encode
display_encode_tool = display_encode(400, 2.2)

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt

# Only test cpd right now
# input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/lpips/different_rho'
os.makedirs(save_root_path, exist_ok=True)
loss_fn_alex = lpips.LPIPS(net='alex').eval()
loss_fn_vgg = lpips.LPIPS(net='vgg').eval()
loss_fn_squeeze = lpips.LPIPS(net='squeeze').eval()

default_W = 224
default_H = 224
T_freq_band_list = np.logspace(np.log10(0.5), np.log10(32), 20)
default_R_freq_band = 4
default_L_b = 100
default_contrast_mask = 0.1
contrast_test_list = np.logspace(np.log10(0.01), np.log10(0.5), 20)
default_ppd = 60

csv_data = {}
csv_data['T_freq_band'] = []
csv_data['contrast_test'] = []
csv_data['loss_fn_alex'] = []
csv_data['loss_fn_vgg'] = []
csv_data['loss_fn_squeeze'] = []

json_plot_data = {}
json_plot_data['T_freq_band_matrix'] = []
json_plot_data['contrast_test_matrix'] = []
json_plot_data['loss_fn_alex_matrix'] = []
json_plot_data['loss_fn_vgg_matrix'] = []
json_plot_data['loss_fn_squeeze_matrix'] = []

# reference_pattern = default_L_b * torch.ones([1, 3, default_H, default_W])
# norm_reference_pattern = (reference_pattern - 0.5) * 2

plot_contrast_mask_matrix = np.zeros([len(T_freq_band_list), len(contrast_test_list)])
plot_contrast_test_matrix = np.zeros([len(T_freq_band_list), len(contrast_test_list)])
plot_loss_fn_alex_matrix = np.zeros([len(T_freq_band_list), len(contrast_test_list)])
plot_loss_fn_vgg_matrix = np.zeros([len(T_freq_band_list), len(contrast_test_list)])
plot_loss_fn_squeeze_matrix = np.zeros([len(T_freq_band_list), len(contrast_test_list)])

for T_freq_band_index in range(len(T_freq_band_list)):
    T_freq_band_value = T_freq_band_list[T_freq_band_index]
    for contrast_test_index in range(len(contrast_test_list)):
        contrast_test_value = contrast_test_list[contrast_test_index]
        csv_data['T_freq_band'].append(T_freq_band_value)
        csv_data['contrast_test'].append(contrast_test_value)
        plot_contrast_mask_matrix[T_freq_band_index, contrast_test_index] = T_freq_band_value
        plot_contrast_test_matrix[T_freq_band_index, contrast_test_index] = contrast_test_value
        T_vid, R_vid = generate_contrast_masking_band_limit_noise(W=default_W, H=default_H,
                                                                  T_freq_band=T_freq_band_value,
                                                                  R_freq_band=default_R_freq_band,
                                                                  L_b=default_L_b,
                                                                  contrast_mask=T_freq_band_value,
                                                                  contrast_test=contrast_test_value,
                                                                  ppd=default_ppd)
        T_vid_c = display_encode_tool.L2C(T_vid)
        R_vid_c = display_encode_tool.L2C(R_vid)
        T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32)[None, None, ...]
        R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32)[None, None, ...]
        T_vid_ct = T_vid_ct.expand(-1, 3, -1, -1)
        R_vid_ct = R_vid_ct.expand(-1, 3, -1, -1)
        norm_T_vid_ct = (T_vid_ct - 0.5) * 2
        norm_R_vid_ct = (R_vid_ct - 0.5) * 2

        loss_fn_alex_value = float(loss_fn_alex(norm_T_vid_ct, norm_R_vid_ct).cpu())
        loss_fn_vgg_value = float(loss_fn_vgg(norm_T_vid_ct, norm_R_vid_ct).cpu())
        loss_fn_squeeze_value = float(loss_fn_squeeze(norm_T_vid_ct, norm_R_vid_ct).cpu())

        csv_data['loss_fn_alex'].append(loss_fn_alex_value)
        plot_loss_fn_alex_matrix[T_freq_band_index, contrast_test_index] = loss_fn_alex_value
        csv_data['loss_fn_vgg'].append(loss_fn_vgg_value)
        plot_loss_fn_vgg_matrix[T_freq_band_index, contrast_test_index] = loss_fn_vgg_value
        csv_data['loss_fn_squeeze'].append(loss_fn_squeeze_value)
        plot_loss_fn_squeeze_matrix[T_freq_band_index, contrast_test_index] = loss_fn_squeeze_value

        df = pd.DataFrame(csv_data)
        df.to_csv(os.path.join(save_root_path, f'lpips_test_on_contrast_masking_different_rho_contour_plot_ppd_{default_ppd}_temporary.csv'),
                  index=False)
json_plot_data['T_freq_band_matrix'].append(plot_contrast_mask_matrix.tolist())
json_plot_data['contrast_test_matrix'].append(plot_contrast_test_matrix.tolist())
json_plot_data['loss_fn_alex_matrix'].append(plot_loss_fn_alex_matrix.tolist())
json_plot_data['loss_fn_vgg_matrix'].append(plot_loss_fn_vgg_matrix.tolist())
json_plot_data['loss_fn_squeeze_matrix'].append(plot_loss_fn_squeeze_matrix.tolist())

df = pd.DataFrame(csv_data)
df.to_csv(os.path.join(save_root_path, f'lpips_test_on_contrast_masking_different_rho_contour_plot_ppd_{default_ppd}_final.csv'), index=False)
with open(os.path.join(save_root_path, f'lpips_test_on_contrast_masking_different_rho_contour_plot_ppd_{default_ppd}_final.json'), 'w') as fp:
    json.dump(json_plot_data, fp)
