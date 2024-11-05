import matplotlib.pyplot as plt
import json
import numpy as np
import itertools
import os
from tqdm import tqdm
from scipy.interpolate import interp1d
from sklearn.metrics import root_mean_squared_error
ppd = 60
from scipy.stats import pearsonr, spearmanr

save_root_path = 'new_contour_plots/baseline/different_c'
os.makedirs(save_root_path, exist_ok=True)

foley_result_json = r'../contrast_masking_data_gabor_on_noise.json'
with open(foley_result_json, 'r') as fp:
    foley_result_data = json.load(fp)
foley_result_x_mask_contrast_list = foley_result_data['mask_contrast_list']
foley_result_y_test_contrast_list = foley_result_data['test_contrast_list']

x_contrast_mask_ticks = [0.01, 0.1]
y_contrast_test_ticks = [0.01, 0.1]

plt.figure(figsize=(4, 3), dpi=300)
plt.plot(foley_result_x_mask_contrast_list, foley_result_y_test_contrast_list, 'k', linestyle='-', linewidth=2,
         label='Human Results', marker='o')
plt.xlabel('Mask Contrast', fontsize=12)
plt.ylabel('Test Contrast', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.xticks(x_contrast_mask_ticks, x_contrast_mask_ticks)
plt.yticks(y_contrast_test_ticks, y_contrast_test_ticks)
plt.tight_layout()
# plt.legend(loc='lower right')
plt.gca().axis('off')
# plt.show()
plt.savefig(r'E:\All_Conference_Papers\CVPR25\Images/CM.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close()
