import pickle

import cv2
import numpy as np

# for i in range(6):
#     breakpoint()

#     data_batch = pickle.load(open(f"./data_pickles/data_batch_{i}.pkl", "rb"))
#     print(data_batch)

#     data_batch = pickle.load(open(f"./data_pickles/data_batch_after_batch_proc_{i}.pkl", "rb"))
#     print(data_batch)

#     inp = pickle.load(open(f"./data_pickles/inp_{i}.pkl", "rb"))
#     print(inp)


model_inputs = pickle.load(open("./data_pickles/model_inputs.pkl", "rb"))
for k, v in model_inputs.items():
    if v.numel() > 10000:
        print(k, v.shape)
    else:
        print(k, v[0].cpu().numpy().tolist())

# save pixel_values in picture. shape: [4096, 1176]
# Qwen2-VL: pixel_values is flat [total_patches, patch_dim]
# patch_dim = 1176 = 3 (channels) * 2 (temporal_patch) * 14 * 14 (patch_size)
# image_grid_thw = [batch, 3] tells (temporal, h_patches, w_patches) per sample

pixel_values = model_inputs["pixel_values"].cpu().numpy()  # [4096, 1176]
image_grid_thw = model_inputs["image_grid_thw"].cpu().numpy()  # [batch, 3]

# Reconstruct first image
t, h, w = image_grid_thw[0]  # [1, 16, 32]
num_patches = t * h * w  # 512 patches for first image
patches = pixel_values[:num_patches]  # [512, 1176]

# Qwen2-VL encoding: patches.reshape then transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
# Final flatten_patches: [grid_t*grid_h*grid_w, C*temporal_patch*patch_h*patch_w]
# So 1176 = C(3) * temporal(2) * 14 * 14, in that order
patch_size = 14
temporal_patch_size = 2
merge_size = 2  # Qwen2-VL default

# After transpose(0,3,6,4,7,2,1,5,8), patches become:
# (grid_t, grid_h//merge, grid_w//merge, merge, merge, C, temporal, patch_h, patch_w)
# Then flatten last 4 dims: C * temporal * patch_h * patch_w = 1176
# First 5 dims flattened to grid_t*grid_h*grid_w = 512

# Reverse: reshape [512, 1176] to [grid_t, grid_h//merge, grid_w//merge, merge, merge, C, temporal, ph, pw]
grid_h_merged = h // merge_size  # 8
grid_w_merged = w // merge_size  # 16

patches = patches.reshape(
    t, grid_h_merged, grid_w_merged, merge_size, merge_size, 3, temporal_patch_size, patch_size, patch_size
)
# [1, 8, 16, 2, 2, 3, 2, 14, 14]

# Inverse of transpose(0, 3, 6, 4, 7, 2, 1, 5, 8) is transpose to get back original order
# Original: (grid_t, temp, C, gh_m, merge, ph, gw_m, merge, pw)
# After transpose: (0, 3, 6, 4, 7, 2, 1, 5, 8) = (gt, gh_m, gw_m, m, m, C, tp, ph, pw)
# Inverse: put dim i at position where original[j] = i
# Original positions: 0->0, 1->6, 2->5, 3->1, 4->3, 5->7, 6->2, 7->4, 8->8
patches = patches.transpose(0, 6, 5, 1, 3, 7, 2, 4, 8)
# Now: (grid_t, temporal, C, grid_h_merged, merge, patch_h, grid_w_merged, merge, patch_w)
# [1, 2, 3, 8, 2, 14, 16, 2, 14]

# Merge spatial dims back
img_h = h * patch_size  # 16 * 14 = 224
img_w = w * patch_size  # 32 * 14 = 448
image = patches.reshape(t * temporal_patch_size, 3, img_h, img_w)
# [2, 3, 224, 448]

# Convert to HWC
image = image.transpose(0, 2, 3, 1)  # [2, 224, 448, 3]

# Take first temporal frame and denormalize (ImageNet stats)
img = image[0]
mean = np.array([0.48145466, 0.4578275, 0.40821073])
std = np.array([0.26862954, 0.26130258, 0.27577711])
img = img * std + mean
img = np.clip(img * 255, 0, 255).astype(np.uint8)

cv2.imwrite("reconstructed_pixel_values-vla0.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print(f"Saved reconstructed image: {img.shape}")

# for original vla0
labels = pickle.load(open("./data_pickles/labels.pkl", "rb"))
print(labels[0].cpu().numpy().tolist())
breakpoint()

"""
input_ids [151644, 8948, 198, 2082, 55856, 279, 1946, 2168, 323, 7023, 12305, 6168, 369, 279, 1790, 220, 23, 259, 76632, 13, 8886, 1917, 702, 220, 22, 15336, 13, 9258, 264, 3175, 8500, 315, 220, 20, 21, 25780, 320, 15, 12, 16, 15, 15, 15, 1817, 701, 14064, 279, 220, 23, 259, 76632, 94559, 13, 39565, 1172, 3550, 18663, 5109, 13, 12064, 770, 13, 151645, 198, 151644, 872, 198, 151652, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151653, 2508, 279, 6149, 26482, 315, 279, 21921, 151645, 198, 151644, 77091, 198, 22, 21, 22, 220, 19, 24, 22, 220, 19, 24, 24, 220, 16, 22, 16, 220, 22, 23, 22, 220, 21, 17, 18, 220, 15, 220, 22, 21, 15, 220, 19, 24, 19, 220, 20, 15, 15, 220, 16, 20, 19, 220, 22, 22, 24, 220, 21, 17, 23, 220, 15, 220, 22, 19, 19, 220, 19, 24, 15, 220, 20, 15, 15, 220, 16, 18, 16, 220, 22, 20, 22, 220, 21, 18, 24, 220, 15, 220, 22, 17, 19, 220, 19, 23, 19, 220, 20, 15, 15, 220, 16, 16, 22, 220, 22, 17, 24, 220, 21, 20, 22, 220, 15, 220, 22, 15, 16, 220, 19, 21, 18, 220, 20, 15, 15, 220, 16, 17, 15, 220, 21, 22, 16, 220, 21, 22, 17, 220, 15, 220, 21, 22, 15, 220, 19, 17, 19, 220, 19, 22, 21, 220, 16, 20, 15, 220, 21, 15, 19, 220, 21, 24, 18, 220, 15, 220, 21, 17, 19, 220, 18, 22, 21, 220, 19, 20, 16, 220, 16, 23, 22, 220, 20, 18, 24, 220, 22, 15, 16, 220, 15, 220, 20, 24, 19, 220, 18, 18, 22, 220, 19, 19, 15, 220, 17, 17, 15, 220, 20, 15, 15, 220, 22, 15, 22, 220, 15, 151645, 198, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643, 151643]
attention_mask [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
pixel_values torch.Size([4096, 1176])
image_grid_thw [1, 16, 32]
Saved reconstructed image: (224, 448, 3)
[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 22, 21, 22, 220, 19, 24, 22, 220, 19, 24, 24, 220, 16, 22, 16, 220, 22, 23, 22, 220, 21, 17, 18, 220, 15, 220, 22, 21, 15, 220, 19, 24, 19, 220, 20, 15, 15, 220, 16, 20, 19, 220, 22, 22, 24, 220, 21, 17, 23, 220, 15, 220, 22, 19, 19, 220, 19, 24, 15, 220, 20, 15, 15, 220, 16, 18, 16, 220, 22, 20, 22, 220, 21, 18, 24, 220, 15, 220, 22, 17, 19, 220, 19, 23, 19, 220, 20, 15, 15, 220, 16, 16, 22, 220, 22, 17, 24, 220, 21, 20, 22, 220, 15, 220, 22, 15, 16, 220, 19, 21, 18, 220, 20, 15, 15, 220, 16, 17, 15, 220, 21, 22, 16, 220, 21, 22, 17, 220, 15, 220, 21, 22, 15, 220, 19, 17, 19, 220, 19, 22, 21, 220, 16, 20, 15, 220, 21, 15, 19, 220, 21, 24, 18, 220, 15, 220, 21, 17, 19, 220, 18, 22, 21, 220, 19, 20, 16, 220, 16, 23, 22, 220, 20, 18, 24, 220, 22, 15, 16, 220, 15, 220, 20, 24, 19, 220, 18, 18, 22, 220, 19, 19, 15, 220, 17, 17, 15, 220, 20, 15, 15, 220, 22, 15, 22, 220, 15, 151645, 198, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
"""


import pickle
import time
import timeit

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

print(processor.decode(model_inputs["input_ids"][0], skip_special_tokens=False))

"""
<|im_start|>system
Analyze the input image and predict robot actions for the next 8 timesteps. Each action has 7 dimensions. Output a single sequence of 56 integers (0-1000 each), representing the 8 timesteps sequentially. Provide only space separated numbers. Nothing else.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>open the middle drawer of the cabinet<|im_end|>
<|im_start|>assistant
767 497 499 171 787 623 0 760 494 500 154 779 628 0 744 490 500 131 757 639 0 724 484 500 117 729 657 0 701 463 500 120 671 672 0 670 424 476 150 604 693 0 624 376 451 187 539 701 0 594 337 440 220 500 707 0<|im_end|>
<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>
"""
