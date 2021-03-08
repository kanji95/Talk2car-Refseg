import os
import numpy as np
from PIL import Image


root_dir = "/ssd_scratch/cvit/kanishk"
mask_dir = os.path.join(root_dir, "val_masks_new")

mask_files = os.listdir(mask_dir)
print(len(mask_files))

correct = 0
total = 0

for mask_file in mask_files:
    if "test" not in mask_file:
        continue

    mask_path = os.path.join(mask_dir, mask_file)

    mask_img = Image.open(mask_path).convert('L')
    mask_img = np.array(mask_img)

    h, w = mask_img.shape

    total += 1
    if mask_img[w//2, h//2] > 0:
        correct += 1

print(total, correct)
print(correct/total)
