import numpy as np
import os
import json
from collections import Iterable

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable

from PIL import Image
from .vocabulary import Vocabulary
from .math import jaccard


class ResizeAnnotation:
    """Resize the largest of the sides of the annotation to a given size"""

    def __init__(self, size):
        if not isinstance(size, (int, Iterable)):
            raise TypeError("Got inappropriate size arg: {}".format(size))

        self.size = size

    def __call__(self, img):
        im_h, im_w = img.shape[-2:]
        scale_h, scale_w = self.size / im_h, self.size / im_w
        resized_h = int(np.round(im_h * scale_h))
        resized_w = int(np.round(im_w * scale_w))
        out = (
            F.interpolate(
                Variable(img).unsqueeze(0).unsqueeze(0),
                size=(resized_h, resized_w),
                mode="bilinear",
                align_corners=True,
            )
            .squeeze()
            .data
        )
        return out


class Talk2Car(data.Dataset):
    def __init__(
        self,
        root,
        split,
        vocabulary="/home/kanishk/vigil/autonomous_grounding/dataloader/vocabulary.txt",
        transform=None,
        mask_transform=None,
        glove_path="",
        max_len=30,
    ):
        self.root = root
        self.split = split

        # import pdb; pdb.set_trace();
        self.annotations = []
        for split_ in ["train", "val"]:
            with open(f"/home/kanishk/vigil/autonomous_grounding/dataloader/annotation_{split_}.txt", "r") as f:
                for line in f.readlines():
                    image_id, sentence = int(line.split()[0]), line.split()[1:]
                    sentence = " ".join(sentence)
                    self.annotations.append([image_id, sentence, split_])

        self.data = {}
        with open(
            "/home/kanishk/vigil/autonomous_grounding/dataloader/talk2car_w_rpn_no_duplicates.json",
            "rb",
        ) as f:
            data = json.load(f)[self.split]
            self.data = {int(k): v for k, v in data.items()}  # Map to int
            ## self.data["val"] = {int(k): v for k, v in data["val"].items()}  # Map to int
            ## self.data["train"] = {int(k): v for k, v in data["train"].items()}

        self.img_dir = os.path.join(self.root, "imgs")
        self.mask_dir = os.path.join(self.root, "val_masks_new")

        self.transform = transform
        self.mask_transform = mask_transform

        self.vocabulary = Vocabulary(vocabulary, glove_path, max_len)

    def __len__(self):
        ## return len(self.annotations)
        return len(self.data.keys())

    def __getitem__(self, idx):
        ## idx = self.annotations[item][0]
        ## split = self.annotations[item][2]
        output = {"index": torch.LongTensor([idx])}
        ## sample = self.data[split][idx]
        sample = self.data[idx]

        # Load image
        img_path = os.path.join(self.img_dir, sample["img"])

        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        output["orig_image"] = np.array(img.resize((448, 448)))

        if self.transform is not None:
            img = self.transform(img)
        output["image"] = img

        ## sample["command"] = self.annotations[item][1]
        phrase, phrase_mask = self.vocabulary.tokenize(sample["command"])

        output["orig_phrase"] = sample["command"]
        output["phrase"] = phrase
        output["phrase_mask"] = phrase_mask

        mask_path = os.path.join(self.mask_dir, f"gt_img_ann_{self.split}_{idx}.png")

        ## gt = sample["referred_object"]
        ## x0, y0, x1, y1 = gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]
        
        mask_img = Image.open(mask_path).convert("L")
        mask_img = torch.from_numpy(np.array(mask_img))
        
        mask_img = mask_img.float() 
        mask_img = self.mask_transform(mask_img)
        mask = torch.zeros_like(mask_img)
        mask[mask_img > 0] = 1

        output["seg_mask"] = mask

        return output

    def number_of_words(self):
        # Get number of words in the vocabulary
        return self.vocabulary.number_of_words

    def convert_index_to_command_token(self, index):
        return self.data[index]["command_token"]

    def convert_command_to_text(self, command):
        # Takes value from command key and transforms it into human readable text
        return " ".join(self.vocabulary.ix2sent_drop_pad(command.numpy().tolist()))


## def main():
##     """ A simple example """
##     root = "/ssd_scratch/cvit/kanishk/imgs/"
##     split = "train"
##     dataset = Talk2Car(root, split, "./utils/vocabulary.txt", transforms.ToTensor())
## 
##     print("=> Load a sample")
##     index = np.random.choice(range(8348))
##     sample = dataset.__getitem__(index)
##     img = np.transpose(sample["image"].numpy(), (1, 2, 0))
##     command = dataset.convert_command_to_text(sample["command"])
##     print("Command in human readable text for image {%s}: %s" % (index, command))
## 
##     import matplotlib.pyplot as plt
##     import matplotlib.patches as patches
## 
##     print("=> Plot image with bounding box around referred object")
##     fig, ax = plt.subplots(1)
##     ax.imshow(img)
##     xl, yb, xr, yt = sample["gt_bbox_lbrt"].tolist()
##     w, h = xr - xl, yt - yb
##     rect = patches.Rectangle((xl, yb), w, h, fill=False, edgecolor="r")
##     ax.add_patch(rect)
##     plt.axis("off")
##     plt.show()
## 
##     print("=> Plot image with region proposals (red), gt bbox (blue)")
##     fig, ax = plt.subplots(1)
##     ax.imshow(img)
##     for i in range(sample["rpn_bbox_lbrt"].size(0)):
##         bbox = sample["rpn_bbox_lbrt"][i].tolist()
##         xl, yb, xr, yt = bbox
##         w, h = xr - xl, yt - yb
##         rect = patches.Rectangle((xl, yb), w, h, fill=False, edgecolor="r")
##         ax.add_patch(rect)
## 
##     gt_box = (sample["rpn_bbox_lbrt"][sample["rpn_gt"].item()]).tolist()
##     xl, yb, xr, yt = gt_box
##     w, h = xr - xl, yt - yb
##     rect = patches.Rectangle((xl, yb), w, h, fill=False, edgecolor="b")
##     ax.add_patch(rect)
##     plt.axis("off")
##     plt.tight_layout()
##     plt.show()
##     plt.savefig("bboxes.png", bbox_inches="tight")
## 
## 
## if __name__ == "__main__":
##     main()
