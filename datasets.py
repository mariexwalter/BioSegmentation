"""
From:
https://www.kaggle.com/code/duuuscha/train-u-net
"""
import random
from glob import glob
from os.path import join, basename, splitext

import cv2
import torch
from skimage import io
from warnings import warn


class FromFolderToRam(torch.utils.data.Dataset):

    def __init__(self, augmentation, img_folder: str, img_ext: str, lab_folder: str, lab_ext: str):
        self.augment = augmentation
        self._load_data_to_ram(img_folder, img_ext, lab_folder, lab_ext)

    def _load_data_to_ram(self, img_folder: str, img_ext: str, lab_folder: str, lab_ext: str):
        def to_id(x):
            return splitext(basename(x))[0]

        img_files_ = glob(join(img_folder, f"*{img_ext}"))
        assert img_files_, ('No files found for ' +
                            join(img_folder, f"*{img_ext}"))

        lab_files = glob(join(lab_folder, f"*{lab_ext}"))
        assert lab_files, ('No files found for ' +
                           join(img_folder, f"*{img_ext}"))
        lab_files = {to_id(x): x for x in lab_files}

        self.data = []

        for img_file in img_files_:
            img_id = to_id(img_file)
            if img_id not in lab_files.keys():
                warn(f"No label found for {img_file}")
                continue

            lab_file = lab_files.pop(img_id)

            self.data.append([
                self.load_image(img_file),
                self.load_label(lab_file)
            ])

        if lab_files:
            warn(f"No images found for labels: {lab_files}")

    def load_image(self, path):
        raise NotImplementedError()

    def load_label(self, path):
        raise NotImplementedError()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image, label = self.augment(image, label)
        return image, label


class BreastCancer(FromFolderToRam):
    def load_image(self, path):
        return get_tiff_image(path)

    def load_label(self, path):
        label = get_tiff_image(path)
        # RGB image -> to binary image
        return (label.mean(-1) > 0.).astype('int32')

class AortaTissue(FromFolderToRam):
    pass

def get_tiff_image(path, normalized=True, resize=(512, 512)):
    image = io.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    if normalized:
        return image / 255
    return image
