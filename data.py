import os

import kornia
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from kornia.augmentation.container import AugmentationSequential
from torch import nn
from torch.utils.data import DataLoader, Dataset

import utils


img_augment = AugmentationSequential(
    kornia.augmentation.RandomResizedCrop((224, 224), (0.8, 1), p=0.3),
    kornia.augmentation.Resize((224, 224)),
    kornia.augmentation.RandomBrightness(brightness=(0.8, 1.2), clip_output=True, p=0.2),
    kornia.augmentation.RandomContrast(contrast=(0.8, 1.2), clip_output=True, p=0.2),
    kornia.augmentation.RandomGamma((0.8, 1.2), (1.0, 1.3), p=0.2),
    kornia.augmentation.RandomSaturation((0.8, 1.2), p=0.2),
    kornia.augmentation.RandomHue((-0.1, 0.1), p=0.2),
    kornia.augmentation.RandomSharpness((0.8, 1.2), p=0.2),
    kornia.augmentation.RandomGrayscale(p=0.2),
    data_keys=["input"],
)


class NSDDataset(Dataset):
    def __init__(self, root_dir, extensions=None, pool_num=15724, pool_type="max", length=None):
        self.root_dir = root_dir
        self.extensions = extensions if extensions else []
        self.pool_num = pool_num
        self.pool_type = pool_type
        self.samples = self._load_samples()
        self.samples_keys = sorted(self.samples.keys())
        if length is None:
            self.length = len(self.samples_keys)
        else:
            if length == 0:
                raise ValueError("length must be non-zero")
            self.length = length
            if 0 < length <= len(self.samples_keys):
                self.samples_keys = self.samples_keys[:length]
            elif length < 0:
                self.samples_keys = self.samples_keys[length:]

    def _load_samples(self):
        samples = {}
        for file_name in os.listdir(self.root_dir):
            sample_id, ext = file_name.split(".", maxsplit=1)
            if ext not in self.extensions:
                continue
            file_path = os.path.join(self.root_dir, file_name)
            samples.setdefault(sample_id, {"subj": file_path})[ext] = file_path
        return samples

    @staticmethod
    def _load_image(image_path):
        image = Image.open(image_path).convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image.transpose(2, 0, 1))

    @staticmethod
    def _load_npy(npy_path):
        return torch.from_numpy(np.load(npy_path))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx % len(self.samples_keys)
        sample_key = self.samples_keys[idx]
        sample = self.samples[sample_key]
        items = []
        for ext in self.extensions:
            if ext == "jpg":
                items.append(self._load_image(sample[ext]))
            elif ext == "nsdgeneral.npy":
                voxel = self._load_npy(sample[ext])
                items.append(pool_voxels(voxel, self.pool_num, self.pool_type))
            elif ext == "coco73k.npy":
                items.append(self._load_npy(sample[ext]))
            elif ext == "subj":
                items.append(int(sample[ext].split("/")[-2].split("subj")[-1]))
        return items


def pool_voxels(voxels, pool_num, pool_type):
    voxels = voxels.float()
    if pool_num is None or pool_type is None:
        return voxels
    if pool_type == "avg":
        return nn.AdaptiveAvgPool1d(pool_num)(voxels)
    if pool_type == "max":
        return nn.AdaptiveMaxPool1d(pool_num)(voxels)
    if pool_type == "resize":
        return F.interpolate(voxels.unsqueeze(1), size=pool_num, mode="linear", align_corners=False).squeeze(1)
    raise ValueError(f"Unsupported pool_type: {pool_type}")


def get_dataloader(
    root_dir,
    batch_size,
    num_workers=0,
    seed=42,
    is_shuffle=False,
    extensions=("nsdgeneral.npy", "jpg", "coco73k.npy", "subj"),
    pool_type="max",
    pool_num=15724,
    length=None,
    drop_last=True,
):
    utils.seed_everything(seed)
    dataset = NSDDataset(
        root_dir=root_dir,
        extensions=list(extensions),
        pool_num=pool_num,
        pool_type=pool_type,
        length=length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=is_shuffle,
        drop_last=drop_last,
    )


def get_subject_dls(subject, data_path, batch_size, val_batch_size, num_workers, pool_type, pool_num, length, seed):
    train_path = f"{data_path}/webdataset_avg_split/train/subj0{subject}"
    val_path = f"{data_path}/webdataset_avg_split/val/subj0{subject}"
    train_dl = get_dataloader(
        train_path,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        pool_type=pool_type,
        pool_num=pool_num,
        length=length,
        drop_last=True,
    )
    val_dl = get_dataloader(
        val_path,
        batch_size=val_batch_size,
        num_workers=num_workers,
        seed=seed,
        pool_type=pool_type,
        pool_num=pool_num,
        drop_last=True,
    )
    print(train_path, "\n", val_path)
    print("number of train data:", len(train_dl.dataset))
    print("number of val data:", len(val_dl.dataset))
    return train_dl, val_dl


def get_subject_test_dl(subject, data_path, batch_size=1, num_workers=0, pool_type="max", pool_num=15724, seed=42):
    test_path = f"{data_path}/webdataset_avg_split/test/subj0{subject}"
    return get_dataloader(
        test_path,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        is_shuffle=False,
        pool_type=pool_type,
        pool_num=pool_num,
        drop_last=False,
    )
