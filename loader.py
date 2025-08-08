import torch
import numpy as np
from monai.data import (Dataset)
import scipy.io
import os


class CustomDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.images = data['arr_0']
        self.labels = data.get('arr_1')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = torch.from_numpy(image).float().permute(2, 0, 1)

        item = {'image': image}
        if self.labels is not None:
            label = self.labels[index]
            label = torch.from_numpy(label).float().permute(2, 0, 1)
            item['label'] = label
        return item


class CustomDataset_test(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.images = data['arr_0']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        return {'image': image}


class MatDataset(Dataset):
    def __init__(self, images_dir, labels_dir=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        self.image_files = []
        for f in os.listdir(images_dir):
            if f.endswith('.mat'):
                self.image_files.append(f)

        self.image_files.sort(key=lambda x: int(x.split('.')[0]))

        if labels_dir:
            self.has_labels = True
            for img_file in self.image_files:
                label_path = os.path.join(labels_dir, img_file)
                if not os.path.exists(label_path):
                    print(f"Warning: Label file {label_path} not found")
        else:
            self.has_labels = False

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.images_dir, self.image_files[index])
        img_data = scipy.io.loadmat(img_path)
        image = img_data['data'].astype(np.float32)  # (256, 256, 3)

        image = torch.from_numpy(image).permute(2, 0, 1)

        item = {'image': image, 'filename': self.image_files[index]}

        if self.has_labels:
            label_path = os.path.join(self.labels_dir, self.image_files[index])
            if os.path.exists(label_path):
                label_data = scipy.io.loadmat(label_path)
                label = label_data['data'].astype(np.float32)  # (256, 256, 1)

                label = torch.from_numpy(label).squeeze(-1)
                item['label'] = label
            else:
                item['label'] = torch.zeros((256, 256), dtype=torch.float32)

        return item
