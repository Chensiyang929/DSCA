import torch
import torch.utils.data as data
import scipy.io as sio
import numpy as np
import h5py
import torchvision

import torch
import torch.utils.data as data
import numpy as np
import h5py
import torchvision


def sensor_norm(file_path):
    """PanCollection radiometric range: GF2 is 10-bit, others are 11-bit."""
    return 1023.0 if "gf2" in str(file_path).lower() else 2047.0


class Dataset_SDE(data.Dataset):
    def __init__(self, file_path, name):
        super(Dataset_SDE, self).__init__()
        dataset = h5py.File(file_path, 'r')

        norm = sensor_norm(file_path)
        ms = np.array(dataset['ms'][name], dtype=np.float32) / norm
        lms = np.array(dataset['lms'][name], dtype=np.float32) / norm
        pan = np.array(dataset['pan'][name], dtype=np.float32) / norm

        ms = torch.from_numpy(ms).float()
        lms = torch.from_numpy(lms).float()
        pan = torch.from_numpy(pan).float()

        MS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        self.ms_crops = MS_crop(ms)
        LMS_crop = torchvision.transforms.TenCrop(lms.shape[1] / 2)
        self.lms_crops = LMS_crop(lms)
        PAN_crop = torchvision.transforms.TenCrop(pan.shape[1] / 2)
        self.pan_crops = PAN_crop(pan)

    def __getitem__(self, item):
        return self.ms_crops[item], self.lms_crops[item], self.pan_crops[item]

    def __len__(self):
        return len(self.ms_crops)
    
class Dataset(data.Dataset):
    def __init__(self, file_path, name):
        '''
        name:图片的名字，范围是[0,19]
        '''
        
        
        super(Dataset, self).__init__()

        dataset = h5py.File(file_path, 'r')

        norm = sensor_norm(file_path)
        ms = np.array(dataset['ms'][name], dtype=np.float32) / norm
        lms = np.array(dataset['lms'][name], dtype=np.float32) / norm
        pan = np.array(dataset['pan'][name], dtype=np.float32) / norm

        ms = torch.from_numpy(ms).float()
        lms = torch.from_numpy(lms).float()
        pan = torch.from_numpy(pan).float()

        # MS_crop = torchvision.transforms.TenCrop(ms.shape[1] / 2)
        # self.ms_crops = MS_crop(ms)
        self.ms_crops = ms
        self.lms_crops = lms
        self.pan_crops = pan



    def __getitem__(self, item):
        return self.ms_crops, self.lms_crops, self.pan_crops

    def __len__(self):
        return 1

class ReducedDataset(data.Dataset):
    """预构造的 reduced 仿真数据
    文件由 prepare_reduced_data.py 生成，包含 lms/pan（Wald 降采样后）与 gt（原始低分辨率 ms）。
    """
    def __init__(self, file_path, name, patch_size=0):
        super(ReducedDataset, self).__init__()

        dataset = h5py.File(file_path, 'r')

        norm = sensor_norm(file_path)
        lms = np.array(dataset['lms'][name], dtype=np.float32) / norm
        pan = np.array(dataset['pan'][name], dtype=np.float32) / norm
        gt = np.array(dataset['gt'][name], dtype=np.float32) / norm

        lms = torch.from_numpy(lms).float()
        pan = torch.from_numpy(pan).float()
        gt = torch.from_numpy(gt).float()

        self.samples = []
        self.sample_regions = []
        self.last_index = None
        patch_size = int(patch_size or 0)
        if patch_size > 0:
            height, width = gt.shape[-2:]
            if height % patch_size or width % patch_size:
                raise ValueError(
                    f"reduced image {height}x{width} is not divisible by patch size {patch_size}"
                )
            for top in range(0, height, patch_size):
                for left in range(0, width, patch_size):
                    region = (..., slice(top, top + patch_size), slice(left, left + patch_size))
                    self.samples.append((lms[region], pan[region], gt[region]))
                    self.sample_regions.append((top, left, patch_size, patch_size))
        else:
            self.samples.append((lms, pan, gt))
            self.sample_regions.append((0, 0, gt.shape[-2], gt.shape[-1]))

    def __getitem__(self, item):
        self.last_index = int(item)
        return self.samples[item]

    def __len__(self):
        return len(self.samples)
