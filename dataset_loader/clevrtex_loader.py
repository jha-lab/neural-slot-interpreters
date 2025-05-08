from torch.utils.data import Dataset
import h5py
import torch
import numpy as np


class CLEVrTexProgramDataSet(object):
    def __init__(self, root, phase, transform = None, max_program_len=10):

        assert phase in ['train', 'val', 'test']
        with h5py.File(root+'images.h5', 'r') as f:
            self.imgs = f[phase][:]

        phase_label = phase + '_'
        self.labels = np.load(root+f'{phase_label}labels.npy', allow_pickle=True)
        self.max_program_len = max_program_len
        self.transform = transform

    def __getitem__(self, index):

        img = self.imgs[index]
        img = torch.from_numpy(img).permute(2, 0, 1)
        if self.transform is not None:
            img = self.transform(img)
        img = img.float() 
        label  = np.array(self.labels[index])
        #convert to tensor
        label = torch.from_numpy(label).float()
        length = torch.tensor(label.shape[0])
        #pad to max_program_len
        if label.shape[0] < self.max_program_len:
            label = torch.cat([label, torch.zeros(self.max_program_len-label.shape[0], 6)], dim=0)
        return img, label, length

    def __len__(self):

        return len(self.imgs)