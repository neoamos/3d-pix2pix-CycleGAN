import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
#from data.image_folder import make_dataset
import pickle
import numpy as np


class NoduleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.pkl_file = os.path.join(opt.dataroot, "crops.pkl")
        self.heatmaps_dir = os.path.join(opt.dataroot, "heatmaps/real")
        self.scans_dir = os.path.join(opt.dataroot, "scans_processed")
        self.samples = pickle.load(open(self.pkl_file, 'rb'))
        self.samples = [ j for i in self.samples for j in i]

        random.shuffle(self.samples)
        
        self.scans = {}
        self.heatmaps = {}


    def __getitem__(self, index):
        #returns samples of dimension [channels, z, x, y]

        sample = self.samples[index]

        #load scan and heatmap if it hasnt already been loaded
        if not self.scans.has_key(sample['suid']):
            print(sample['suid'])
            self.scans[sample['suid']] = np.load(os.path.join(self.scans_dir, sample['suid'] + ".npz"))['a']
            self.heatmaps[sample['suid']] = np.load(os.path.join(self.heatmaps_dir, sample['suid'] + ".npz"))['a']
        scan = self.scans[sample['suid']]
        heatmap = self.heatmaps[sample['suid']]

        #crop
        b = sample['bounds']
        scan_crop = scan[b[0]:b[3], b[1]:b[4], b[2]:b[5]]
        heatmap_crop = heatmap[b[0]:b[3], b[1]:b[4], b[2]:b[5]]

        #convert to torch tensors with dimension [channel, z, x, y]
        scan_crop = torch.from_numpy(scan_crop[None, :])
        heatmap_crop = torch.from_numpy(heatmap_crop[None, :])
        return {
                'A' : scan_crop,
                'B' : heatmap_crop
                }

    def __len__(self):
        return len(self.samples)

    def name(self):
        return 'NodulesDataset'

if __name__ == '__main__':
    #test
    n = NoduleDataset()
    n.initialize("datasets/nodules")
    print(len(n))
    print(n[0])
    print(n[0]['A'].size())
