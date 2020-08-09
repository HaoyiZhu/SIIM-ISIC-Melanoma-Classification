import os
import time
import math
import random

import matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL.Image import open

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from efficientnet_pytorch import EfficientNet

from tqdm import tqdm

num_gpu = torch.cuda.device_count()


BATCH_SIZE = 10

filename = 'model_10_loss_37.07799096acc_98.2370.pth'

base_path = './data'
test_img_path = os.path.join(base_path, 'jpeg/test/')

test = pd.read_csv(os.path.join(base_path, 'test.csv'))
sample = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))

x_test = sample['image_name'].tolist()
y_test = sample['target'].tolist()

class MyDataset_val(Dataset):
    def __init__(self, x_list, y_list, imgpath):
        self.x_list = x_list
        self.y_list = y_list
        self.imgpath = imgpath
        self.augmentation = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(300),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])
        assert len(self.x_list) == len(self.y_list)
        
    def __len__(self):
        return len(self.y_list)
    
    def __getitem__(self, idx):
        path = os.path.join(self.imgpath, (self.x_list[idx]+'.jpg'))
        image = open(path)
        image = self.augmentation(image)
        label = self.y_list[idx]
        return self.x_list[idx], image, label

def test(m, filename, batch_size):
    if torch.cuda.device_count() > 1:
        m = nn.DataParallel(m)
    m = m.cuda()

    test_dataset = MyDataset_val(x_test, y_test, test_img_path)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False) # optional: , num_workers=2

    result = {'image_name':[],'target':[]}
    with torch.no_grad():
        m.eval()

        for image_name, inps, labels in tqdm(test_loader):
            if isinstance(inps, list):
                inps = [inp.cuda() for inp in inps]
            else:
                inps = inps.cuda()
            labels = labels.cuda()
            outputs = m(inps)
            
            column1 = torch.exp(outputs.data[:, 0]) 
            column2 = torch.exp(outputs.data[:, 1])
            predicted = column2 / (column1 + column2)
            
            result['image_name'].extend(image_name)
            result['target'].extend(predicted.tolist())

        res = pd.DataFrame(result)
        res.to_csv("./result/result-{}.csv".format(filename),index=False)

def main():
    m = EfficientNet.from_name('efficientnet-b1')
    m._fc.out_features = 2
    m.load_state_dict(torch.load('./exp/{}'.format(filename)))
    if torch.cuda.device_count() > 1:
        m = nn.DataParallel(m)
    m = m.cuda()
    
    test(m, filename, BATCH_SIZE)
    print(f'##### finish testing #####')

if __name__ == "__main__":
    main()