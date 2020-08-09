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

LR = 0.001
BATCH_SIZE = 6
TEST_BATCH_SIZE = 4
LR_STEP = [6, 10]
LR_FACTOR = 0.1
# nThreads = 6 # optional
BEGIN_EPOCH = 0
END_EPOCH = 12

snapshot = 2


base_path = './data'
train_img_path = os.path.join(base_path, 'jpeg/train/')
test_img_path = os.path.join(base_path, 'jpeg/test/')

train = pd.read_csv(os.path.join(base_path, 'train.csv'))
test = pd.read_csv(os.path.join(base_path, 'test.csv'))
sample = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))

x_train = train['image_name'].tolist()
y_train = train['target'].tolist()
x_test = sample['image_name'].tolist()
y_test = sample['target'].tolist()

class MyDataset(Dataset):
    def __init__(self, x_list, y_list, imgpath):
        self.x_list = x_list
        self.y_list = y_list
        self.imgpath = imgpath
        self.augmentation = transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(300),
            transforms.RandomAffine(degrees=30, translate=(0.2,0.2)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
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
        return image, label

class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.005,
                 reduction='mean',
                 lb_ignore=255,
                 ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        label[[a, torch.arange(label.size(1)), *b]] = 0

        if self.reduction == 'mean':
            loss = -torch.sum(torch.sum(logs*label, dim=1)) / n_valid
        elif self.reduction == 'none':
            loss = -torch.sum(logs*label, dim=1)
        return loss

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

def train(train_loader, m, criterion, optimizer):
    m.train()
    train_loader = tqdm(train_loader, dynamic_ncols=True)
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    avg_loss = 0.0
    avg_acc = 0.0
    
    for i, (inps, labels) in enumerate(train_loader):
        if isinstance(inps, list):
            inps = [inp.cuda().requires_grad_() for inp in inps]
        else:
            inps = inps.cuda().requires_grad_()
        labels = labels.cuda()
        
        optimizer.zero_grad()
        outputs = m(inps)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cuda().sum()
        
        avg_loss = sum_loss / (i + 1)
        avg_acc = 100. * float(correct) / float(total)
        # TQDM
        
        train_loader.set_description(
            'loss: {loss:.8f} | acc: {acc:.4f}'.format(
                loss=avg_loss,
                acc=avg_acc)
        )
        
    train_loader.close()
    
    return avg_loss, avg_acc

def test(m, epoch, batch_size):
    if torch.cuda.device_count() > 1:
        m = nn.DataParallel(m)
    m = m.cuda()

    test_dataset = MyDataset_val(x_test, y_test, test_img_path)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False) # optional: , num_workers=2

    result = {'image_name':[],'target':[]}
    with torch.no_grad():
        m.eval()

        correct = 0
        total = 0

        for image_name, inps, labels in test_loader:
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
        res.to_csv('./result/result{}.csv'.format(epoch),index=False)

def main():
    m = EfficientNet.from_pretrained('efficientnet-b1')
    m._fc.out_features = 2
    # m.load_state_dict(torch.load('./exp/model_5.pth'))
    if torch.cuda.device_count() > 1:
        m = nn.DataParallel(m)
    m = m.cuda()
    criterion = LabelSmoothSoftmaxCE().cuda()
    optimizer = torch.optim.Adam(m.parameters(), lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=LR_STEP, gamma=LR_FACTOR)
    
    train_dataset = MyDataset(x_train, y_train, train_img_path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE * num_gpu, shuffle=True) # optional: , num_workers=nThreads
    
    for i in range(BEGIN_EPOCH, END_EPOCH):
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'############# Starting Epoch {i} | LR: {current_lr} #############')
        
        # training
        loss, acc = train(train_loader, m, criterion, optimizer)
        print('Train-{i} epoch | loss:{loss:.8f} | train acc:{acc:.4f}'.format(i=i,loss=loss, acc=acc))
        
        lr_scheduler.step()
        
        if i % snapshot == 0:
            # Save checkpoint
            torch.save(m.state_dict(), './exp/model_{i}_loss_{loss:.8f}acc_{acc:.4f}.pth'.format(i=i,loss=loss, acc=acc))
            # Prediction Test
            with torch.no_grad():
                test(m, i, TEST_BATCH_SIZE)
                print(f'##### finish testing #####')
        
    torch.save(m.state_dict(), './exp/final.pth')

if __name__ == "__main__":
    main()