import os
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import numpy as np
import cv2

class ICDARDataset(Dataset):
    def __init__(self, datadir, labelsdir):
        if not os.path.isdir(datadir):
            raise Exception('[ERROR] {} is not a directory'.format(datadir))
        if not os.path.isdir(labelsdir):
            raise Exception('[ERROR] {} is not a directory'.format(labelsdir))

        self.datadir = datadir
        self.img_names = os.listdir(self.datadir)
        self.labelsdir = labelsdir

    def __len__(self):
        return len(self.img_names)

    # rescale coors
    def box_transfer_v2(self,coor_lists,rescale_fac = 1.0):
        gtboxes = []
        for coor_list in coor_lists:
            coors_x = [int(coor_list[2 * i]) for i in range(4)]
            coors_y = [int(coor_list[2 * i + 1]) for i in range(4)]
            xmin = min(coors_x)
            xmax = max(coors_x)
            ymin = min(coors_y)
            ymax = max(coors_y)
            if rescale_fac > 1.0:
                xmin = int(xmin / rescale_fac)
                xmax = int(xmax / rescale_fac)
                ymin = int(ymin / rescale_fac)
                ymax = int(ymax / rescale_fac)
            prev = xmin
            for i in range(xmin // 16 + 1, xmax // 16 + 1):
                next = 16*i-0.5
                gtboxes.append((prev, ymin, next, ymax))
                prev = next
            gtboxes.append((prev, ymin, xmax, ymax))
        return np.array(gtboxes)

    def parse_gtfile(self, gt_path, rescale_fac = 1.0):
        coor_lists = list()
        with open(gt_path,encoding='UTF-8') as f:
            content = f.readlines()
            for line in content:
                coor_list = line.split(',')[:8]
                if len(coor_list)==8:
                    coor_lists.append(coor_list)
        return self.box_transfer_v2(coor_lists,rescale_fac)

    def __getitem__(self, index: int):
        img_name = self.img_names[index]
        img_path = os.path.join(self.datadir, img_name)

        img = cv2.imread(img_path)

        # open default image
        if img is None:
            print(img_path)
            with open('error_imgs.txt','a') as f:
                f.write('{}\n'.format(img_path))
            img_name = 'img_2647.jpg'
            img_path = os.path.join(self.datadir, img_name)
            img = cv2.imread(img_path)

        h, w, c = img.shape
        rescale_fac = max(h, w) / 1600
        if rescale_fac > 1.0:
            h = int(h/rescale_fac)
            w = int(w/rescale_fac)
            img = cv2.resize(img,(w,h))

        gt_path = os.path.join(self.labelsdir, 'gt_'+img_name.split('.')[0]+'.txt')
        gtbox = self.parse_gtfile(gt_path,rescale_fac)



