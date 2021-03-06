import os
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import numpy as np
import cv2
import sys
from ctpn_utils import cal_rpn
from config import IMAGE_MEAN

class ICDARDataset(Dataset):
    def __init__(self, datadir, labelsdir):
        datadir = os.path.abspath(datadir)
        labelsdir = os.path.abspath(labelsdir)
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

        # clip image
        if np.random.randint(2) == 1:
            img = img[:, ::-1, :]
            newx1 = w - gtbox[:, 2] - 1
            newx2 = w - gtbox[:, 0] - 1
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2

        [cls, regr], base_anchors = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)

        m_img = img - IMAGE_MEAN

        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])

        cls = np.expand_dims(cls, axis = 0)

        # generate tensor

        # ?????????channel first
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()

        cls = torch.from_numpy(cls).float()

        regr = torch.from_numpy(regr).float()

        return m_img, cls, regr

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = ICDARDataset('C:\\Users\\star\\Desktop\\cv\\ocr\\ctpn\\train_data\\train_img', 'C:\\Users\\star\\Desktop\\cv\\ocr\\ctpn\\train_data\\train_label')
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.num_workers)
    for m_img, cls, regr in dataset:
        # print(m_img.size())
        imgs = m_img.to(device)
        clss = cls.to(device)
        regrs = regr.to(device)
        print(cls[0][0])
        print(regr[0][0])
        break