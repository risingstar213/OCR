import torch
from torchvision import transforms
from torch.utils.data import sampler
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np

# def pepper

class AddPepperNoise(object):
    def __init__(self, rate = 0.02):
        self.rate = rate
    def __call__(self, img):
        w, h = img.size
        num_noise = int(w * h * self.rate)
        for k in range(num_noise):
            i = random.randint(0, h - 1)
            j = random.randint(0, w - 1)
            img.putpixel((j, i), int(np.random.random() * 255))
        return img


def ConverseColor(img):
    w, h = img.size
    for i in range(w):
        for j in range(h):
            v = np.array(img.getpixel((i, j)))
            img.putpixel((i, j), tuple(255 - v))
    
    return img

def data_transform(img):
    w, h = img.size[:2]
    w -= random.randint(0, 10) * 10
    trans = transforms.Compose([
        transforms.ColorJitter(brightness = 0.15, contrast = 0.15, saturation = 0.15,hue = 0.15),
        transforms.RandomApply([transforms.Resize([h, w])], p = 0.1),
        transforms.Lambda(lambda img : AddPepperNoise()(img)),
        transforms.RandomApply([transforms.Lambda(lambda img : ConverseColor(img))], p = 0.2)
    ])

    return trans(img)




class CRNNDataset(Dataset):
    def __init__(self, info_filename, train = True, transform = data_transform , 
                    target_transform = None, remove_blank = False):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.info_filename - info_filename
        if isinstance(self.info_filename,str):
            self.info_filename = [self.info_filename]
        
        self.train = train
        self.files = []
        self.labels = []
        for info_name in self.info_filename:
            with open(info_name) as f:
                content = f.readlines()
                for line in content:
                    if '\t' in line:
                        if len(line.split('\t')) != 2:
                            print(line)
                        fname, label = line.split('\t')

                    else:
                        fname, label = line.split('g:')
                        fname += 'g'
                    
                    if remove_blank:
                        label = label.strip()
                    else:
                        label = ' ' + label.strip() + ' '
                    self.files.append(fname)
                    self.labels.append(label)
        return

    def name(self):
        return 'CRNNDataset'

    def __getitem__(self, index: int):
        img = Image.open(self.files[index])
        if self.transform is not None:
            img = self.transform(img)
        img = img.convert('L')

        label = self.labels[index]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return (img, label)


class randomSequtialSampler(sampler.Sampler):
    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples

class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.LANCZOS,is_test=False):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.is_test = is_test

    def __call__(self, img):
        w,h = self.size
        w0 = img.size[0]
        h0 = img.size[1]
        if w<=(w0/h0*h):
            img = img.resize(self.size, self.interpolation)
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
        else:
            w_real = int(w0/h0*h)
            img = img.resize((w_real,h), self.interpolation)
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
            start = random.randint(0,w-w_real-1)
            if self.is_test:
                start = 5
                w+=10
            tmp = torch.zeros([img.shape[0], h, w])+0.5
            tmp[:,:,start:start+w_real] = img
            img = tmp
        return img



class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels

if __name__ == '__main__':
    img = Image.open('C:/Users/star/Desktop/cv/ocr/crnn/dataset/1.jpg')
    print(img.size)
    img = resizeNormalize((32, 100))(img)
    print(img.shape)
    # img.save('C:/Users/star/Desktop/cv/ocr/crnn/dataset/1.1.jpg')
    
    
    

