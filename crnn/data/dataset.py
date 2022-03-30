import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


def data_transform(img):
    h, w = img.shape[:2]
    trans = transforms.Compose([
        transforms.RandomRotation(20, expand=True, center = [h / 2, w / 2]),
        transforms.ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1,hue = 0.1),
        transforms.Resize([h, w]),
        # transforms.ToTensor()
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

if __name__ == '__main__':
    img = torch.zeros([45, 45])

