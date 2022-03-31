from re import T
from data import dataset
import torch
from torch.utils import DataLoader
import utils
import keys
from torch.nn import CTCLoss
import crnn
import os

import config

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dataset = dataset.CRNNDataset(config.train_infofile)
    if not config.random_sample:
        sampler = dataset.randomSequtialSampler(train_dataset, config.batch_size)
    else:
        sampler = None
    train_loader = torch.utils.DataLoader(
        train_dataset, batch_size = config.batch_size,
        suffle=True, sampler=sampler,
        num_workers=int(config.workers),
        collate_fn=dataset.alignCollate(imgH=config.imgH, imgW=config.imgW, keep_ratio=config.keep_ratio)
    )
    test_dataset = dataset.CRNNDataset(info_filename=config.val_infofile, transform=dataset.resizeNormalize((config.imgW, config.imgH), is_test=True))
    '''
    test_loader = torch.utils.DataLoader(

    )'''

    converter = utils.strLabelConverter(keys.alphabet)
    
    criterion = CTCLoss(reduction = 'sum', zero_infinity=True)

    best_acc = 0.9

    model = crnn.CRNN(config.imgH, config.nc, config.nclass, config.nh)

    if config.pretrained_model!='' and os.path.exists(config.pretrained_model):
        print('loading pretrained model from %s' % config.pretrained_model)
        model.load_state_dict(torch.load(config.pretrained_model))
    else:
        model.apply(weights_init)

    optimizer = torch.optim.Adam(model.parameters())

    



    
