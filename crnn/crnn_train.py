
from data import dataset
import torch
from torch.utils import DataLoader
from torch.autograd import Variable
import utils
import keys
from torch.nn import CTCLoss
import crnn
import os


import config

lr = 0.0003
betas = (0.5, 0.999)
niter = 100
device = None

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_on_batch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    image = cpu_images.to(device)

    text, length = converter.encode(cpu_texts)

    preds = net(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds.log_softmax(2).cpu(), text, preds_size, length)
    if torch.isnan(cost):
        print(batch_size,cpu_texts)
    else:
        net.zero_grad()
        cost.backward()
        optimizer.step()
    return cost



if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = crnn.CRNN(config.imgH, config.nc, config.nclass, config.nh)
    
    checkpoints_weight = config.pretrained_weights
    if os.path.exists(checkpoints_weight):
        print('using pretrained weight: {}'.format(checkpoints_weight))
        cc = torch.load(checkpoints_weight, map_location=device)
        model.load_state_dict(cc['model_state_dict'])
        resume_epoch = cc['epoch']
    else:
        model.apply(weights_init)

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
   

    converter = utils.strLabelConverter(keys.alphabet)
    
    criterion = CTCLoss(reduction = 'sum', zero_infinity=True)

    best_acc = 0.9


    if config.pretrained_model!='' and os.path.exists(config.pretrained_model):
        print('loading pretrained model from %s' % config.pretrained_model)
        model.load_state_dict(torch.load(config.pretrained_model))
    else:
        model.apply(weights_init)

    loss_avg = utils.averager()
    optimizer = torch.optim.Adam(model.parameters(), betas=betas)

    best_loss = 1e18
    best_model = None
    for epoch in range(niter):
        loss_avg.reset()
        print(f'epoch {epoch}...')
        train_iter = iter(train_loader)
        i = 0
        n_batch = len(train_loader)
        while i < len(train_loader):
            for p in model.parameters():
                p.requires_grad = True
            crnn.train()
            cost = train_on_batch(crnn, criterion, optimizer)
            print('epoch: {} iter: {}/{} Train loss: {:.3f}'.format(epoch, i, n_batch, cost.item()))
            loss_avg.add(cost)
            loss_avg.add(cost)
            i += 1
        print('Train loss: %f' % (loss_avg.val()))
        if loss_avg.val() < best_loss:
            best_loss = loss_avg.val()
            best_model = model
        
        state = {'epoch': epoch, 'model_state_dict': best_model.state_dict()}

        check_path = check_path = os.path.join(config.checkpoints_dir,
                              f'v3_crnn_ep{epoch:02d}.pth')
        try:
            torch.save(state, check_path)
        except BaseException as e:
            print(e)
            print('fail to save to {}'.format(check_path))
            print('saving to {}'.format(check_path))






    
