import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
import config


class RPN_REGR_Loss(nn.Module):
    def __init__(self, device, sigma = 9.0):
        super().__init__()
        self.sigma = sigma
        self.device = device
    
    def forward(self, input, target):
        try:
            cls = target[0, :, 0]
            regr = target[0, :, 1:3]
            # apply regression to positive sample
            regr_keep = (cls == 1).nonzero()[:, 0]
            regr_true = regr[regr_keep]
            regr_pred = input[0][regr_keep]
            diff = torch.abs(regr_true - regr_pred)

            less_one = (diff < 1.0 / self.sigma).float()
            loss = less_one * 0.5 * diff ** 2 * self.sigma + torch.abs(1- less_one) * (diff - 0.5/self.sigma)

            loss = torch.sum(loss, 1)
            loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)

        except Exception as e:
            print('RPN_REGR_Loss Exception:', e)
            # print(input, target)
            loss = torch.tensor(0.0)

        return loss.to(self.device)

class RPN_CLS_Loss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.L_cls = nn.CrossEntropyLoss(reduction='none')
        self.pos_neg_ratio = 3

    def forward(self, input, target):
        # 是否执行OHEM策略
        if config.OHEM:
            cls_gt = target[0][0]
            num_pos = 0
            loss_pos_sum = 0

            if len((cls_gt == 1).nonzero()) != 0:
                cls_pos = (cls_gt == 1).nonzero()[:, 0]
                gt_pos = cls_gt[cls_pos].long()
                cls_pred_pos = input[0][cls_pos]

                loss_pos = self.L_cls(cls_pred_pos.view(-1, 2), gt_pos.view(-1))
                loss_pos_sum = loss_pos.sum()
                num_pos = len(loss_pos)
            
            cls_neg = (cls_gt == 0).nonzero()[:, 0]
            gt_neg = cls_gt[cls_neg].long()
            cls_pred_neg = input[0][cls_neg]

            loss_neg = self.L_cls(cls_pred_neg.view(-1, 2), gt_neg.view(-1))
            loss_neg_topK, _ = torch.topk(loss_neg, min(len(loss_neg), config.RPN_TOTAL_NUM-num_pos))
            loss_cls = loss_pos_sum+loss_neg_topK.sum()
            loss_cls = loss_cls/config.RPN_TOTAL_NUM
            return loss_cls.to(self.device)

        else:
            y_true = target[0][0]
            cls_keep = (y_true != -1).nonzero()[:, 0]
            cls_true = y_true[cls_keep].long()
            cls_pred = input[0][cls_keep]
            loss = F.nll_loss(F.log_softmax(cls_pred, dim = -1), cls_true)
            loss = torch.clamp(torch.mean(loss), 0, 10) \
                        if loss.numel() > 0 else torch.tensor(0.0)
            
            return loss

class basic_conv(nn.Module):
    def __init__(self, chan_in, 
                        chan_out, 
                        kernel_size, 
                        stride = 1, 
                        padding = 0, 
                        dilation = 1, 
                        groups = 1, 
                        bn = True,
                        relu = True,
                        bias = True):
        super().__init__()
        self.chan_out = chan_out
        self.conv = nn.Conv2d(chan_in, chan_out, kernel_size, stride, padding, dilation, groups, bias)
        if bn:
            self.bn = nn.BatchNorm2d(chan_out, eps = 1e-5, momentum = 0.01, affine = True)
        else:
            self.bn = None

        if relu:
            self.relu = nn.ReLU(inplace = True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)

        return x

class CPTNModel(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.vgg16(pretrained=False)
        layers = list(base_model.features)[:-1]
        self.base_layers = nn.Sequential(*layers)
        self.rpn = basic_conv(512, 512, 3, 1, 1, bn = False)

        self.brnn = nn.GRU(512, 128, bidirectional=True, batch_first = True)

        self.lstm_fc = basic_conv(256, 512, 3, 1, 1, relu=True, bn=False)
        self.rpn_class = basic_conv(512, 10 * 2, 1, 1, relu = False, bn = False)
        self.rpn_regress =  basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)

    def forward(self, x):
        x = self.base_layers(x)
        # rpn 建议网络
        x = self.rpn(x) # b, c, h, w

        # 调整至channel_last
        # b, h, w, c
        x1 = x.permute(0, 2, 3, 1).contiguous()
        b = x1.size()
        # [b * h, w, c]
        x1 = x1.view(b[0] * b[1], b[2], b[3])

        x2, _ = self.brnn(x1)

        xsz = x.size()
        # [4, 20, 20, 256] [b, h, w, 256]
        x3 = x2.view(xsz[0], xsz[2], xsz[3], 256)
        # 调整至channel_first, [b, c, h, w]
        x3 = x3.permute(0, 3, 1, 2).contiguous()
        x3 = self.lstm_fc(x3)
        x = x3

        # 分类头(fcn), 是否包含文字
        cls = self.rpn_class(x)
        # 回归头(fcn)，框偏移
        regr = self.rpn_regress(x)

        # [b, h, w, c]
        cls = cls.permute(0,2,3,1).contiguous()
        regr = regr.permute(0,2,3,1).contiguous()
        # [b, h * w * 10, 2]
        cls = cls.view(cls.size(0), cls.size(1)*cls.size(2)*10, 2)
        # [b, h * w * 10, 2]
        regr = regr.view(regr.size(0), regr.size(1)*regr.size(2)*10, 2)

        return cls, regr


if __name__ == '__main__':
    a = torch.tensor([[0, 0.4, 0.6, 0.9]])
    print(a)
    print(a.nonzero())