import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, chan_in, chan_hidden, chan_out):
        super().__init__()
        self.brnn = nn.LSTM(chan_in, chan_hidden, bidirectional=True)
        self.embedding = nn.Linear(chan_hidden*2, chan_out)
        return

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        s, b, c = recurrent.size()
        t_rec = recurrent.view(s * b, c)

        output = self.embedding(t_rec)
        output = output.view(s, b, -1)
        return output

class CRNN(nn.Module):
    def __init__(self, imgH, chan_in, nclass, chan_hidden, leakyRelu = False):
        super().__init__()

        # (1, 1, 32, 128)
        self.conv1 = nn.Conv2d(chan_in, 64, 3, 1, 1)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2, 2)

        # (1, 64, 16, 64)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)

        # (1, 128, 8, 32)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = nn.ReLU(True)
        self.poo3 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))

        # (1, 256, 4, 32)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_2 = nn.ReLU(True)
        self.pool4 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))

        # (1, 512, 2, 32)
        self.conv5 = nn.Conv2d(512, 512, 2, 1, 0)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(True)

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, chan_hidden, chan_hidden),
            BidirectionalLSTM(chan_hidden, chan_hidden, nclass))

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(input)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3_2(self.conv3_2(self.relu3_1(self.bn3(self.conv3_1(x))))))
        x = self.pool4(self.relu4_2(self.conv4_2(self.relu4_1(self.bn4(self.conv4_1(x))))))
        conv = self.relu5(self.bn5(self.conv5(x)))

        b, c, h, w = conv.size()
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)

        output = self.rnn(conv)

        return output