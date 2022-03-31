IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7

RPN_POSITIVE_NUM = 150
RPN_TOTAL_NUM = 300
OHEM = True

IMAGE_MEAN = [123.68, 116.779, 103.939]

pretrained_weights = 'checkpoints/v3_ctpn_ep22_0.3801_0.0971_0.4773.pth'
icdar17_mlt_img_dir = './train_data/train_img/'
icdar17_mlt_gt_dir = './train_data/train_label/'

num_workers = 0

checkpoints_dir = './checkpoints'
outputs = r'./logs'
