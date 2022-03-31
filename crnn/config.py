import keys
train_infofile = 'dataset/train.txt'
val_infofile = ''
num_workers = 0

random_sample = True
batch_size = 1
workers = 0
imgH = 32
imgW = 280

nc = 1
nclass = len(keys.alphabet)+1
nh = 256


pretrained_model = True
pretrained_weights = None
checkpoints_dir = './checkpoints'