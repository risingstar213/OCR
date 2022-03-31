import keys
train_infofile = 'data_set/infofile_train_10w.txt'
val_infofile = 'data_set/infofile_test.txt'
num_workers = 0

random_sample = True
batch_size = 50
workers = 0
imgH = 32
imgW = 280

nc = 1
nclass = len(keys.alphabet)+1
nh = 256