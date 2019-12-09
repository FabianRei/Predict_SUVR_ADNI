from dl.data.get_dataset import get_dataset
from dl.neural_network.resnet import ResNet50
from dl.neural_network.train_test import train
from datetime import datetime
from dl.data.project_logging import Logger, CsvWriter
import numpy as np
import torch
from torch import nn
from torch import optim
import sys
import os
import GPUtil
import pickle


windows_db = False
test_split = 0.2
if windows_db:
    h5_path = r'C:\Users\Fabian\stanford\fed_learning\federated_learning_data\slice_data.h5'
else:
    h5_path = '/scratch/reith/fl/experiments/one_slice_dataset/slice_data.h5'
extra_info = '_pretrain_normalizeData'
num_epochs = 30
lr = 0.01
decrease_after = 3
rate_of_decrease = 0.33
gpu_device = 0
save_pred_labels = True
label_names = ['label_amyloid']

device_ids = GPUtil.getAvailable(order='first', limit=6, maxLoad=0.1, maxMemory=0.1, excludeID=[], excludeUUID=[])
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_ids[gpu_device])

data, labels = get_dataset(h5_path, label_names=label_names)
np.random.seed(42)
shuff_idxs = np.random.permutation(len(data))
data = data[shuff_idxs]
labels = labels[shuff_idxs]
# normalize data
data -= data.mean()
data /= data.std()

if windows_db:
    data = data[:100]
    labels = labels[:100]
cutoff = int(len(data)*0.2)
test_data = data[:cutoff]
test_labels = labels[:cutoff]
train_data = data[cutoff:]
train_labels = labels[cutoff:]

test_data = torch.from_numpy(test_data).type(torch.float32)
test_labels = torch.from_numpy(test_labels).type(torch.long)
train_data = torch.from_numpy(train_data).type(torch.float32)
train_labels = torch.from_numpy(train_labels).type(torch.long)
num_classes = len(torch.unique(train_labels))

Net = ResNet50(pretrained=True, num_classes=num_classes)
Net.cuda()
criterion = nn.NLLLoss()
out_path = os.path.dirname(h5_path)
time_stamp = datetime.now().strftime('%m-%d_%H-%M')
log_path = os.path.join(out_path, f"training_log_{time_stamp}{extra_info}.txt")
sys.stdout = Logger(log_path)
csv_path = os.path.join(out_path, f"train_test_accuracy_{time_stamp}{extra_info}.csv")
csv_writer = CsvWriter(csv_path, header=['test_acc', 'train_acc', 'train_loss', 'epoch'])

for i in range(num_epochs):
    if i % decrease_after == 0:
        if i > 0:
            lr = lr*rate_of_decrease
        print(f"Trainig for {decrease_after} epochs with a learning rate of {lr}..")
    optimizer = optim.Adam(Net.parameters(), lr=lr)
    train_result = train(batch_size=32, train_data=train_data, train_labels=train_labels, test_data=test_data,
                         test_labels=test_labels, Net=Net, optimizer=optimizer, criterion=criterion,
                         test_interval=1, epochs=1, dim_in='default')
    Net, test_acc, test_pred_label, train_acc, train_loss, train_pred_label = train_result
    csv_writer.write_row(test_acc=test_acc, train_acc=train_acc, train_loss=train_loss, epoch=i)
    if save_pred_labels:
        pickle_fn = os.path.join(out_path, f"epoch_{i}_pred_labels_train_test_epoch_{time_stamp}{extra_info}.p")
        pickle_object = {'train': train_pred_label, 'test': test_pred_label}
        with open(pickle_fn, 'wb') as f:
            pickle.dump(pickle_object, f)
    print(f"Test accuracy is {test_acc * 100:.2f} percent")






