import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as models
from torch.utils import data
import h5py as h5

class MyDataset(data.Dataset):
    def __init__(self, archive, transform=None):
        self.archive = h5.File(archive, 'r')
        self.labels = self.archive['labels']
        self.data = self.archive['video']
        self.transform = transform

    def __getitem__(self, index):
        datum = self.data[index]
        if self.transform is not None:
            datum = self.transform(datum)
        return datum, self.labels[index]
        
    def __len__(self):
        return len(self.labels)

    def close(self):
        self.archive.close()

class CustomDataset(data.Dataset):
    def __init__(self, data_path, label_path, video_path, transform=None):
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
        self.video_paths = np.load(video_path)
        self.transform = transform

    def __getitem__(self, index):
        datum = self.data[index]
        if self.transform is not None:
            datum = self.transform(datum)
        return datum, self.labels[index], self.video_paths[index]
        
    def __len__(self):
        return len(self.labels)

class LCRNN(nn.Module):
    # input_dim: Number of features for each frames
    # hidden_dim: LSTM hidden units
    # num_layers: Number of lstm units stacked on top
    # s_len: sequence length for input i.e. number of frames in a video
    def __init__(self, input_dim=2048, hidden_dim=128, num_layers=1, seq_len=16, dropout_rate=0.3, batch_size = 2):
        super(LCRNN, self).__init__()
        self.num_classes = 6
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.batch_size = batch_size

        self.dropout = nn.Dropout(p=dropout_rate)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_dim, self.num_classes)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        hidden_state = torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
        cell_state = torch.randn(self.num_layers, self.batch_size, self.hidden_dim)
        hidden = (hidden_state, cell_state)
        # N x 16 x 2048
        x = self.dropout(x)
        # N x 16 x 2048
        x, hidden = self.lstm(x, hidden)
        # N x 16 x 128
        x = self.dropout(x)
        # N x 16 x 128
        x = self.fc1(x)
        # N x 16 x 6
        x = self.softmax(x)
        # N x 16 x 6
        x = torch.mean(x, 1)
        # N x 6
        return x, hidden

batch_size = 128
NET_PATH = './chow_net.pth'

train = np.load('sequences/train.npy')
train_labels = np.load('sequences/train_labels.npy')
val = np.load('sequences/validation.npy')
val_labels = np.load('sequences/validation_labels.npy')
test = np.load('sequences/test.npy')
test_labels = np.load('sequences/test_labels.npy')

train_data = TensorDataset(torch.from_numpy(train), torch.from_numpy(train_labels))
val_data = TensorDataset(torch.from_numpy(val), torch.from_numpy(val_labels))
test_data = TensorDataset(torch.from_numpy(test), torch.from_numpy(test_labels))
# train_data = MyDataset('./processed_data/train_data.h5')
custom = CustomDataset('./processed_data/test_data.npy', './processed_data/test_labels.npy', './processed_data/test_video_paths.npy')

train_loader = DataLoader(train_data, shuffle=True, batch_size=4, drop_last=True)
val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(custom, shuffle=True, batch_size=batch_size, drop_last=True)

net = LCRNN(batch_size=batch_size)
# # inp = torch.from_numpy(train[0:2])
# # # inp = inp.unsqueeze(0)
# # out = net(inp.float())

def training():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(100):  # loop over the dataset multiple times

        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            net.zero_grad()
            # forward + backward + optimize
            outputs, hidden = net(inputs.float())
            _, class_labels = torch.max(labels.data, 1)
            loss = criterion(outputs, class_labels)
            loss.backward()
            optimizer.step()

            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss))

    torch.save(net.state_dict(), NET_PATH)

def evaluate(loader):
    net.load_state_dict(torch.load(NET_PATH))
    classes = ('backhand', 'bvolley', 'service', 'forehand',
                'fvolley', 'smash')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels, i = data
            outputs, hidden = net(images.float())
            _, formatted_labels = torch.max(labels.data, 1)
            _, predicted = torch.max(outputs.data, 1)
            total += formatted_labels.size(0)
            correct += (predicted == formatted_labels).sum().item()
    print('Accuracy of the network on the %d images: %d %%' % (total, 100 * correct / total))

    class_correct = list(0. for i in range(6))
    class_total = list(0. for i in range(6))
    with torch.no_grad():
        for data in loader:
            images, labels, i = data
            outputs, hidden = net(images.float())
            _, formatted_labels = torch.max(labels.data, 1)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == formatted_labels).squeeze()
            for i in range(batch_size):
                label = formatted_labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(6):
        print('Accuracy of %d %5s : %2d %%' % (class_total[i], classes[i], 100 * class_correct[i] / class_total[i]))

# print(train_data.__getitem__(2))
evaluate(test_loader)
# for data in test_loader:
#     images, labels, i = data
#     print(i)
# print('done')
# Inception_V3 = models.inception_v3(pretrained=True)
# Inception_V3.fc = nn.Identity()
# for param in Inception_V3.parameters():
#     print(param)
#     param.requires_grad = False

# # for param in Inception_V3.parameters():
#     # print(param.requires_grad)

# print(net.parameters())

# test = np.load('./processed_data/train/p1_backhand_s3.npy')
# print(test.shape)