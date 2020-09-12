# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# net = Net()

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# for epoch in range(2):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0

# print('Finished Training')

import os
import csv
import cv2
import shutil
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

# with open('data_file.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for row in csv_reader:
#         source = '/Users/SujayGarlanka/Projects/ML/jamak.ai/stroke_recognition/videos/' + row[2] + '.avi'
#         destination = '/Users/SujayGarlanka/Projects/ML/jamak.ai/stroke_recognition/data/' + row[1]
#         dest = shutil.move(source, destination)

def img_show_np(img):
    cv2.imshow('image', img) 
    #waits for user to press any key  
    #(this is necessary to avoid Python kernel form crashing) 
    cv2.waitKey(0)  
    #closing all open windows  
    cv2.destroyAllWindows()  

def convert_data_to_sequence():
    """
    This function, used in extract_seq_features(), obtains a list of
    frames from a given sample video. Each frame is a N x M x 3 array.
    """
    Inception_V3 = models.inception_v3(pretrained=True)
    # print(Inception_V3)
    # removed = list(Inception_V3.children())[:-1]
    # Inception_V3 = nn.Sequential(*removed)
    Inception_V3.fc = nn.Identity()
    Inception_V3.eval()
    num_frames = 16
    with open('data_file.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        length_of_csv = 1980
        count = 1
        for row in csv_reader:
            path = os.path.join('data', row[1], row[2] + '.avi')

            vidcap = cv2.VideoCapture(path)

            frames = []

            # extract frames
            while True:
                success, image = vidcap.read()
                if not success:
                    break
                img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frames.append(img_RGB)
            # downsample if desired and necessary
            if num_frames < len(frames):
                skip = len(frames) // num_frames
                frames = [frames[i] for i in range(0, len(frames), skip)]
                frames = frames[:num_frames]

            sequence = []
            for img in frames:
                PIL_image = Image.fromarray(np.uint8(img)).convert('RGB')
                preprocess = transforms.Compose([
                    transforms.Resize(299),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                         0.229, 0.224, 0.225]),
                ])
                input_tensor = preprocess(PIL_image)
                input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
                # move the input and model to GPU for speed if available
                if torch.cuda.is_available():
                    input_batch = input_batch.to('cuda')
                    Inception_V3.to('cuda')
                with torch.no_grad():
                    features = Inception_V3(input_batch)
                sequence.append(features[0].tolist())  # only take first dimension
            features_path = os.path.join('sequences', row[1], row[2] + '.npy')
            np.save(features_path, sequence)
            print('Progress percent: ' + str(count/length_of_csv * 100))
            count+=1

convert_data_to_sequence()
