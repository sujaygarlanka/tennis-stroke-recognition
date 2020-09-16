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

def convert_video_to_sequence():
    """
    This file imports RGB videos, breaks each one into its frames,
    and runs each frame through the Inception V3 net to get a features list
    for each frame. It finally concatenates the features list for all the frames in a video
    and saves it to a numpy file in the sequences folder. 
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

def save_train_val_test_data():
    train = []
    train_labels = []
    validation = []
    validation_labels = []
    test = []
    test_labels = []
    with open('data_file.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            stroke = row[1]
            video_name = row[2]
            path = os.path.join('sequences', stroke, video_name + '.npy')
            data = np.load(path)
            if row[0] == 'train':
                train.append(data)
                train_labels.append(get_one_hot_encoding(stroke))
            elif row[0] == 'validation':
                validation.append(data)
                validation_labels.append(get_one_hot_encoding(stroke))
            elif row[0] == 'test':
                test.append(data)
                test_labels.append(get_one_hot_encoding(stroke))

    train_path = os.path.join('sequences', 'train.npy')
    train_labels_path = os.path.join('sequences', 'train_labels.npy')
    val_path = os.path.join('sequences', 'validation.npy')
    val_labels_path = os.path.join('sequences', 'validation_labels.npy')
    test_path = os.path.join('sequences', 'test.npy')
    test_labels_path = os.path.join('sequences', 'test_labels.npy')

    np.save(train_path, np.array(train))
    np.save(train_labels_path, np.array(train_labels))
    np.save(val_path, np.array(validation))
    np.save(val_labels_path, np.array(validation_labels))
    np.save(test_path, np.array(test))
    np.save(test_labels_path, np.array(test_labels))

def get_one_hot_encoding(label):
    encoding = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if label == 'backhand':
        encoding[0] = 1.0
    elif label == 'bvolley':
        encoding[1] = 1.0
    elif label == 'service':
        encoding[2] = 1.0
    elif label == 'forehand':
        encoding[3] = 1.0
    elif label == 'fvolley':
        encoding[4] = 1.0
    elif label == 'smash':
        encoding[5] = 1.0
    return encoding
    
save_train_val_test_data()