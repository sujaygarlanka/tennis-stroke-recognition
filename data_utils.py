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
import h5py

def convert_video_to_sequence():
    """
    This file imports RGB videos, breaks each one into its frames,
    and runs each frame through the Inception V3 net to get a features list
    for each frame. It finally concatenates the features list for all the frames in a video
    and adds it to a list of training, validation, or testing examples. Finally, the examples are saved to
    numpy files.
    """
    Inception_V3 = models.inception_v3(pretrained=True)
    Inception_V3.fc = nn.Identity()
    Inception_V3.eval()
    num_frames = 16
        # First index is features of the video, second is the label, third is file path to the original video
    train_data = [[],[],[]]
    val_data = [[],[],[]]
    test_data = [[],[],[]]
    with open('data_file.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        length_of_csv = 1980
        count = 1
        for row in csv_reader:
            stroke = row[1]
            video_name = row[2]
            video_path = os.path.join('data', stroke, video_name + '.avi')

            vidcap = cv2.VideoCapture(video_path)
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

            if row[0] == 'train':
                train_data[0].append(sequence)
                train_data[1].append(get_one_hot_encoding(stroke))
                train_data[2].append(video_path)
            elif row[0] == 'validation':
                val_data[0].append(sequence)
                val_data[1].append(get_one_hot_encoding(stroke))
                val_data[2].append(video_path)
            elif row[0] == 'test':
                test_data[0].append(sequence)
                test_data[1].append(get_one_hot_encoding(stroke))
                test_data[2].append(video_path)

            print('Progress percent: ' + str(count/length_of_csv * 100))
            count+=1

    np.save(os.path.join('processed_data', 'train_data.npy'), np.array(train_data[0]))
    np.save(os.path.join('processed_data', 'train_labels.npy'), np.array(train_data[1]))
    np.save(os.path.join('processed_data', 'train_video_paths.npy'), np.array(train_data[2]))
    
    np.save(os.path.join('processed_data', 'val_data.npy'), np.array(val_data[0]))
    np.save(os.path.join('processed_data', 'val_labels.npy'), np.array(val_data[1]))
    np.save(os.path.join('processed_data', 'val_video_paths.npy'), np.array(val_data[2]))
    
    np.save(os.path.join('processed_data', 'test_data.npy'), np.array(test_data[0]))
    np.save(os.path.join('processed_data', 'test_labels.npy'), np.array(test_data[1]))
    np.save(os.path.join('processed_data', 'test_video_paths.npy'), np.array(test_data[2]))

def save_train_val_test_data():
    # First index is features of the video, second is the label, third is file path to the original video
    train_data = [[],[],[]]
    val_data = [[],[],[]]
    test_data = [[],[],[]]
    with open('data_file.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        length_of_csv = 1980
        count = 1
        for row in csv_reader:
            stroke = row[1]
            video_name = row[2]
            path = os.path.join('sequences', stroke, video_name + '.npy')
            data = np.load(path)
            video_path = os.path.join('data', stroke, video_name + '.avi')
            if row[0] == 'train':
                train_data[0].append(data)
                train_data[1].append(get_one_hot_encoding(stroke))
                train_data[2].append(video_path)
            elif row[0] == 'validation':
                val_data[0].append(data)
                val_data[1].append(get_one_hot_encoding(stroke))
                val_data[2].append(video_path)
            elif row[0] == 'test':
                test_data[0].append(data)
                test_data[1].append(get_one_hot_encoding(stroke))
                test_data[2].append(video_path)

    np.save(os.path.join('processed_data', 'train_data.npy'), np.array(train_data[0]))
    np.save(os.path.join('processed_data', 'train_labels.npy'), np.array(train_data[1]))
    np.save(os.path.join('processed_data', 'train_video_paths.npy'), np.array(train_data[2]))
    
    np.save(os.path.join('processed_data', 'val_data.npy'), np.array(val_data[0]))
    np.save(os.path.join('processed_data', 'val_labels.npy'), np.array(val_data[1]))
    np.save(os.path.join('processed_data', 'val_video_paths.npy'), np.array(val_data[2]))
    
    np.save(os.path.join('processed_data', 'test_data.npy'), np.array(test_data[0]))
    np.save(os.path.join('processed_data', 'test_labels.npy'), np.array(test_data[1]))
    np.save(os.path.join('processed_data', 'test_video_paths.npy'), np.array(test_data[2]))

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

def convert_video_to_net_data():
    num_frames = 16
    train_data = h5py.File('./processed_data/train_data.h5', 'w')
    train_data.create_dataset('video', (0, 16, 3, 299, 299), maxshape=(None, 16, 3, 299, 299), compression="gzip")
    train_data.create_dataset('labels', (0,6), maxshape=(None, 6), compression="gzip")
    validation_data = h5py.File('./processed_data/validation_data.h5', 'w')
    validation_data.create_dataset('video', (0, 16, 3, 299, 299), maxshape=(None, 16, 3, 299, 299), compression="gzip")
    validation_data.create_dataset('labels', (0,6), maxshape=(None, 6), compression="gzip")
    test_data = h5py.File('./processed_data/test_data.h5', 'w')
    test_data.create_dataset('video', (0, 16, 3, 299, 299), maxshape=(None, 16, 3, 299, 299), compression="gzip")
    test_data.create_dataset('labels', (0,6), maxshape=(None, 6), compression="gzip")
    with open('data_file.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        length_of_csv = 1980
        count = 1
        for row in csv_reader:
            stroke = row[1]
            video_name = row[2]

            path = os.path.join('data', stroke, video_name + '.avi')
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

            video = []
            # preprocess frames for Inception net
            for img in frames:
                PIL_image = Image.fromarray(np.uint8(img))
                preprocess = transforms.Compose([
                    transforms.Resize(299),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                            0.229, 0.224, 0.225]),
                ])
                image = preprocess(PIL_image)
                video.append(image.tolist())

            if row[0] == 'train':
                train_data["video"].resize((train_data["video"].shape[0] + 1), axis = 0)
                train_data["video"][-1] = video
                train_data["labels"].resize((train_data["labels"].shape[0] + 1), axis = 0)
                train_data["labels"][-1] = get_one_hot_encoding(stroke)
            elif row[0] == 'validation':
                validation_data["video"].resize((validation_data["video"].shape[0] + 1), axis = 0)
                validation_data["video"][-1] = video
                validation_data["labels"].resize((validation_data["labels"].shape[0] + 1), axis = 0)
                validation_data["labels"][-1] = get_one_hot_encoding(stroke)
            elif row[0] == 'test':
                test_data["video"].resize((test_data["video"].shape[0] + 1), axis = 0)
                test_data["video"][-1] = video
                test_data["labels"].resize((test_data["labels"].shape[0] + 1), axis = 0)
                test_data["labels"][-1] = get_one_hot_encoding(stroke)
            print('Progress percent: ' + str(count/length_of_csv * 100))
            count+=1
    train_data.close()
    validation_data.close()
    test_data.close()

# convert_video_to_net_data()
save_train_val_test_data()

# Show Image
# show_image = image.numpy()
# show_image = np.rollaxis(show_image, 0, 3)
# show_image = show_image - show_image.min()
# show_image = show_image/show_image.max()
# plt.imshow(show_image)
# plt.show()