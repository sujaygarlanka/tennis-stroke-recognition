import numpy as np

train = np.load('sequences/train.npy')
validation = np.load('sequences/validation.npy')
test = np.load('sequences/test.npy')

print(train.shape)
print(validation.shape)
print(test.shape)