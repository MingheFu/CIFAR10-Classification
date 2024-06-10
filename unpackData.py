import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if __name__ == '__main__':
    data = []
    labels = []
    for i in range(1, 6):
        batch_name = f'data_batch_{i}'
        batch = unpickle(batch_name)
        data.append(batch[b'data'])
        labels.extend(batch[b'labels'])

    # Convert list to numpy arrays
    data = np.vstack(data)
    labels = np.array(labels)
    np.save('data.npy', data)
    np.save('labels.npy', labels)
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)
###
    test_batch = unpickle('test_batch')
    test_data = test_batch[b'data']
    test_labels = test_batch[b'labels']
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    np.save('test_data.npy', test_data)
    np.save('test_labels.npy', test_labels)

    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    print("Test Data shape:", test_data.shape)
    print("Test Labels shape:", test_labels.shape)

