import os
import numpy as np
import h5py
import random
import torch
from torch.utils.data import Dataset


class SpeechDigitsDataset(Dataset):
    """
    A class used to read and transform the N-TIDIGITS18 dataset (available at http://sensors.ini.uzh.ch/databases.html)

    Attributes:
        data_root (str): path of the HDF5 dataset file (starting from the current directory)
        mode (str): [train] or [test]
        nb_digits (int): if positive, select data with number digits equal to nb_digits (default -1: all data is selected)
        transform (BinningHistogram): a callable object used to transform the dataset items
    """
    def __init__(self, data_root, mode, train_proportion=.8, label_dct=None, nb_digits=-1, transform=None):
        assert mode in ["train", "test"], 'mode should be "train" or "test"'

        self.filename = os.path.join(os.path.abspath(data_root), 'n-tidigits.hdf5')
        self.labels = []
        self.digits_labels = []
        self.mode = mode
        self.transform = transform

        file = h5py.File(self.filename, 'r')
        labels_dset = file['/'][mode + '_labels']
        if (nb_digits >= 0):
            self.labels = [elem for elem in labels_dset if (len(SpeechDigitsDataset.get_label_info(elem)[-1]) == nb_digits)]
        else:
            self.labels = [elem for elem in labels_dset]

        random.seed(0)
        train_idx = random.sample(range(len(self.labels)), int(np.round(train_proportion * len(self.labels))))
        test_idx = [idx for idx in range(len(self.labels)) if idx not in train_idx]

        if self.mode=="train":
            train_labels = [self.labels[idx] for idx in train_idx]
            self.labels = train_labels
        elif self.mode=="test":
            test_labels = [self.labels[idx] for idx in test_idx]
            self.labels = test_labels

        self.digits_labels = [SpeechDigitsDataset.get_label_info(elem)[-1] for elem in self.labels]

        if nb_digits==1:
            if label_dct is None:
                label_dct = {'o': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'z': 0}
            digits = [label_dct.get(digit) for digit in self.digits_labels]
            self.digits_labels = digits

            if self.mode == "train":
                label_weights = 1./np.unique(self.digits_labels, return_counts=True)[1]
                label_weights /=  np.sum(label_weights)
                self.weights = torch.DoubleTensor([label_weights[label] for label in self.digits_labels])

        file.close()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        digits_label = self.digits_labels[idx]

        file = h5py.File(self.filename, 'r')

        addresses_group = file['/'][self.mode + '_addresses']
        timestamps_group = file['/'][self.mode + '_timestamps']

        addresses_dset = addresses_group[label]
        addresses = np.zeros(addresses_dset.size)
        addresses_dset.read_direct(addresses)

        timestamps_dset = timestamps_group[label]
        timestamps = np.zeros(timestamps_dset.size)
        timestamps_dset.read_direct(timestamps)

        file.close()

        item = np.array([addresses, timestamps]).T

        if self.transform is not None:
            item = self.transform(item)

        return item, digits_label

    def get_item_from_label(self, label):
        """
        A method to extract an item corresponding to label

        Args:
            label (numpy.bytes_): full label of the item

        Returns:
            item (numpy.ndarray) : An item containing addresses and timestamps of the label.
        """
        idx = self.labels.index(label)
        item = self[idx][0]
        return item

    @staticmethod
    def get_label_info(label):
        """
        A static method to extract information from the label. A full label becomes a list [gender, initials, sample, digits].

        Args:
            label (numpy.bytes_): full label of the item

        Returns:
            label_info (list) : A list containing information of the label [gender, initials, sample, digits].
        """
        label_info = str(label)[2:-1].split("-")
        return label_info

    def get_max_size(self):
        """
        A method to get the maximum size of the recordings in the selected data of the dataset (before the potential transform operation).

        Args:

        Returns:
            max_size (int) : the maximum size of the recordings.
        """
        max_size = 0
        file = h5py.File(self.filename, 'r')
        for idx in range(len(self)):
            label = self.labels[idx]
            timestamps_group = file['/'][self.mode + '_timestamps']
            timestamps_dset = timestamps_group[label]
            size = len(timestamps_dset)
            if size > max_size: max_size = size
        file.close()
        return max_size

        # max_size = 0
        # for i in range(len(self)):
        #     item = self[i][0]
        #     if len(item) > max_size:
        #         max_size = len(item)
        # return max_size

    def get_max_end_time(self):
        """
        A method to get the maximum end time of the recordings in the selected data of the dataset.

        Args:

        Returns:
            max_end_time (float) : the maximum end time of the recordings.
        """
        max_end_time = 1.
        file = h5py.File(self.filename, 'r')
        for idx in range(len(self)):
            label = self.labels[idx]
            timestamps_group = file['/'][self.mode + '_timestamps']
            timestamps_dset = timestamps_group[label]
            end_time = timestamps_dset[-1]
            if end_time > max_end_time: max_end_time = end_time
        file.close()
        return max_end_time

class BinningHistogram:
    def __init__(self, binning_method="time", T_l=0.005, E=25):
        self.binning_method = binning_method
        self.T_l = T_l
        self.E = E

    @staticmethod
    def time_binning(item, T_l):
        """Time-Binned Spike Count Features.

            This static method returns time-binned spike count features for a certain duration T_l [1].

            Args:
                item (numpy.ndarray): The item is a numpy array containing timestamps and addresses.
                T_l (float): Frame duration.

            Returns:
                features (numpy.ndarray) : The return value. True for success, False otherwise.

            References:
                [1] Jithendar Anumula, Daniel Neil, Tobi Delbruck, & Shih-Chii Liu (2018). Feature Representations for Neuromorphic Audio Spike StreamsFrontiers in Neuroscience, 12.
            """

        j = 0
        addresses = item[:, 0]
        timestamps = item[:, 1]
        features = []

        while j < timestamps[-1] / T_l:
            feature = np.zeros(64)
            for i in range(64):
                interest = timestamps[addresses == i]
                feature[i] = np.sum((interest < (j + 1) * T_l) * (interest >= j * T_l))
            features.append(feature)
            j += 1
        return np.stack(features)

    @staticmethod
    def event_binning(item, E):
        """Event-Binned Spike Count Features.

            This static method returns event-binned spike count features for a a number of events E [1].

            Args:
                item (numpy.ndarray): The item is a numpy array containing timestamps and addresses.
                E (float): Number of events in a frame.

            Returns:
                features (numpy.ndarray) : The return value. True for success, False otherwise.

            References:
                [1] Jithendar Anumula, Daniel Neil, Tobi Delbruck, & Shih-Chii Liu (2018). Feature Representations for Neuromorphic Audio Spike StreamsFrontiers in Neuroscience, 12.
            """
        j = 0
        features = []

        while j < len(item) / E:
            feature = np.zeros(64)
            region = item[j * E:(j + 1) * E, :]
            for i in range(len(region)):
                feature[int(region[i][0])] = region[i][1]
            features.append(feature)
            j += 1
        return np.stack(features)

    def __call__(self, item):
        assert self.binning_method in ["time", "event"], 'binning_method should be [time] or [event]'
        if self.binning_method == "time":
            return BinningHistogram.time_binning(item, self.T_l)
        else:
            assert type(param) == int, "You're in event binning mode, you should input an integer."
            return BinningHistogram.event_binning(item, self.E)

class Pad:
    """
    A class used to pad the result of the binning histogram to a fixed size

    Attributes:
        size: size of the padding
    """
    def __init__(self, size):
        self.size = size

    """
    A method to get the maximum end time of the recordings in the selected data of the dataset.

    Args:
        features (numpy.ndarray): the feature array created by the BinningHistogram class.

    Returns:
        features (numpy.ndarray) : the zero-padded features to the specified size across the 0-axis.
    """
    def __call__(self, features):
        pad_size = (self.size - len(features))
        padding = np.zeros((pad_size, 64))
        features = np.vstack((features, padding))
        return features


if __name__ == "__main__":
    data_root = "../notebooks"
    train_dataset = SpeechDigitsDataset(data_root, train_proportion=0.8, mode="train", nb_digits=1)
    print(SpeechDigitsDataset.get_label_info(b'man-jr-b-6'))
    print(len(train_dataset))
    # print(train_dataset[0])
    # print(train_dataset[0][0].shape)
    # print(train_dataset.get_item_from_label(b'man-jr-b-6'))
    # print(train_dataset.get_max_end_time())
