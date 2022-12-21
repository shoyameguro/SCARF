import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_boston
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

income_columns = ['age', 'workclass', 'fnlwgt', 'education',
                  'education-num', 'marital-status', 'occupation',
                  'relationship', 'race', 'sex', 'capital-gain',
                  'capital-loss', 'hours-per-week', 'native-country', 'income']


class TabularDataset(Dataset):
    '''
    income train path './data/income_data.csv'
    income test path './data/income_test.csv'
    '''

    def __init__(self, x, y):
        self.x = torch.tensor(x).to(torch.float)
        self.y = torch.tensor(y).to(torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class CorruptionDataset(Dataset):
    def __init__(self, x, x_tilde, corruption):
        self.x = x
        self.x_tilde = x_tilde
        self.corruption = corruption

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.x_tilde[index], self.corruption[index]


def read_csv(path, label, columns, header=None):
    df = pd.read_csv(path, header=header, names=columns)
    x = df.drop(label, axis=1)
    y = pd.get_dummies(df[label])
    return x.to_numpy(), y.to_numpy()


def mnist_to_tabular(x, y):
    x = x / 255.0
    y = np.asarray(pd.get_dummies(y))

    # flatten
    no, dim_x, dim_y = np.shape(x)
    x = np.reshape(x, [no, dim_x * dim_y])
    return x, y


def get_dataset(data_name, label_data_rate):
    '''
    input:
        data_name: str
        label_data_rate: float
    return:
        labeled dataset, unlabeled dataset, test dataset
        unlabeled dataset is a dataset that contains labels but is not used for training
    '''
    if data_name == 'iris':
        data = load_iris()
        data.target = np.asarray(pd.get_dummies(data.target))
        x_train, x_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            test_size=0.2,
            random_state=42)
    elif data_name == 'wine':
        data = load_wine()
        data.target = np.asarray(pd.get_dummies(data.target))
        x_train, x_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            test_size=0.2,
            random_state=42)
    elif data_name == 'boston':
        data = load_boston()
        data.target = np.asarray(pd.get_dummies(data.target))
        x_train, x_test, y_train, y_test = train_test_split(
            data.data,
            data.target,
            test_size=0.2,
            random_state=42)
    elif data_name == 'mnist':
        train_set = torchvision.datasets.MNIST('../../../data', train=True, download=True)
        test_set = torchvision.datasets.MNIST('../../../data', train=False, download=True)
        x_train, y_train = mnist_to_tabular(train_set.data.numpy(), train_set.targets.numpy())
        x_test, y_test = mnist_to_tabular(test_set.data.numpy(), test_set.targets.numpy())

    # Divide labeled and unlabeled data
    idx = np.random.permutation(len(y_train))
    # Label data : Unlabeled data = label_data_rate:(1-label_data_rate)
    label_idx = idx[:int(len(idx)*label_data_rate)]
    unlab_idx = idx[int(len(idx)*label_data_rate):]

    # Unlabeled data
    x_unlab = x_train[unlab_idx, :]
    y_unlab = y_train[unlab_idx, :]

    # Labeled data
    x_label = x_train[label_idx, :]
    y_label = y_train[label_idx, :]

    return TabularDataset(x_label, y_label),\
        TabularDataset(x_unlab, y_unlab),\
        TabularDataset(x_test, y_test)


if __name__ == '__main__':
    a, _, _ = get_dataset('mnist', 0.1)
    print(a[[0, 1]])
