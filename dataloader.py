import numpy as np
import torch


def Dataloader(start, end, file):
    '''
    read n days' price
    '''
    temp = np.load(file)
    return (torch.from_numpy((np.arange(start, end) / 2000).reshape((1, end - start, 1)))).float(), (
        torch.from_numpy(temp[start:end].reshape((1, end - start, 1)) / 60000)).float()


def Reverse_Dataloader(start, end, file):
    '''
    read n days' price
    '''
    temp = np.load(file)
    return (torch.from_numpy(((np.arange(start, end))[::-1] / 2000).reshape((1, end - start, 1)))).float(), (
        torch.from_numpy((temp[start:end].reshape((1, end - start, 1)))[::-1] / 60000)).float()


if __name__ == "__main__":
    pass
