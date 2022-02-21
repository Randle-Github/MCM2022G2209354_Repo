import dataloader
import torch
import numpy as np
import matplotlib.pyplot as plt
import datasets.utils.visual as vis


def lagrange(x, y, w):
    n = len(x)
    res = 0
    for i in range(n):
        temp = 1
        for j in range(n):
            if i != j: temp = temp * (w - x[j]) / (x[i] - x[j])
        res += temp * y[i]
    return res


def Predict_Demo(n):
    '''
    a demo of prediction of price
    '''
    t = Suitable_t(n)
    x, y = dataloader.Dataloader(n + t, "datasets/Gold.npy")
    x = np.array(x[0, :, 0].flatten())
    y = np.array(y[0, :, 0].flatten())
    pred = []
    for i in range(n, n + t):
        pred.append(lagrange(x[:i], y[:i], i))
    print(pred)
    vis.visualise(np.arange(n+t), y, "green", "real curve")
    vis.visualise(np.arange(n, n + t), np.array(pred), "red", "predicted curve")
    plt.legend()
    # plt.ylim(1000, 2400)
    plt.xlabel("date code")
    plt.ylabel("dollar/oz.")
    plt.show()


def Suitable_t(n):
    return min(int(n / 3), 1938 - n)


if __name__ == "__main__":
    Predict_Demo(50)