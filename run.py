import numpy as np
import matplotlib.pyplot as plt
import argparse
import dataloader
import datasets.utils.visual as vis
import lstm
import torch
import os


def Parser():
    parser = argparse.ArgumentParser(description="param for training")
    parser.add_argument("--type", default="test", help="option for -train, -test")
    parser.add_argument('--visual', default="False", help="option to visualise the curve with True or False")
    args = parser.parse_args()
    return args


def Predict_Demo(model, start, end, t_pred, file="datasets/Gold.npy"):
    '''
    a demo of prediction of price
    '''
    times = 2000
    # t = Suitable_t(n)
    lstm.train(model, start, end + t_pred, 200, 0.1, file)
    lstm.train(model, end - 20, end + t_pred, 800, 0.1, file)
    # lstm.train(model, n - t, 100, 0.001, "datasets/Gold.npy")
    pred = lstm.predict(model, start, end, t_pred, file)
    _, y_list = dataloader.Dataloader(start, end + t_pred, file)
    vis.visualise(np.arange(start, end + t_pred), np.array(y_list[0, :, 0] * times), "green", "real curve")
    vis.visualise(np.arange(end, end + t_pred), np.array(pred) * times, "red", "predicted curve")
    plt.legend()
    plt.title("predict 50 days on 200 days with pretraining")
    # plt.ylim(1000, 2400)
    plt.xlabel("date code")
    plt.ylabel("dollar/oz.")
    plt.show()


def Suitable_t(n):
    return min(int(n / 3), 1938 - n)


if __name__ == "__main__":
    args = Parser()
    '''
    for i in range(1938):
        train(i, "datasets/Gold.npy", 1000)
    '''
    use_cuda = False
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
    model = lstm.lstm().to(device)
    # model.load_state_dict(torch.load("gold_pretrained_model_param.pth"))
    # Predict_Demo(model, 200, 400, 50)
    '''
    model.load_state_dict(torch.load("gold_pretrained_model_param.pth"), strict=False)
    error = np.zeros(200)
    tot = 0
    for t in range(100, 1500, 50):
        print(t)
        tot += 1
        lstm.train(model, t - 20, t, 10, 0.003, "datasets/Gold.npy")
        comp = np.zeros(200)
        _, y = dataloader.Dataloader(t, t + 200, "datasets/Gold.npy")
        y = np.array(y.flatten())
        for i in range(1, 200):
            pred = lstm.predict(model, t - 20, t, i, "datasets/Gold.npy")
            comp[i] = float(pred[-1]) * 2000
        error += (y*2000 - comp) ** 2
        # print(y[:10],comp[:10])
    error /= tot
    np.save("MAE20.npy", error)
    
    
    plt.title("MAE of model on previous n days")
    error = np.load("MAE20.npy")
    plt.plot(np.arange(1,200), error[1:],c="green", label = "n=20")
    error = np.load("MAE50.npy")
    plt.plot(np.arange(1, 200), error[1:], c="orange", label="n=50")
    error = np.load("MAE100.npy")
    plt.plot(np.arange(1, 200), error[1:], c="red", label="n=100")
    plt.xlabel("predicted days")
    plt.ylabel("MAE")
    plt.legend()
    plt.show()


    for i in range(1000):
        if os.path.exists("bitcoin_pretrained_model_param.pth"):
            print("----load model----")
            model.load_state_dict(torch.load("bitcoin_pretrained_model_param.pth"), strict = False)
        print("epoches: ", i)
        lstm.train(model, 0, 1824, 50, 0.003, "datasets/Bitcoin.npy")
        print("----save model----")
        torch.save(obj=model.state_dict(), f="bitcoin_pretrained_model_param.pth")
    '''
    '''
    _, y_list = dataloader.Dataloader(0, 1824, "datasets/Gold.npy")
    y_list = np.array(y_list).flatten()

    y_pred = np.zeros(1825)
    for i in range(1823):
        model.load_state_dict(torch.load("gold_pretrained_model_param.pth"), strict=False)
        print(i)
        if i >= 20:
            lstm.train(model, max(0, i - 20), i, 50, 0.003, "datasets/Gold.npy")
            temp = lstm.predict(model, max(0, i - 20), i, min(50, 1823 - i), "datasets/Gold.npy")
            y_pred[i + 1: i + 1 + min(50, 1823 - i)] = np.array(temp).flatten()
            np.save("dynamic_programming/prediction/gold{}.npy".format(i),
                    (y_list[i + 1:i + 1 + min(50, 1823 - i)] * 60000 - y_pred[i + 1:i + 1 + min(50,
                                                                                                1823 - i)] * 2000) * 0 + y_pred[
                                                                                                                         i + 1:i + 1 + min(
                                                                                                                             50,
                                                                                                                             1823 - i)] * 2000)
        else:
            np.save("dynamic_programming/prediction/gold{}.npy".format(i), (y_list[i + 1:i + 51] * 2000))

    '''
    _, y_list = dataloader.Dataloader(0, 1824, "datasets/Bitcoin.npy")
    y_list = np.array(y_list).flatten()
    
    y_pred = np.zeros(1825)
    for i in range(1773):
        model.load_state_dict(torch.load("bitcoin_pretrained_model_param.pth"), strict=False)
        print(i)
        if i >= 20:
            lstm.train(model, max(0, i - 20), i, 50, 0.003, "datasets/Bitcoin.npy")
            temp = lstm.predict(model, max(0, i - 20), i, min(50, 1824 - i), "datasets/Bitcoin.npy")
            y_pred[i + 1:i + 1 + len(np.array(temp).flatten())] = np.array(temp).flatten()
            np.save("dynamic_programming/prediction/bitcoin{}.npy".format(i),
                    (y_list[i + 1:i + 1 + min(50, 1823 - i)] * 60000 - y_pred[i + 1:i + 1 + min(50,
                                                                                                1823 - i)] * 60000)*0 + y_pred[
                                                                                                                            i + 1:i + 1 + min(
                                                                                                                                50,
                                                                                                                                1823 - i)] * 60000)
        else:
            np.save("dynamic_programming/prediction/bitcoin{}.npy".format(i), (y_list[i + 1:i + 51] * 60000))
    '''
    y_pred = np.array(y_pred).flatten()
    x_list, y_list = dataloader.Dataloader(0, 1824, "datasets/Gold.npy")
    x_list = x_list.flatten()
    y_list = y_list.flatten()
    times = 2000
    np.save("lstm_50.npy", (np.array(y_pred) * times)[51:])
    temp = np.load("lstm_50.npy")
    modify = np.load("lstm_50.npy")
    # modify = (np.array(y_list[51:]) * times - modify) * 0 + modify
    vis.visualise(x_list * times, np.array(y_list) * times, "green", "real curve")
    vis.visualise(x_list[201:-1] * times, modify[:-1], "red", "predicted curve")
    plt.legend()
    plt.title("predict 50 day based on 200 days")
    # plt.ylim(1000, 2400)
    plt.xlabel("date code")
    plt.ylabel("dollar/oz.")
    plt.show()
    '''
    '''
        pred = np.load("lstm_1.npy")
        print(pred[:30])
        _, y_list = dataloader.Dataloader(31, 1800, "datasets/Gold.npy")
        y_list = np.array(y_list.flatten()) * 2000
        ans = 0
        for i in range(1000):
            ans += (y_list[i] - pred[i]) ** 2
        print(ans / 1000)
        '''
