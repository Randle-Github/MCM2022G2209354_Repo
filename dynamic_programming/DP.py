import utils
import dataloader
import numpy as np
import random
import matplotlib.pyplot as plt

gold_allow = np.load("prediction/gold_allow.npy")
Gold_price = np.load("prediction/Gold.npy")
Bitcoin_price = np.load("prediction/Bitcoin.npy")


class Day():
    def __init__(self, k, day_price):
        """
        k: remain the previous k states
        day_price: this days's price
        """
        self.S = np.zeros((k, 3))
        self.P = day_price
        for i in range(len(self.P)):
            if self.P[i] == 0:
                self.P[i] += 0.001
        self.k = k
        self.operation = np.zeros((k, 3))  # record (ratio[0], ratio[1], type)

    def insert(self, temp):
        """
        insert a new state
        """
        s = temp[0]
        o = temp[1]
        f = utils.F(s, self.P)
        if f > utils.F(self.S[self.k - 1], self.P):
            self.S[self.k - 1] = s
            self.operation[self.k - 1] = o
            for i in range(self.k - 2, -1, -1):
                if f > utils.F(self.S[i], self.P):
                    self.S[i + 1] = self.S[i]
                    self.S[i] = s
                    self.operation[i + 1] = self.operation[i]
                    self.operation[i] = o


def transform(A, B, pos, initial=False, date=0):
    global gold_allow
    a = A
    b = B
    for j in [0., 0.5, 1.]:
        for k in [0., 0.5, 1.]:
            o1 = np.array([j, k, 0])
            if initial == True:
                a.operation[pos] = o1

            flag = True
            if date not in gold_allow and j != 0:
                flag = False

            if flag == True:
                b.insert(utils.Operation(a.S[pos], o1, a.P, a.operation[pos]))

            flag = True
            ###############################
            o1 = np.array([j, k, 1])
            if date not in gold_allow:
                flag = False
            if initial == True:
                a.operation[pos] = o1
            if flag == True:
                b.insert(utils.Operation(a.S[pos], o1, a.P, a.operation[pos]))

            ###############################
            o1 = np.array([j, k, 2])
            if initial == True:
                a.operation[pos] = o1
            flag = True
            if date not in gold_allow and k != 0:
                flag = False

            if flag == True:
                b.insert(utils.Operation(a.S[pos], o1, a.P, a.operation[pos]))

    return b


if __name__ == "__main__":
    Days = []
    k = 8  # states num in one day
    t = 20  # predicted days
    h = 1.07  # predicted return limitation
    modify = 0.4
    for i in range(1825):
        Days.append(Day(k, np.array([1, Gold_price[i], Bitcoin_price[i]])))
    (Days[0].S)[0, 0] = 1000.

    strategy = []
    pred_y = np.zeros(1823)
    for i in range(1823):
        predicted_Days = []
        predicted_gold = np.load("prediction/gold{}.npy".format(i))
        predicted_bitcoin = np.load("prediction/bitcoin{}.npy".format(i))
        for j in range(i + 1, min(1824, i + t)):
            predicted_Days.append(Day(k, np.array(
                [1., modify * (Gold_price[j] - predicted_gold[j - i - 1]) + predicted_gold[j - i - 1],
                 modify * (Bitcoin_price[j] - predicted_bitcoin[j - i - 1]) + predicted_bitcoin[j - i - 1]])))
            predicted_Days[j - i - 1] = transform(Days[i], predicted_Days[j - i - 1], 0, initial=True, date=i)

        for j in range(i + 1, min(1824, i + t)):
            for r in range(j + 1, min(1824, i + t)):
                for p in range(k):
                    predicted_Days[r - i - 1] = transform(predicted_Days[j - i - 1], predicted_Days[r - i - 1],
                                                          p, date=j)
        temp = predicted_Days[-1]
        if utils.F(temp.S[0], temp.P) <= h * utils.F(Days[i].S[0], Days[i].P):
            temp.operation[0] = np.zeros(3)
        Days[i + 1].S[0] = utils.transform(Days[i].S[0], Days[i].P, temp.operation[0])
        pred_y[i] = utils.F(Days[i].S[0], Days[i].P)
        print(i, ":", Days[i].S[0], Days[i].P, temp.operation[0], utils.F(Days[i].S[0], Days[i].P))
        if temp.operation[0][0] != 0 or temp.operation[0][1] != 0:
            tt = np.zeros(4)
            tt[0] = i
            tt[1:] = temp.operation[0][:3]
            strategy.append(tt)

    np.save("prediction_.npy", pred_y)
    np.save("strategy_.npy", np.array(strategy))
    plt.title("return based on 20 predicted days")
    plt.xlabel("date")
    plt.ylabel("value")
    plt.plot(np.arange(1823), pred_y, color="green", label="profit")
    plt.legend()
    plt.show()

    '''
    # human
    if i == 100:
        Days[i+1].S[0] = transform(Days[i].S[0]. Days[i].P, np.array([112]))
    elif i ==440:
    elif i == 750:
    elif i==1000:
    elif i == 1500:
    elif i == 1650:
    elif i ==
    '''
