import numpy as np


def Profitrate(s0, p0, s1, p1):
    # s0,p0(初始状态)，s1,p1(当前状态)
    return (np.dot(s1, p1) - np.dot(s0, p0)) / np.dot(s0, p0)


'''
def Accuracy(p[1:n],pre):
    # validation precision
    a1=mean((abs(pre(1,:)-p))/p)
    a2''''''
    return a
'''


def Maxwithdrawrate(s1, p1, p):
    '''
    p是后面的价格
    '''
    # s2(当前状态)，p[2:t](从当前到预测终点的所有价格)
    D = 0
    for pi in p:
        D = max(s1 * (p1 - pi) / (s1 * p1))
    return D


def Expectation(r, a, D, k):
    # e = r * a - D
    e = r - k * D
    return e


def F(s, p):
    """
    return the total price
    """
    return np.dot(s, p)


'''
def opration(pre,s1):
    #每次操作的最小单位为当前拥有量的10%
    #s1 当前状态
'''


def Operation(s1, o1, p1, o):
    '''
    s[0]:money s[1]:gold s[2]:bitcoin
    p[0]:money p[1]:gold p[2]:bitcoin
    type: 0:->money 1:-?gold 2:->bitcoin
    ratio[0]: 1->0 ratio[1]: 2->0 (type==0)
    '''
    s2 = np.zeros(np.shape(s1))
    if o1[2] == 0:
        s2[0] = s1[0] + 0.99 * o1[0] * s1[1] * p1[1] + 0.98 * o1[1] * s1[2] * p1[2]
        s2[1] = (1 - o1[0]) * s1[1]
        s2[2] = (1 - o1[1]) * s1[2]

    elif o1[2] == 1:
        s2[0] = (1 - o1[0]) * s1[0]
        s2[1] = s1[1] + 0.99 * o1[0] * s1[0] / p1[1] + 0.98 * 0.97 * o1[1] * s1[2] * p1[2] / p1[1]
        s2[2] = (1 - o1[1]) * s1[2]

    else:
        s2[0] = (1 - o1[0]) * s1[0]
        s2[1] = (1 - o1[1]) * s1[1]
        s2[2] = s1[2] + 0.98 * o1[0] * s1[0] / p1[2] + 0.98 * 0.97 * o1[1] * s1[1] * p1[1] / p1[2]

    return s2, o


def transform(s, p, o):
    temp = np.zeros(3)
    if o[2] == 0:
        temp[0] = s[0] + 0.99 * o[0] * s[1] * p[1] + 0.98 * o[1] * s[2] * p[2]
        temp[1] = (1 - o[0]) * s[1]
        temp[2] = (1 - o[1]) * s[2]

    elif o[2] == 1:
        temp[0] = (1 - o[0]) * s[0]
        temp[1] = s[1] + 0.99 * o[0] * s[0] / p[1] + 0.98 * 0.97 * o[1] * s[2] * p[2] / p[1]
        temp[2] = (1 - o[1]) * s[2]

    else:
        temp[0] = (1 - o[0]) * s[0]
        temp[1] = (1 - o[1]) * s[1]
        temp[2] = s[2] + 0.98 * o[0] * s[0] / p[2] + 0.97 * o[1] * s[1] * p[1] / p[2]

    return temp
