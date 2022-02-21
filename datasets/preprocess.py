import json
import utils.visual as vis
import matplotlib.pyplot as plt
import numpy as np


def reset_time_line():
    # set 9/11/16 as day0
    len_month = [20, 31, 30, 31,
                 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
                 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
                 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
                 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
                 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    timeline = 0
    _map = {}
    year, month, day = 16, 9, 10
    for days in len_month:
        for i in range(days):
            day += 1
            date = str(month) + "/" + str(day) + "/" + str(year)
            _map[date] = timeline
            timeline += 1
        day = 0
        month += 1
        if month == 13:
            month = 1
            year += 1
    with open("date_map.json", "w") as f:
        json.dump(_map, f)
    return _map

if __name__ == "__main__":
    reset_time_line()
    with open("date_map.json", "r") as f:
        _map = json.load(f)
    x = []
    y = []
    option = "LBMA-GOLD.txt"
    if option == "LBMA-GOLD.txt":
        with open("LBMA-GOLD.txt", "r") as f:
            content = f.readlines()
            for line in content[1:-1]:
                date, temp = line.split(",")
                if temp == "\n" or temp.rstrip("\n") == " ":
                    continue
                x.append(_map[date])
                y.append(float(temp.rstrip("\n")))
        np.save("gold_allow.npy", np.array(x))
        vis.visualise(x, y, "orange", "gold")
        plt.xlabel("date code")
        plt.ylabel("dollar/ounce")
        plt.title("Gold Price")
        plt.legend()
        plt.show()
    else:
        with open("BCHAIN-MKPRU.txt", "r") as f:
            content = f.readlines()
            for line in content[1:-1]:
                date, temp = line.split(",")
                if temp == "\n" or temp.rstrip("\n") == " ":
                    continue
                x.append(_map[date])
                y.append(float(temp.rstrip("\n")))
        vis.visualise(x, y, "green", "bitcoin")
        plt.xlabel("date code")
        plt.ylabel("dollar/unit")
        plt.title("Bitcoin Price")
        plt.legend()
        plt.show()