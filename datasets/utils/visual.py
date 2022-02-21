import numpy as np
import matplotlib.pyplot as plt


def visualise(x_axis, y_axis, color, label):
    plt.plot(x_axis, y_axis, c=color, label=label)
