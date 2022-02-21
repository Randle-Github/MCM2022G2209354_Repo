import numpy as np

def Dataloader(start, end, file = None):
    full = np.load(file)
    return full[start: end]