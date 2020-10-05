import numpy as np

def readFile(fileName, dtype, delimiter):
    matrix = np.loadtxt(fileName, dtype= dtype, delimiter= delimiter)
    return matrix



