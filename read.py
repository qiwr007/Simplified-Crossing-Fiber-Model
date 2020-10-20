import numpy as np

# Use NumPy to read the matrix from file
def readFile1(fileName, dtype, delimiter):
    matrix = np.loadtxt(fileName, dtype= dtype, delimiter= delimiter)
    return matrix

def readFile(fileName):
    matrix = np.loadtxt(fileName)
    return matrix

