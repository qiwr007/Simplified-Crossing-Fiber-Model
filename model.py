import numpy as np
from numpy import cos, sin, pi, exp, sqrt
from scipy import integrate
from scipy.optimize import fsolve
import math


def rotation_matrix(theta, phi):
    # create rotation matrix
    return np.array([[cos(theta)*cos(phi), -1*sin(theta), sin(theta)*cos(phi)],
                    [cos(theta)* sin(phi), cos(phi), sin(theta)*sin(phi)],
                    [sin(theta), 0, -1 * cos(theta)]])


def BehrensSim (S0, f, bval, bvecs, d, theta, phi):
    if len(f) != len(theta) or len(f) != len(phi):
        print("Warning: parameters involving the number of fibers do not have the same length")
        return

    A_mat = np.array([[1,0,0],[0,0,0],[0,0,0]])
    ball_comp = (1 - sum(f)) * np.exp(-1 * bval * d)

    stick_comp = np.zeros(len(bval))
    for k in range(len(f)):
        theta_k = theta[k] * pi /180
        phi_k = phi[k] * pi /180

    Si = S0 * (ball_comp + stick_comp)
    return Si


def error_function(x):
    return exp(-1 * x * x)


def erf(x):
    f = lambda x: exp(-1 * x * x)
    v, err = integrate.quad(f, -x, x)
    return v


def solve(si, bi): # data is several pairs of Si and bi
    s_max = np.amax(si)
    s_avg = np.average(si)
    s0 = si[0]  # the average of Si when bi = 0

    def equations(p):
        # list the set of the two equations, Eq(3) and Eq(4) in paper
        sum_f, d = p
        return s0 * ((1 - sum_f) * exp(-bi * d) + sum_f * sqrt(pi) * erf(sqrt(bi * d)) / sqrt(bi * d)) - s_avg, \
               s0 * ((1 - sum_f) * exp(-1 * bi * d) + sum_f) - s_max

    sum_f, d = fsolve(equations, (1, 1))
    return sum_f, d


if __name__ == "__main__":
    si = np.array([1, 2, 3, 5, 1, 2, 10])
    bi = 100
    aa, bb = solve(si, bi)
    print(aa)
    print(bb)
