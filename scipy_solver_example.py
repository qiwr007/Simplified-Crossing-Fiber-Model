from scipy.optimize import fsolve
import math

def solve():
    def equations(p):
        x, y = p
        return x+y**2-4, math.exp(x) + x*y - 3
    x, y = fsolve(equations, (1, 1))
    return x, y


if __name__ == "__main__":
    print(solve())


