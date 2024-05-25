import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy
import os

def load_solution(filename = "laplace_solution.bin"):
    res = np.fromfile(filename, dtype=np.float32)
    n, m, sol = int(res[0]), int(res[1]), res[2:]
    return sol.reshape((n, m)), n, m

if __name__ == '__main__':
    sol, n, m = load_solution()    
    A, _, _ = load_solution('sle_matrix.bin')
    b, _, _ = load_solution('sle_rhs.bin')
    
    plt.figure(figsize=(12, 12), dpi=150)
    plt.title('Laplace solution')
    plt.imshow(sol, cmap=cm.jet)
    plt.savefig(f"laplace_solution_{n}.png")