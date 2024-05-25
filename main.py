import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from bin_to_png import load_solution

if __name__ == '__main__':
    cmd = 'nvcc -o a.out laplace_solver.cu && ./a.out'
    os.system(cmd)
    os.system('rm ./a.out')
    
    sol, n, m = load_solution()    
    A, _, _ = load_solution('sle_matrix.bin')
    b, _, _ = load_solution('sle_rhs.bin')
    
    plt.figure(figsize=(12, 12), dpi=150)
    plt.title('Laplace solution')
    plt.imshow(sol, cmap=cm.jet)
    plt.savefig(f"laplace_solution_{n}.png")