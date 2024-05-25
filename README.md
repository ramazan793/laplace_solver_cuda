# CUDA Laplace Solver

This project provides a CUDA-based implementation for solving the Laplace equation. The Laplace equation is a second-order partial differential equation widely used in various fields such as physics and engineering.

## Laplace Equation

The Laplace equation in two dimensions is given by:

$$
\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0
$$

with the following boundary conditions:

$$
\begin{cases}
u(x=0, y) = x0 \\
u(x=1, y) = x1 \\
u(x, y=0) = y0 \\
u(x, y=1) = y1
\end{cases}
$$

![laplace_solution_100_init_0](https://github.com/ramazan793/laplace_solver_cuda/assets/25179317/7bfd667a-077f-4a4b-a665-84228b5364a9)
![laplace_solution_100_init_2](https://github.com/ramazan793/laplace_solver_cuda/assets/25179317/d02e824c-c9a7-45f9-8de0-bfa741539778)
![laplace_solution_100_init_1](https://github.com/ramazan793/laplace_solver_cuda/assets/25179317/e28bf8ab-269a-4d4b-8f46-aab3753e4c91)

## Implementation

The solution uses the finite difference method to discretize the Laplace equation on a grid and solve it iteratively until convergence. CUDA is used to accelerate the computation by leveraging the parallel processing capabilities of GPUs.

### Requirements

- CUDA Toolkit
- C++ Compiler
- Python (for visualization)

### Runnng
```
python main.py
```
OR
```
nvcc -o a.out laplace_solver.cu
./a.out
python bin_to_png.py
```
