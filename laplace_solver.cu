#include <stdio.h>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__host__ void save_to_file(float *data, int size, float n, float m, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Unable to open a file");
        return;
    }
    fwrite(&n, sizeof(float), 1, file);
    fwrite(&m, sizeof(float), 1, file);
    fwrite(data, sizeof(float), size, file);
    fclose(file);
}

// get SLE for discretized laplacian
__global__ void fill_the_sle(int n, int m, 
                            float x0, float y0, float x1, float y1, 
                            float *sle_matrix, float *sle_rhs) {
    // int global_id = (blockIdx.x * gridDim.y) * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x;

    int x = blockIdx.x;
    int y = blockIdx.y * blockDim.x + threadIdx.x;

    // redundant (block_dim - M % block_dim) threads
    if (y > m)
        return;

    int flatten_idx = x * m + y; // variable index [0, n*m] 
    int sle_size = n * m;

    sle_matrix[flatten_idx * sle_size + flatten_idx] = -4;
    
    if (x == 0) {
        sle_rhs[flatten_idx] -= x0;
    } else {
        int neighbor_idx = (x - 1) * m + y;
        sle_matrix[flatten_idx * sle_size + neighbor_idx] = 1;
    }

    if (x == n - 1) {
        sle_rhs[flatten_idx] -= x1;
    } else {
        int neighbor_idx = (x + 1) * m + y;
        sle_matrix[flatten_idx * sle_size + neighbor_idx] = 1;
    }

    if (y == 0) {
        sle_rhs[flatten_idx] -= y0;
    } else {
        int neighbor_idx = flatten_idx - 1;
        sle_matrix[flatten_idx * sle_size + neighbor_idx] = 1;
    }

    if (y == m - 1) {
        sle_rhs[flatten_idx] -= y1;
    } else {
        int neighbor_idx = flatten_idx + 1;
        sle_matrix[flatten_idx * sle_size + neighbor_idx] = 1;
    }
}


// elimination of a single column 
__global__ void ge_step(int pivot_row, int n, float *matrix, float *rhs, int backward, float *factor_col) {
    int x = blockIdx.x;
    int y = blockIdx.y * blockDim.x + threadIdx.x;

    if ((y > n) || ((backward == 1) && (x >= pivot_row)) || ((backward == 0) && (x <= pivot_row)))
        return;

    float factor = factor_col[x] / matrix[pivot_row*n + pivot_row];
    matrix[x*n + y] -= matrix[pivot_row*n + y] * factor; 

    if (y == n - 1)
        rhs[x] -= rhs[pivot_row] * factor;

}

// matrix – identity, rhs – roots
__global__ void ge_finish(int n, float *matrix, float *rhs) {
    int x = blockIdx.x;
    rhs[x] = rhs[x] / matrix[x*n + x];
    matrix[x*n + x] = 1;
}

__global__ void get_factor_col(float *matrix, int col, int n, float *d_out) {
    d_out[blockIdx.x] = matrix[blockIdx.x*n + col];
}


// gaussian elimination. kernel-splitting approach
__host__ void gauss_elimination(int n, float *matrix, float *rhs, int max_threads) {
    int blocks_per_row = n / max_threads +(((n % max_threads)) != 0);
    int threads_per_block = max_threads;
    if (n < max_threads)
        threads_per_block = n;

    printf("Grid dim: %d %d, threads per block: %d\n", n, blocks_per_row, threads_per_block);

    float *factor_col = 0;
    gpuErrchk(cudaMalloc((void**)&factor_col, n * sizeof(float)));

    for (int i = 0; i < n-1; i++) {
        get_factor_col<<<n, 1>>>(matrix, i, n, factor_col);
        cudaDeviceSynchronize();

        ge_step<<<dim3(n, blocks_per_row, 1), threads_per_block>>>(i, n, matrix, rhs, 0, factor_col);
        cudaDeviceSynchronize();
    }

    for (int i = n-1; i > 0; i--) {
        get_factor_col<<<n, 1>>>(matrix, i, n, factor_col);
        cudaDeviceSynchronize();

        ge_step<<<dim3(n, blocks_per_row, 1), threads_per_block>>>(i, n, matrix, rhs, 1, factor_col);
        cudaDeviceSynchronize();
    }

    ge_finish<<<n, 1>>>(n, matrix, rhs);

}

__host__ void print_sle(int n, float *matrix, float *rhs) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f\t", matrix[i*n + j]);
        }
        printf("|\t%.2f\n", rhs[i]);
    }
}

int main() {
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    int max_threads = prop.maxThreadsPerBlock;

    cudaSetDevice(0);
    int N = 100;
    int M = 100;
    int x0 = 0, x1 = 0, y0 = 1, y1 = 0;
    int sle_size = N * M;

    int blocks_per_row = M / max_threads + ((M % max_threads) != 0);
    int threads_per_block = max_threads;
    float *sle_matrix = 0; 
    float *sle_rhs = 0;

    if (M < max_threads)
        threads_per_block = M;

    gpuErrchk(cudaMalloc((void**)&sle_matrix, sle_size * sle_size * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&sle_rhs, sle_size * sizeof(float)));
    gpuErrchk(cudaMemset(sle_matrix, 0, sle_size * sle_size * sizeof(float)));
    gpuErrchk(cudaMemset(sle_rhs, 0, sle_size * sizeof(float)));

    cudaDeviceSynchronize();
    fill_the_sle<<<dim3(N, blocks_per_row, 1), threads_per_block>>>(N, M, x0, y0, x1, y1, sle_matrix, sle_rhs);
    
    float *h_matrix = (float *) malloc(sizeof(float) * sle_size * sle_size);
    float *h_rhs = (float *) malloc(sizeof(float) * sle_size);

    cudaDeviceSynchronize();
    gpuErrchk(cudaMemcpy(h_matrix, sle_matrix, sle_size * sle_size * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_rhs, sle_rhs, sle_size * sizeof(float), cudaMemcpyDeviceToHost));

    // printf("Obtained SLE (A|b): \n");
    // print_sle(sle_size, h_matrix, h_rhs);
    save_to_file(h_matrix, sle_size*sle_size, sle_size, sle_size, "sle_matrix.bin");
    save_to_file(h_rhs, sle_size, sle_size, 1, "sle_rhs.bin");

    gauss_elimination(sle_size, sle_matrix, sle_rhs, max_threads);

    cudaDeviceSynchronize();
    gpuErrchk(cudaMemcpy(h_matrix, sle_matrix, sle_size * sle_size * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_rhs, sle_rhs, sle_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // printf("Solved SLE: \n");
    // print_sle(sle_size, h_matrix, h_rhs);

    save_to_file(h_rhs, sle_size, N, M, "laplace_solution.bin");

    cudaFree(sle_matrix);
    cudaFree(sle_rhs);
    free(h_matrix);
    free(h_rhs);

    return 0;
}
