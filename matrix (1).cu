#include <stdio.h>

// Define matrix dimensions
#define N 10
#define M 10

// CUDA kernel to add matrices
__global__ void matrixAdd(int *a, int *b, int *c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M)
    {
        int index = row * M + col;
        c[index] = a[index] + b[index];
    }
}

int main()
{
    int a[N][M], b[N][M], c[N][M];
    int *d_a, *d_b, *d_c;
    int size = N * M * sizeof(int);

    // Initialize matrices a and b
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            a[i][j] = i + j;
            b[i][j] = i - j;
        }
    }

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy matrices from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimGrid(ceil((float)N / 16), ceil((float)M / 16), 1);
    dim3 dimBlock(16, 16, 1);

    // Launch the kernel
    matrixAdd<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);

    // Copy the result matrix from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the result matrix
    printf("Matrix C:\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    // Free allocated memory on the GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

