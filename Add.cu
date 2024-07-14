#include <stdio.h>


// CUDA kernel to add numbers
__global__ void matrixAdd(int *a, int *b, int *c)
{
    *c = *a + *b;
}

int main()
{
    int a=5, b=7, c;
    int *d_a, *d_b, *d_c;
    int size = N * M * sizeof(int);
    

    // Allocate memory on the GPU
    cudaMalloc((void *)&d_a, size);
    cudaMalloc((void *)&d_b, size);
    cudaMalloc((void *)&d_c, size);

    // Copy matrices from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);


    // Launch the kernel
    matrixAdd<<<1, 1>>>(d_a, d_b, d_c);

    // Copy the result matrix from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the result
    printf("%d ", c);
    
    // Free allocated memory on the GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

