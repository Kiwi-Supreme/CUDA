int main() {
    // Host arrays and size
    int size = 1024;
    int *h_a, *h_b, *h_c;
    h_a = new int[size];
    h_b = new int[size];
    h_c = new int[size];

    // Initialize host arrays h_a and h_b

    // Device arrays
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size * sizeof(int));
    cudaMalloc((void**)&d_b, size * sizeof(int));
    cudaMalloc((void**)&d_c, size * sizeof(int));

    // Copy host data to device
    cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_c, d_a, d_b, size);

    // Copy result from device to host
    cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
