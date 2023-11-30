/* Kernel for implementing matrix matrix multiplication */

__global__ 
void mat_mult_kernel(long int* matrix1, long int* matrix2, 
                long int* result, int n_row1, int n_row2, int n_col2) 
{
    // Determine thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;


    if (tid < (n_row1 * n_col2)) 
    {
        int row = tid / n_col2;     // Determine row
        int col = tid % n_col2;     // Determine column
        long int sum = 0.0f;
        for (int k = 0; k < n_row2; k++) {
            sum += matrix1[row * n_row2 + k] * matrix2[k * n_col2 + col];
        }
        result[tid] = sum;
    }
}
/* END mat_mult_kernel */