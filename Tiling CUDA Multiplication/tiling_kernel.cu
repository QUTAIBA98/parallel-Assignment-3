#define TILE_SIZE 16

__global__ 
void tiling_kernel(const long int* matrix1, const long int* matrix2,
                    long int* result, int n_row1, int n_row2, int n_col2) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ long int tile1[TILE_SIZE][TILE_SIZE];
    __shared__ long int tile2[TILE_SIZE][TILE_SIZE];

    long int sum = 0;

    for (int t = 0; t < (n_row2 + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < n_row1 && t * TILE_SIZE + threadIdx.x < n_row2)
            tile1[threadIdx.y][threadIdx.x] = matrix1[row * n_row2 + t * TILE_SIZE + threadIdx.x];
        else
            tile1[threadIdx.y][threadIdx.x] = 0;

        if (col < n_col2 && t * TILE_SIZE + threadIdx.y < n_row2)
            tile2[threadIdx.y][threadIdx.x] = matrix2[(t * TILE_SIZE + threadIdx.y) * n_col2 + col];
        else
            tile2[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            sum += tile1[threadIdx.y][k] * tile2[k][threadIdx.x];

        __syncthreads();
    }

    if (row < n_row1 && col < n_col2)
        result[row * n_col2 + col] = sum;
}
