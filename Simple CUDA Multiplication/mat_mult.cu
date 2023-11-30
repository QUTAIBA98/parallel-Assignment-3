#include <stdio.h>
#include <fstream>
#include <iostream>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "mat_mult_kernel.cu"

#define BILLION  1000000000.0				// For clock_gettime()
#define BLOCK_SIZE 1024                     // BLOCK Size of CUDA Thread grid

/* Function Declarations */
void err_check(cudaError_t ret, char* msg, int exit_code);

int main(int argc, char* argv[])
{
    // Catch command line errors
    if (argc != 8)
    {
        printf("Usage: ./mat_mult  mat_a.csv n_row_1 n_col_1   mat_b.csv n_row_2 n_col_2  results_matrix.csv\n");
        return EXIT_FAILURE;
    }
    printf("\nNOTE: For correct results, entered values of n_Rows and n_Cols of\n");
    printf("matrices must match the dimensions of matrices in .csv input files\n");

    // Timing and error variables
    struct timespec start, end;
    cudaError_t cuda_ret;

    // Get input files from the command line and open for reading
    FILE* inputMatrix1 = fopen(argv[1], "r");
    FILE* inputMatrix2 = fopen(argv[4], "r");

    // Get dimensions of 1st matrix from command line
    int n_row1 = strtol(argv[2], NULL, 10);
    int n_col1 = strtol(argv[3], NULL, 10);

    // Get dimensions of 2nd matrix from command line
    int n_row2 = strtol(argv[5], NULL, 10);
    int n_col2 = strtol(argv[6], NULL, 10);

    // Get name of output file from command line and open for writing
    FILE* outputFile = fopen(argv[7], "w");

    // Initialize the two input matrices and the resultant matrix
    long int* matrix1 = (long int*)malloc((n_row1 * n_col1) * sizeof(long int));
    long int* matrix2 = (long int*)malloc((n_row2 * n_col2) * sizeof(long int));
    long int* result = (long int*)malloc((n_row1 * n_col2) * sizeof(long int));

    // Parse the input csv files and fill in the input matrices
    for (int i = 0; i < n_row1; i++) {
        for (int j = 0; j < n_col1; j++) {
            fscanf(inputMatrix1, "%ld,", &matrix1[i * n_col1 + j]);
        }
    }
    for (int i = 0; i < n_row2; i++) {
        for (int j = 0; j < n_col2; j++) {
            fscanf(inputMatrix2, "%ld,", &matrix2[i * n_col2 + j]);
        }
    }
    
    // For use with CUDA kernel
    // Set CUDA grid and block dimensions
    int num_blocks = ceil((float)(n_row1 * n_col2) / (float)BLOCK_SIZE);
    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    // Allocate memory for matrix1 on device
    long int* device_matrix1;
    cuda_ret = cudaMalloc((void**)&device_matrix1, (n_row1 * n_col1) * sizeof(long int));
    err_check(cuda_ret, (char*)"Unable to allocate matrix1 to device memory!", 1);

    // Copy matrix1 to device memory
    cuda_ret = cudaMemcpy(device_matrix1, matrix1, (n_row1 * n_col1) * sizeof(long int), cudaMemcpyHostToDevice);
    err_check(cuda_ret, (char*)"Unable to read matrix1 from host memory!", 2);

    // Allocate memory for matrix2 on device
    long int* device_matrix2;
    cuda_ret = cudaMalloc((void**)&device_matrix2, (n_row2 * n_col2) * sizeof(long int));
    err_check(cuda_ret, (char*)"Unable to allocate matrix2 to device memory!", 3);

    // Copy matrix2 to device memory
    cuda_ret = cudaMemcpy(device_matrix2, matrix2, (n_row2 * n_col2) * sizeof(long int), cudaMemcpyHostToDevice);
    err_check(cuda_ret, (char*)"Unable to read matrix2 from host memory!", 4);

    // Allocate memory for result on device
    long int* device_result;
    cuda_ret = cudaMalloc((void**)&device_result, (n_row1 * n_col2) * sizeof(long int));
    err_check(cuda_ret, (char*)"Unable to allocate result to device memory!", 5);

    // Start clock
    clock_gettime(CLOCK_REALTIME, &start);

    // Launch mat_mult_kernel
    mat_mult_kernel << <dimGrid, dimBlock>> > (
        device_matrix1,         // matrix 1 on device
        device_matrix2,         // matrix 2 on device
        device_result,          // for storing results on device
        n_row1,                 // number of rows of Mat 1
        n_row2,                 // number of rows of Mat 2
        n_col2);                // number of cols of Mat 2

    // Synchronize threads and check for error 
    cuda_ret = cudaDeviceSynchronize();
    err_check(cuda_ret, (char*)"Unable to launch mat_mult_kernel!", 6);

    // Stop clock
    clock_gettime(CLOCK_REALTIME, &end);

    // Copy the final results from the device to host
    cuda_ret = cudaMemcpy(result, device_result, (n_row1 * n_col2) * sizeof(long int), cudaMemcpyDeviceToHost);
    err_check(cuda_ret, (char*)"Unable to read final results matrix from device memory!", 7);

    // Write the resultant matrix to the results file
    for (int i = 0; i < n_row1; i++) {
        for (int j = 0; j < n_col2; j++) {
            fprintf(outputFile, "%ld,", result[i * n_col2 + j]);
        }
        fprintf(outputFile, "\n");
    }

    // Calculate elapsed time
    double ElapsedTime = (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) / BILLION;

    // Print the execution time
    printf("\nDone! Please check output files for result.\n\t Kernel execution time:  %f sec\n\n", ElapsedTime);

    // Close the files
    fclose(inputMatrix1);
    fclose(inputMatrix2);
    fclose(outputFile);

    // Free memory
    free(matrix1);
    free(matrix2);
    free(result);
    cudaFree(device_matrix1);
    cudaFree(device_matrix2);
    cudaFree(device_result);

	return 0;
}

/* Function definitions */
/* Error Check ----------------- //
*   Exits if there is a CUDA error.
*/
void err_check(cudaError_t ret, char* msg, int exit_code) {
	if (ret != cudaSuccess)
		fprintf(stderr, "%s \"%s\".\n", msg, cudaGetErrorString(ret)),
		exit(exit_code);
} // End Error Check ----------- //