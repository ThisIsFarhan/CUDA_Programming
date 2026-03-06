#include<stdio.h>

__global__ void matrix_transpose(float* ip, float* op, int ROWS, int COLS){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < ROWS && y < COLS){
    op[y * ROWS + x] = ip[x * COLS + y];
  }
}


int main(){
  int rows = 2;
  int cols = 3;

  float* h_ip = (float*)malloc(rows*cols*sizeof(float));
  float* h_op = (float*)malloc(cols*rows*sizeof(float));

  for (int i = 0 ; i < rows*cols ; i++){
    h_ip[i] = (float)i+1;
  }

  float *d_ip, *d_op;
  cudaMalloc(&d_ip, rows*cols*sizeof(float));
  cudaMalloc(&d_op, cols*rows*sizeof(float));

  cudaMemcpy(d_ip, h_ip, rows*cols*sizeof(float), cudaMemcpyHostToDevice);

  dim3 blocksize(32,32);
  dim3 gridsize((rows + blocksize.x - 1)/blocksize.x, (cols + blocksize.y - 1)/blocksize.y);

  matrix_transpose<<<gridsize, blocksize>>>(d_ip, d_op, rows, cols);
  cudaMemcpy(h_op, d_op, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);

  printf("Input Matrix (%d x %d):\n", rows, cols);
  for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
          printf("%.1f  ", h_ip[i * cols + j]);
      }
      printf("\n");
  }

  printf("\nTransposed Matrix (%d x %d):\n", cols, rows);
  for (int i = 0; i < cols; i++) {
      for (int j = 0; j < rows; j++) {
          printf("%.1f  ", h_op[i * rows + j]);
      }
      printf("\n");
  }

  cudaFree(d_ip); 
  cudaFree(d_op);
  free(h_ip);  
  free(h_op);


}
