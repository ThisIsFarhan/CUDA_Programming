#include<stdio.h>
#define TILE_DIM 32

__global__ void matrix_transpose(float* ip, float* op, int ROWS, int COLS){
  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x; //cols_idx
  int y = blockIdx.y * TILE_DIM + threadIdx.y; //rows_idx

  //writing to tile
  if (x < COLS && y < ROWS){
    tile[threadIdx.y][threadIdx.x] = ip[y * COLS + x]; //ip[row * COLS + col]
  }
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x; //col in output
  y = blockIdx.x * TILE_DIM + threadIdx.y; //row in output

  //writing to global
  if (x < ROWS && y < COLS){
    op[y * ROWS + x] = tile[threadIdx.x][threadIdx.y];
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
  dim3 gridsize((cols + blocksize.x - 1)/blocksize.x, (rows + blocksize.y - 1)/blocksize.y);

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
