#include<iostream>
#include<cmath>

__global__ void matrix_multiply(const float *A, const float *X, float *B, int A_ROWS, int A_COLS){
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < A_ROWS){
    float sum = 0.0f;
    for (int j = 0 ; j < A_COLS ; j++){
      sum += A[row * A_COLS + j] * X[j];
    }
    B[row] = sum;
  }
}


int main(){
  int A_ROWS = 3;
  int A_COLS = 3;

  int X_ROWS = 3;
  int X_COLS = 1;

  float *A = new float[A_ROWS * A_COLS];
  float *X = new float[X_ROWS * X_COLS];
  float *B = new float[A_COLS * X_COLS];

  for (int i = 0 ; i < A_ROWS * A_COLS ; i++){
    A[i] = (float)i + 1.0f;
    if (i < X_ROWS * X_COLS){
      X[i] = 2.0f;
    }
  }

  for (int i = 0 ; i < A_COLS * X_COLS ; i++){
    B[i] = 0.0f;
  }

  float *d_a, *d_b, *d_x;

  cudaMalloc(&d_a, A_ROWS * A_COLS * sizeof(float));
  cudaMalloc(&d_x, X_ROWS * X_COLS * sizeof(float));
  cudaMalloc(&d_b, A_COLS * X_COLS * sizeof(float));

  cudaMemcpy(d_a, A, A_ROWS * A_COLS * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, X, X_ROWS * X_COLS * sizeof(float), cudaMemcpyHostToDevice);

  int blocksize = 256;
  int gridsize = (int)ceil((float)A_ROWS/blocksize); //1
  matrix_multiply<<<gridsize, blocksize>>>(d_a, d_x, d_b, A_ROWS, A_COLS);
  cudaMemcpy(B, d_b, A_COLS * X_COLS * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0 ; i < A_COLS * X_COLS ; i++){
    std::cout << B[i] << " ";
  }
}
