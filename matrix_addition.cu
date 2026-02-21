#include <iostream>
#include <cmath>

__global__ void matrix_add(const float* A, const float* B, float* C, int rows, int cols){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < rows && j < cols){
    int index = i * cols + j;
    C[index] = A[index] + B[index];
  }
}

int main(){
  int rows = 3;
  int cols = 3;

  int N = rows * cols;

  float* A = new float[N];
  float* B = new float[N];
  float* C = new float[N];

  for (int i = 0 ; i < N ; i++){
    A[i] = (float)i + 1.0f;
    B[i] = 2.0f;
    C[i] = 0.0f;
  }

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, N * sizeof(float));
  cudaMalloc(&d_b, N * sizeof(float));
  cudaMalloc(&d_c, N * sizeof(float));

  cudaMemcpy(d_a, A, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, N*sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(32, 16);
  dim3 gridDim((int)ceil((float)N/blockDim.x),(int)ceil((float)N/blockDim.y));
  matrix_add<<<gridDim, blockDim>>>(d_a, d_b, d_c, rows, cols);
  cudaMemcpy(C, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "A: \n";
  for (int i = 0 ; i < N ; i++){
    std::cout << A[i] << " ";
  }

  std::cout << "\nB: \n";
  for (int i = 0 ; i < N ; i++){
    std::cout << B[i] << " ";
  }

  std::cout << "\nC: \n";
  for (int i = 0 ; i < N ; i++){
    std::cout << C[i] << " ";
  }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  delete[] A;
  delete[] B;
  delete[] C;
  
}
