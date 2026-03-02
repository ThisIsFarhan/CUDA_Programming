#include<iostream>
#include<cmath>

__global__ void layer_norm(float *ip, float *op, int rows, int cols){

  int row = blockIdx.x;
  int col = threadIdx.x;
  extern __shared__ float shared_mem[];

  if (row < rows && col < cols){

    //each thread load the data
    float x = ip[row * cols + col];
    shared_mem[col] = x;
    __syncthreads();

    //each thread (col) inside a block (row) computes the mean for that row
    float mean = 0.0f;
    for (int i = 0 ; i < cols ; i++){
      mean += shared_mem[i];
    }
    mean /= cols;
    __syncthreads();

    // variance calculation
    float diff = x - mean;
    float var_term = diff*diff;
    shared_mem[col] = var_term;
    __syncthreads();

    float var = 0.0f;
    for (int i = 0 ; i < cols ; i++){
      var += shared_mem[i];
    }
    var /= cols;
    float inverse_std_dev = rsqrtf(var + 1e-7);

    op[row * cols + col] = diff * inverse_std_dev;
  }
}

int main(){
  int rows = 2;
  int cols = 4;

  float h_ip[rows * cols] = {
    1,2,3,4,
    2,4,6,8
  };

  float h_op[rows * cols] = {0};

  float *d_ip, *d_op;

  cudaMalloc(&d_ip, rows*cols*sizeof(float));
  cudaMalloc(&d_op, rows*cols*sizeof(float));

  cudaMemcpy(d_ip, h_ip, rows*cols*sizeof(float), cudaMemcpyHostToDevice);

  dim3 grid(rows);
  dim3 block(cols);

  size_t shared_mem_size = cols*sizeof(float);

  layer_norm<<<grid, block, shared_mem_size>>>(d_ip, d_op, rows, cols);
  cudaMemcpy(h_op, d_op, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Input:\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << h_ip[i * cols + j] << " ";
        }
        std::cout << "\n";
    }

  std::cout << "\n";

  std::cout << "LayerNorm Output:\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << h_op[i * cols + j] << " ";
        }
        std::cout << "\n";
    }

    cudaFree(d_ip);
    cudaFree(d_op);

    return 0;
}
