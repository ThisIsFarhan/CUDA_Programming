#include<iostream>

__global__ void blockwise_prefix(int *ip, int *op, int n){
  extern __shared__ int shared_mem[];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x * 2 + tid;

  if (idx < n){
    //blockdim.x = 8 threads in each block
    shared_mem[tid] = ip[idx] + ip[idx+blockDim.x]; //0+9, 1+10, 2+11,......
    __syncthreads();

    // result: [0+9, 1+10,.......,7+15] ---> 8 sized array from 16 size
    for (int stride = 1; stride < blockDim.x ; stride *= 2){
      int temp = 0;
      if (tid >= stride){
        temp = shared_mem[tid - stride];
      }
      __syncthreads();
      shared_mem[tid] += temp;
      __syncthreads();
    }

    op[idx] = shared_mem[tid];
  }
}


int main(){
  int N = 16;
  int input[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
  int output[16] = {0};

  int *d_input, *d_output;

  cudaMalloc(&d_input, N * sizeof(int));
  cudaMalloc(&d_output, N * sizeof(int));

  cudaMemcpy(d_input, input, N * sizeof(int), cudaMemcpyHostToDevice);

  int blocksize = 8;
  int gridsize = N / blocksize;
  int shared_mem_size = blocksize * sizeof(int); //8 * 4 = 32 bytes of shared memory

  blockwise_prefix<<<gridsize, blocksize, shared_mem_size>>>(d_input, d_output, N);

  cudaMemcpy(output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0 ; i < N ; i++){
    std::cout << output[i] << " ";
  }
}
