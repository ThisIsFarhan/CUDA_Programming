#include<iostream>
#include<cmath>
#include<chrono>
#include<cuda_runtime.h>

__global__ void vec_add(const float* A, const float* B, float* C, int N){
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N){
    C[i] = A[i] + B[i];
  }
}

void run_vec_add(int N){
  //int N = 1 << 22;

  // Initialize arrays A, B, C on CPU
  //float A[N], B[N], C[N], C_CPU[N];
  float* A     = new float[N];
  float* B     = new float[N];
  float* C     = new float[N];
  float* C_CPU = new float[N];

  for(int i = 0; i < N ; i++){
    A[i] = (float)i + 1.0f;
    B[i] = 2.0f;
  }

  auto cpu_start = std::chrono::high_resolution_clock::now();
  for (int i = 0 ; i < N ; i++){
    C_CPU[i] = A[i] + B[i];
  }
  auto cpu_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
  std::cout << "CPU time (ms): " << cpu_time.count() << std::endl;

  // memory allocations A, B, C on GPU
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, N*sizeof(float));
  cudaMalloc(&d_b, N*sizeof(float));
  cudaMalloc(&d_c, N*sizeof(float));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // copy the contents of the arrays on CPU to allocated memory on the GPU
  cudaMemcpy(d_a, A, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, N*sizeof(float), cudaMemcpyHostToDevice);

  int blocksize = 256;
  int gridsize = (int)ceil((float)N/blocksize); //1
  vec_add<<<gridsize, blocksize>>>(d_a, d_b, d_c, N);

  cudaMemcpy(C, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float gpu_total_time = 0.0f;
  cudaEventElapsedTime(&gpu_total_time, start, stop);

  std::cout << "GPU total time (ms): " << gpu_total_time << std::endl;

  /*std::cout << "Vec A: " << std::endl;
  for(int i = 0 ; i<N ; i++){
    std::cout << A[i] << " ";
  }
  std::cout << "\n";

  std::cout << "Vec B: " << std::endl;
  for(int i = 0 ; i<N ; i++){
    std::cout << B[i] << " ";
  }
  std::cout << "\n";


  std::cout << "Vec C (GPU): " << std::endl;
  for(int i = 0 ; i<N ; i++){
    std::cout << C[i] << " ";
  }
  std::cout << "\n";

  std::cout << "Vec C (CPU): " << std::endl;
  for(int i = 0 ; i<N ; i++){
    std::cout << C_CPU[i] << " ";
  }
  std::cout << "\n";*/

  
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] C_CPU;
}


int main() {
    // Define multiple vector sizes to compare
    int sizes[] = { 1024, 1 << 20, 1 << 22, 1 << 23 }; // 1K, 1M, 4M, 8M
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    std::cout << "CPU vs GPU vector addition times:\n";
    std::cout << "---------------------------------\n";

    for (int i = 0; i < num_sizes; i++) {
        int N = sizes[i];
        std::cout << "\nVector size: " << N << "\n";
        run_vec_add(N);
    }

    return 0;
}

