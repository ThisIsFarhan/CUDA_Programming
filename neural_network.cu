#include <iostream>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>


__global__ void addBias(float *output, float *bias, int rows, int cols){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && col < cols)
        output[row * cols + col] += bias[col];
}

__global__ void applyReLU(float *input, float *output, int rows, int cols){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && col < cols){
    output[row * cols + col] = fmaxf(input[row * cols + col], 0.0f);
  }
}

__global__ void applySoftmax(float *input, float *output, int rows, int cols){
  int row = blockIdx.x;
  int col = threadIdx.x;
  extern __shared__ float shared_mem[]; //each block has its own shared memory

  if (row < rows && col < cols){

    //each thread load the data
    float x = input[row * cols + col];
    shared_mem[col] = x;
    __syncthreads();

    //each thread finds the largest value in the row
    float max_val = shared_mem[0];
    for (int i = 0 ; i < cols ; i++){
      if (shared_mem[i] > max_val){
        max_val = shared_mem[i];
      }
    }
    __syncthreads();

    //each thread subtract max and exponentiate each element of the row
    shared_mem[col] = expf(shared_mem[col] - max_val);
    __syncthreads();

    //each thread has the sum of the row
    float sum = 0.0f;
    for (int i = 0 ; i < cols ; i++){
      sum += shared_mem[i];
    }
    __syncthreads();

    //each thread divides each element by the sum of the respective row
    output[row * cols + col] = shared_mem[col] / sum;
  }
}

__global__ void softmax_backprop(float *dinputs, float *dvalues, int *y_true, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;

        // copy dvalues into dinputs
        dinputs[idx] = dvalues[idx];

        // subtract 1 at true class index first
        if (col == y_true[row])
            dinputs[idx] -= 1.0f;

        // then divide by rows
        dinputs[idx] /= rows;
    }
}

__global__ void reluBackward(float *dinputs, float *dvalues, float *inputs, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        // copy dvalues, but zero out where original input was <= 0
        dinputs[idx] = inputs[idx] > 0.0f ? dvalues[idx] : 0.0f;
    }
}

__global__ void colSumKernel(float *dvalues, float *dbiases, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int row = 0; row < rows; row++)
            sum += dvalues[row * cols + col];
        dbiases[col] = sum;
    }
}

__global__ void sgdUpdateKernel(float *params, float *dparams, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        params[idx] -= lr * dparams[idx];
}

class LayerDense{
  public:
    float *inputs;
    float *weights;
    float *bias;
    float *output;

    float *dweights;
    float *dbiases;
    float *dinputs;

    int n_inputs, n_neurons, batch_size;
    cublasHandle_t handle;

    LayerDense(int n_inputs, int n_neurons, int batch_size = 1):n_inputs(n_inputs), n_neurons(n_neurons), batch_size(batch_size){
      dweights = nullptr;
      dbiases  = nullptr;
      dinputs  = nullptr;

      cublasCreate(&handle);

      //W [n_inputs X n_neurons]
      //X [batch_size X n_inputs]
      // output [batch_size X n_neurons] = X[batch_size X n_inputs] * W[n_inputs X n_neurons]

      cudaMalloc(&weights, n_inputs * n_neurons * sizeof(float));
      cudaMalloc(&bias, n_neurons * sizeof(float));
      cudaMalloc(&output, batch_size * n_neurons * sizeof(float));

      //weights matrix initialize
      curandGenerator_t gen;
      curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
      curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
      curandGenerateNormal(gen, weights, n_inputs * n_neurons, 0.0f, 0.01f); // mean=0, std=0.01
      curandDestroyGenerator(gen);

      //bias initialization
      cudaMemset(bias, 0, n_neurons * sizeof(float));
    }

    void forward(float *input){
      inputs = input;
      const float alpha = 1.0f, beta = 0.0f;

      // weights^T  [n_neurons  x n_inputs  ]
      // input^T    [n_inputs   x batch_size]
      // output^T   [n_neurons  x batch_size]

      // We wanna do output = W^T * X^T
      cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N, // no transpose
            n_neurons,  // rows of W^T
            batch_size, // colums of X^T
            n_inputs,   // common dimension
            &alpha,
            weights, n_neurons, //leading dimension of W^T
            input,   n_inputs, //leading dimension of X^T
            &beta,
            output,  n_neurons //leading dimension of output^T
        );

      dim3 blocksize(16, 16);
      dim3 gridsize((n_neurons  + blocksize.x - 1) / 16, //cols
                  (batch_size  + blocksize.y - 1) / 16); //rows
      addBias<<<gridsize, blocksize>>>(output, bias, batch_size, n_neurons);
    }

    void backward(float *dvalues) {
      if (!dweights) cudaMalloc(&dweights, n_inputs  * n_neurons  * sizeof(float));
      if (!dbiases) cudaMalloc(&dbiases,  n_neurons * sizeof(float));
      if (!dinputs) cudaMalloc(&dinputs,  batch_size * n_inputs  * sizeof(float));

      const float alpha = 1.0f, beta = 0.0f;

      // 1-- dweights = inputs^T * dvalues
      // inputs^T  [n_inputs  x batch_size]  → A in cuBLAS
      // dvalues   [batch_size x n_neurons]  → B in cuBLAS
      // dweights  [n_inputs  x n_neurons]   → C in cuBLAS
      // cuBLAS: m=n_neurons, n=n_inputs, k=batch_size
      cublasSgemm(handle,
          CUBLAS_OP_N, CUBLAS_OP_T,   // transpose inputs
          n_neurons,   // m
          n_inputs,    // n
          batch_size,  // k
          &alpha,
          dvalues, n_neurons,  // A = dvalues^T
          inputs,  n_inputs,   // B = inputs (op=T gives inputs^T)
          &beta,
          dweights, n_neurons  // C = dweights^T
      );
      
      // 2-- dbiases = colwise sum of dvalues
      colSumKernel<<<(n_neurons + 255) / 256, 256>>>(dvalues, dbiases, batch_size, n_neurons);

      // 3-- dinputs = dvalues * weights^T
      // dvalues  [batch_size x n_neurons]   → A in cuBLAS
      // weights^T[n_neurons  x n_inputs]    → B in cuBLAS
      // dinputs  [batch_size x n_inputs]    → C in cuBLAS
      // cuBLAS: m=n_inputs, n=batch_size, k=n_neurons
      cublasSgemm(handle,
          CUBLAS_OP_T, CUBLAS_OP_N,   // transpose weights
          n_inputs,    // m
          batch_size,  // n
          n_neurons,   // k
          &alpha,
          weights,  n_neurons,  // A = weights (op=T gives weights^T)
          dvalues,  n_neurons,  // B = dvalues^T
          &beta,
          dinputs,  n_inputs    // C = dinputs^T
      );
  }
};

class ReLU{
  public:
    float *inputs;
    float *outputs;
    float *dinputs;

    ReLU() : outputs(nullptr), dinputs(nullptr) {}

    void forward(float *input, int rows, int cols){
      inputs = input;
      if (!outputs) cudaMalloc(&outputs, rows * cols * sizeof(float));
      dim3 blocksize(16, 16);
      dim3 gridsize((cols  + blocksize.x - 1) / blocksize.x, //cols
                  (rows  + blocksize.y - 1) / blocksize.y); //rows
      applyReLU<<<gridsize, blocksize>>>(input, outputs, rows, cols);
    }

    void backward(float *dvalues, int rows, int cols){
      if (!dinputs) cudaMalloc(&dinputs, rows * cols * sizeof(float));

      dim3 blocksize(16, 16);
      dim3 gridsize((cols + blocksize.x - 1) / blocksize.x,
                  (rows + blocksize.y - 1) / blocksize.y);

      reluBackward<<<gridsize, blocksize>>>(dinputs, dvalues, inputs, rows, cols);
    }
};

class Softmax{
  public:
    float *outputs;
    float *dinputs;

    Softmax() : outputs(nullptr), dinputs(nullptr) {}

    double forward(float *input, int *y_true, int rows, int cols) {
        if (!outputs) cudaMalloc(&outputs, rows * cols * sizeof(float));
        applySoftmax<<<rows, cols, cols * sizeof(float)>>>(input, outputs, rows, cols);
        cudaDeviceSynchronize();

        // copy outputs back to host for loss computation
        float *h_outputs = new float[rows * cols];
        cudaMemcpy(h_outputs, outputs, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

        // cross entropy loss — only look at the probability of the true class
        double epsilon = 1e-7;
        double loss = 0.0;
        for (int i = 0; i < rows; i++) {
            int idx = y_true[i];                          // true class index for this sample
            double prob = h_outputs[i * cols + idx];      // probability at true class
            double clipped = std::max(epsilon, std::min(1.0 - epsilon, prob));
            loss += -log(clipped);
        }

        return loss / rows;
    }

    void backward(float *dvalues, int *y_true, int rows, int cols){
      if (!dinputs) cudaMalloc(&dinputs, rows * cols * sizeof(float));
      // one block per row, one thread per col — same as softmax forward
      softmax_backprop<<<rows, cols>>>(dinputs, dvalues, y_true, rows, cols);
    }
};

class SGD{
  public:
    SGD(float lr = 1.0f, float decay_value = 0.0f) : learning_rate(lr), current_lr(lr), decay(decay_value), iteration(0) {}

    void pre_update_params(){
      if (decay > 0){
            current_lr = learning_rate  * (1.0f / (1.0f + decay * iteration));
        }
    }

    void update_params(LayerDense &layer){
      int weight_size = layer.n_inputs  * layer.n_neurons;
      int bias_size   = layer.n_neurons;

      sgdUpdateKernel<<<(weight_size + 255) / 256, 256>>>(layer.weights, layer.dweights, current_lr, weight_size);

      sgdUpdateKernel<<<(bias_size + 255) / 256, 256>>>(layer.bias, layer.dbiases, current_lr, bias_size);
    }

    void post_update_params(){
      iteration += 1;
    }
  private:
    double learning_rate;
    double current_lr;
    double decay;
    std::size_t iteration;
};


int main() {
    int batch_size = 4;
    int n_inputs   = 2;
    int hidden     = 32;
    int n_classes  = 2;

    // XOR input [4x2]
    float h_input[] = {0,0, 0,1, 1,0, 1,1};
    int h_y_true[]  = {1,0,0,1};

    // copy input to device
    float *d_input;
    cudaMalloc(&d_input, batch_size * n_inputs * sizeof(float));
    cudaMemcpy(d_input, h_input, batch_size * n_inputs * sizeof(float), cudaMemcpyHostToDevice);

    int *d_y_true;
    cudaMalloc(&d_y_true, batch_size * sizeof(int));
    cudaMemcpy(d_y_true, h_y_true, batch_size * sizeof(int), cudaMemcpyHostToDevice);

    // build network
    LayerDense dense1(n_inputs,  hidden,     batch_size);
    ReLU       relu1;
    LayerDense dense2(hidden,    n_classes,  batch_size);
    Softmax    softmax;
    SGD        optimizer(0.1f);

    // simple training loop
    for(int epoch = 0; epoch <= 10000; epoch++){
        // forward
        dense1.forward(d_input);
        relu1.forward(dense1.output, batch_size, hidden);
        dense2.forward(relu1.outputs);
        double loss = softmax.forward(dense2.output, h_y_true, batch_size, n_classes);

        // print loss every 1000 epochs
        if(epoch % 1000 == 0)
            std::cout << "Epoch " << epoch << " Loss: " << loss << "\n";

        // backward
        softmax.backward(softmax.outputs, d_y_true, batch_size, n_classes);
        dense2.backward(softmax.dinputs);
        relu1.backward(dense2.dinputs, batch_size, hidden);
        dense1.backward(relu1.dinputs);

        // update
        optimizer.pre_update_params();
        optimizer.update_params(dense1);
        optimizer.update_params(dense2);
        optimizer.post_update_params();
    }

    // --- Inference ---
    std::cout << "\nXNOR Predictions after Training:\n";
    
    // One last forward pass
    dense1.forward(d_input);
    relu1.forward(dense1.output, batch_size, hidden);
    dense2.forward(relu1.outputs);
    softmax.forward(dense2.output, h_y_true, batch_size, n_classes); // Loss return ignored

    // Copy the final probabilities from device to host
    float *h_final_probs = new float[batch_size * n_classes];
    cudaMemcpy(h_final_probs, softmax.outputs, batch_size * n_classes * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < batch_size; i++) {
        float prob0 = h_final_probs[i * n_classes + 0];
        float prob1 = h_final_probs[i * n_classes + 1];
        
        // Determine predicted class (0 or 1)
        int prediction = (prob1 > prob0) ? 1 : 0;
        
        std::cout << "Input: (" << h_input[i*2] << ", " << h_input[i*2+1] 
                  << ") | True: " << h_y_true[i] 
                  << " | Pred: " << prediction 
                  << " (Prob: " << (prediction == 1 ? prob1 : prob0) << ")\n";
    }

    delete[] h_final_probs;

    cudaFree(d_input);
    cudaFree(d_y_true);

    return 0;
}
