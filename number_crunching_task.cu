#include <algorithm>
#include <fstream>
#include <iostream>
#include <chrono>
__global__ void function_a_gpu(double *u, double *v, double *w, const int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  w[tid] = 0;
  for(unsigned int i = tid; i < N; i += stride)
  {
    w[tid] += u[i] * v[i];
  }
}

__global__ void sum_func_a_result(const double *result, double *s, const int N){
  for (unsigned int i = 0; i < N; i++) {
    s[0] += result[i];
  }
}

__global__ void function_b_gpu(double *u, double *v, double *result, const int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for(unsigned int i = tid; i < N; i += stride)
  {
    result[i] = u[i] + v[i];
  }
}

__global__ void function_c_gpu(double *A, double *x, double *y, const int N) { 
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for(unsigned int i = tid; i < N; i += stride){
    y[i] = 0;
  }
  for (unsigned int i = tid; i < N; i+= stride) {
    for (unsigned int j = 0; j < N; j++) {
      y[i] += A[i * N + j] * x[i];
    }
  }
}

__global__ void function_c_gpu_shareMem(double *A, double *x, double *y, const int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  extern __shared__ double sharedMem[];
  double* y_d = sharedMem;
  double* x_d = sharedMem+N;
  for (unsigned int i = tid; i < N; i+= stride) {
    y_d[tid] = 0;
    x_d[tid] = x[i];
  }
  __syncthreads();
  for (unsigned int i = tid; i < N; i+= stride) {
    for (unsigned int j = 0; j < N; j++) {
      y_d[i] += A[i * N + j] * x_d[i];
    }
  }
  for(unsigned int i = tid; i < N; i += stride){
    y[i] = y_d[i];
  }
}


__global__ void function_d_gpu(const double s, const double *x, const double *y, double *z,
                   const int N){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for(unsigned int i = tid; i < N; i += stride){
    if (i % 2 == 0) {
      z[i] = s * x[i] + y[i];
    } else {
      z[i] = x[i] + y[i];
    }
  }
}

void init_datastructures(double *u, double *v, double *A, const int N) {
  for (unsigned int i = 0; i < N; i++) {
    u[i] = 1;
    v[i] = 1;
  }
  for (unsigned int i = 0; i < N * N; i++) {
    A[i] = 1;
  }
}

void print_results_to_file(const double s, const double *x, const double *y,
                           const double *z, const double *A, const long long n,
                           std::ofstream &File) {
  unsigned int N = std::min(n, static_cast<long long>(30));

  File << "N: "
       << "\n"
       << n << "\n";

  File << "s: "
       << "\n"
       << s << "\n";

  File << "x: "
       << "\n";
  for (unsigned int i = 0; i < N; i++) {
    File << x[i] << " ";
  }
  File << "\n";

  File << "y: "
       << "\n";
  for (unsigned int i = 0; i < N; i++) {
    File << y[i] << " ";
  }
  File << "\n";

  File << "z: "
       << "\n";
  for (unsigned int i = 0; i < N; i++) {
    File << z[i] << " ";
  }
  File << "\n";
}

int main(int argc, char **argv) {
  long long N;
  int blockNum = 16;
  int threadPerB = 16;
  if (argc >= 2) {
    N = std::stoi(argv[1]);
    if (argc == 4) {
      N = std::stoi(argv[1]);
      blockNum = std::stoi(argv[2]);
      threadPerB = std::stoi(argv[3]);
    }
  } 
  else {
    std::cout << "Error: Missing problem size N. Please provide N as "
                 "commandline parameter. Usage example for N=10: "
                 "./number_crunching 10"
              << std::endl;
    exit(0);
  }


  double *temp_result, *s, *u, *v, *A, *x, *y, *z;
  cudaMallocManaged(&temp_result, sizeof(double)*N);
  cudaMallocManaged(&s, sizeof(double)*1);
  cudaMallocManaged(&u, sizeof(double)*N);
  cudaMallocManaged(&v, sizeof(double)*N);
  cudaMallocManaged(&A, sizeof(double)*N*N);
  cudaMallocManaged(&x, sizeof(double)*N);
  cudaMallocManaged(&y, sizeof(double)*N);
  cudaMallocManaged(&z, sizeof(double)*N);


  init_datastructures(u, v, A, N);

  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  dim3 blockCot(blockNum);
  dim3 threadPerBlock(threadPerB);

  auto t0 = std::chrono::high_resolution_clock::now();

  function_a_gpu<<<blockCot,threadPerBlock,0,stream1>>>(u,v,temp_result,N);
  sum_func_a_result<<<1,1,0,stream1>>>(temp_result,s,N);

  function_b_gpu<<<blockCot,threadPerBlock,0,stream2>>>(u,v,x,N);

  long long shareMemSize = 2 * sizeof(double) * N;
  if (shareMemSize < 48 * 1024){ //shared memory should below 48 KB for cuda devices of compute capability 2.x
    // std::cout<< "using shared memory" << std::endl;
    function_c_gpu_shareMem<<<blockCot,threadPerBlock,shareMemSize,stream2>>>(A,x,y,N);
  }else{
    // std::cout<< "exceed shared memory size, compute without it" << std::endl;
    function_c_gpu<<<blockCot,threadPerBlock,0,stream2>>>(A,x,y,N);
  }
  cudaStreamSynchronize(stream1);
  cudaStreamSynchronize(stream2);

  function_d_gpu<<<blockCot,threadPerBlock>>>(*s,x,y,z,N);
  cudaDeviceSynchronize();

  auto t1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration< double > fs = t1 - t0;
  std::chrono::milliseconds d = std::chrono::duration_cast< std::chrono::milliseconds >( fs );
  std::cout << d.count() << " for " << blockNum <<" and " << threadPerB <<std::endl;


  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);

  std::ofstream File("task_partial_results.out");
  print_results_to_file(*s, x, y, z, A, N, File);

  // std::cout << "For correctness checking, partial results have been written to "
  //              "task_partial_results.out"
  //           << std::endl;

  cudaFree(u);
  cudaFree(v);
  cudaFree(A);
  cudaFree(x);
  cudaFree(y);
  cudaFree(z);
  cudaFree(temp_result);

  EXIT_SUCCESS;
}
