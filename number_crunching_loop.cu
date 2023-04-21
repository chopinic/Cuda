#include <algorithm>
#include <fstream>
#include <iostream>

double function_a(const double *u, const double *v, const int N) {
  double s = 0;
  for (unsigned int i = 0; i < N; i++) {
    s += u[i] * v[i];
  }
  return s;
}

__global__ void function_a_gpu(double *u, double *v, double *w, const int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  w[tid] = 0;
  for(unsigned int i = tid; i < N; i += stride)
  {
    w[tid] += u[i] * v[i];
  }
}

__global__ void sum_func_a_result(const double *result, double *s){
  int stride = gridDim.x * blockDim.x;
  for (unsigned int i = 0; i < stride; i++) {
    s[0] += result[i];
  }
}

double *function_b(const double *u, const double *v, const int N) {
  double *x = new double[N];
  for (unsigned int i = 0; i < N; i++) {
    x[i] = u[i] + v[i];
  }
  return x;
}

__global__ void function_b_gpu(double *u, double *v, double *result, const int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for(unsigned int i = tid; i < N; i += stride)
  {
    result[i] = u[i] + v[i];
  }
}

double *function_c(const double *A, const double *x, const int N) {
  double *y = new double[N];
  for (unsigned int i = 0; i < N; i++) {
    y[i] = 0;
  }
  for (unsigned int i = 0; i < N; i++) {
    for (unsigned int j = 0; j < N; j++) {
      y[i] += A[i * N + j] * x[i];
    }
  }
  return y;
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


double *function_d(const double s, const double *x, const double *y,
                   const int N) {
  double *z = new double[N];
  for (unsigned int i = 0; i < N; i++) {
    if (i % 2 == 0) {
      z[i] = s * x[i] + y[i];
    } else {
      z[i] = x[i] + y[i];
    }
  }
  return z;
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

  if (argc == 2) {
    N = std::stoi(argv[1]);
  } else {
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

  dim3 blockCot(32);
  dim3 threadPerBlock(32);

//   double s = function_a(u, v, N);
  function_a_gpu<<<blockCot,threadPerBlock>>>(u,v,temp_result,N);
  sum_func_a_result<<<blockCot,threadPerBlock>>>(temp_result,s);
  cudaDeviceSynchronize();

//   double *x = function_b(u, v, N);
  function_b_gpu<<<blockCot,threadPerBlock>>>(u,v,x,N);
  cudaDeviceSynchronize();

//   double *y = function_c(A, x, N);
  function_c_gpu<<<blockCot,threadPerBlock>>>(A,x,y,N);
  cudaDeviceSynchronize();

//   double *z = function_d(s, x, y, N);
  function_d_gpu<<<blockCot,threadPerBlock>>>(*s,x,y,z,N);
  cudaDeviceSynchronize();

  std::ofstream File("loop_partial_results.out");
  print_results_to_file(*s, x, y, z, A, N, File);

  std::cout << "For correctness checking, partial results have been written to "
               "loop_partial_results.out"
            << std::endl;

  cudaFree(u);
  cudaFree(v);
  cudaFree(A);
  cudaFree(x);
  cudaFree(y);
  cudaFree(z);
  cudaFree(temp_result);

  EXIT_SUCCESS;
}
