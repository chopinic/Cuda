#include <algorithm>
#include <fstream>
#include <iostream>
#include <chrono>

double function_a(const double *u, const double *v, const int N) {
  double s = 0;
  for (unsigned int i = 0; i < N; i++) {
    s += u[i] * v[i];
  }
  return s;
}

double *function_b(const double *u, const double *v, const int N) {
  double *x = new double[N];
  for (unsigned int i = 0; i < N; i++) {
    x[i] = u[i] + v[i];
  }
  return x;
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

  double *u = new double[N];
  double *v = new double[N];
  double *A = new double[N * N];

  init_datastructures(u, v, A, N);

  auto t0 = std::chrono::high_resolution_clock::now();
  double s = function_a(u, v, N);
  double *x = function_b(u, v, N);
  double *y = function_c(A, x, N);
  double *z = function_d(s, x, y, N);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration< double > fs = t1 - t0;
  std::chrono::milliseconds d = std::chrono::duration_cast< std::chrono::milliseconds >( fs );
  std::cout << d.count() << " ms \n ";

  std::ofstream File("partial_results.out");
  print_results_to_file(s, x, y, z, A, N, File);

  std::cout << "For correctness checking, partial results have been written to "
               "partial_results.out"
            << std::endl;

  delete[] u;
  delete[] v;
  delete[] A;
  delete[] x;
  delete[] y;
  delete[] z;

  EXIT_SUCCESS;
}
