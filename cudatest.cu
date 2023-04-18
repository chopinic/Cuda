#include <iostream>



__global__ void factorial(const int N, int *f){
  if ( N >= 1) {
    *f = *f*N;
    factorial<<<1,1>>>(N-1,f);
  }
}

using namespace std;

int main()
{

    int N = 5;
    int *f;
    cudaMallocManaged(&f,sizeof(int));
    *f = 1;
    factorial<<<1,1>>>(N,f);
    cudaDeviceSynchronize();
    cout << *f << "\n";

    return EXIT_SUCCESS;
}
