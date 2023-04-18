#include <iostream>



__global__ void factorial(const int N, int *f){
  if ( N >= 1) {
    *f = *f*N;
    factorial<<<1,1>>>(N-1,f);
  }
}


int main()
{
    auto devices = sycl::device::get_devices();
    for(auto d : devices) std::cout << d.get_info<sycl::info::device::name>() << "\n";

    sycl::host_selector cpu;
    sycl::queue q(cpu);

    std::cout << "Device: "
	      << q.get_device().get_info<sycl::info::device::name>()
              << "\n";
    

    int N = 5;
    int *f;
    cudaMallocManaged(&f,sizeof(int));
    *f = 1;
    factorial<<<1,1>>>(N,f);
    cudaDeviceSynchronize();
    std::cout << *f << "\n";

    return EXIT_SUCCESS;
}
