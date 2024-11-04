   // Patric Zhao: patric.zhao@intel.com
   
   #include <CL/sycl.hpp>
   #include <iostream>
   using namespace sycl;
   
   constexpr int N = 10;
   
   int main() {
   //   queue my_gpu_queue(sycl::cpu_selector_v);
     queue my_gpu_queue(sycl::gpu_selector_v);
   
     std::cout << "Selected GPU device: " <<
       my_gpu_queue.get_device().get_info<info::device::name>() << "\n";
   
     int *host_mem   = malloc_host<int>(N, my_gpu_queue);
     int *device_mem = malloc_device<int>(N, my_gpu_queue); 
   
     // Init CPU data
     for(int i = 0; i < N; i++) {
        host_mem[i] = i;
     }
   
     // Copy from host(CPU) to device(GPU)
     my_gpu_queue.memcpy(device_mem, host_mem, N * sizeof(int)).wait();
   
     // do some works on GPU
     // ......
     //
   
     // Copy back from GPU to CPU
     my_gpu_queue.memcpy(host_mem, device_mem, N * sizeof(int)).wait();

     printf("\nData Result\n");
     for(int i = 0; i < N; i++) {
        printf("%d, ", host_mem[i]);
     }
     printf("\nTask Done!\n");
      
     free(host_mem, my_gpu_queue);
     free(device_mem, my_gpu_queue);  
   
     return 0;
   }

