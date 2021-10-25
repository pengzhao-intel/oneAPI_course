  // Patric Zhao:  patric.zhao@intel.com
   
   #include <CL/sycl.hpp>
   #include <iostream>
   using namespace sycl;
   
   constexpr int64_t N = 10000000;
   
   int main() {
     
     // Enable queue profiling  
     auto propList = cl::sycl::property_list {cl::sycl::property::queue::enable_profiling()};
     queue my_gpu_queue(gpu_selector{}, propList);
   
     std::cout << "Selected GPU device: " <<
       my_gpu_queue.get_device().get_info<info::device::name>() << "\n";

   
     int *host_mem   = malloc_host<int>(N, my_gpu_queue);
     int *cpu_mem   = malloc_host<int>(N, my_gpu_queue);
     int *device_mem = malloc_device<int>(N, my_gpu_queue); 
   
     // Init CPU data
     for(int64_t i = 0; i < N; i++) {
        host_mem[i] = i % 6666;
     }

     float duration_cpu = 0.0;
     float duration_gpu = 0.0;
     float duration_total = 0.0;

     std::chrono::high_resolution_clock::time_point s_cpu, e_cpu;
     std::chrono::high_resolution_clock::time_point s_gpu, e_gpu;
     std::chrono::high_resolution_clock::time_point s_t, e_t;

      // warmup
     /*********************************************************************/
      my_gpu_queue.memcpy(device_mem, host_mem, N * sizeof(int)).wait();
      my_gpu_queue.submit([&](handler& h) {

       // Parallel Computation
       h.parallel_for(range{N}, [=](id<1> item) {
         device_mem[item] *= 2;
       });

      });
      my_gpu_queue.wait();
     /*********************************************************************/

     // CPU computation
     printf("\n Start CPU Computation, Number of Elems = %ld \n", N);
     
     s_t = std::chrono::high_resolution_clock::now();
     s_cpu = std::chrono::high_resolution_clock::now();
     // CPU code here
     for(int64_t i = 0; i < N; i++) {
         cpu_mem[i] = host_mem[i] * 2;
     }
     e_cpu = std::chrono::high_resolution_clock::now();
     duration_cpu =  std::chrono::duration<float, std::milli>(e_cpu - s_cpu).count();
   

     s_gpu = std::chrono::high_resolution_clock::now();
     // submit the content to the queue for execution
     auto event = my_gpu_queue.submit([&](handler& h) {
       
       // Parallel Computation      
       h.parallel_for(range{N}, [=](id<1> item) {
         device_mem[item] *= 2;
       });

     });
     // wait the computation done
     my_gpu_queue.wait();
     e_gpu = std::chrono::high_resolution_clock::now();
     e_t = std::chrono::high_resolution_clock::now();
     duration_gpu =  std::chrono::duration<float, std::milli>(e_gpu - s_gpu).count();
     duration_total =  std::chrono::duration<float, std::milli>(e_t - s_t).count();

     // Copy back from GPU to CPU
     my_gpu_queue.memcpy(host_mem, device_mem, N * sizeof(int)).wait();

     printf("\n CPU Computation,   Time = %lf \n", duration_cpu);
     printf("\n GPU Computation,   Time = %lf \n", duration_gpu);
     printf("\n Total Computation, TIme = %lf \n", duration_total);

     printf("\nTask Done!\n");
     
     free(cpu_mem, my_gpu_queue);
     free(host_mem, my_gpu_queue);
     free(device_mem, my_gpu_queue);
   
     return 0;
   }

