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
     float duration_gpu_a = 0.0;
     float duration_gpu_b = 0.0;
     float duration_gpu_c = 0.0;

     std::chrono::high_resolution_clock::time_point s, e;
     std::chrono::high_resolution_clock::time_point s_a, e_a;
     std::chrono::high_resolution_clock::time_point s_b, e_b;
     std::chrono::high_resolution_clock::time_point s_c, e_c;

     // CPU computation
     printf("\n Start CPU Computation, Number of Elems = %ld \n", N);
     
     s = std::chrono::high_resolution_clock::now();
     // CPU code here
     for(int64_t i = 0; i < N; i++) {
         cpu_mem[i] = host_mem[i] * 2;
     }
     e = std::chrono::high_resolution_clock::now();
     duration_cpu =  std::chrono::duration<float, std::milli>(e - s).count();
     printf("\n End CPU Computation, Time = %lf \n", duration_cpu);
   

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
   
     s_c = std::chrono::high_resolution_clock::now();
     // Copy from host(CPU) to device(GPU)
     my_gpu_queue.memcpy(device_mem, host_mem, N * sizeof(int)).wait();

     s_b = std::chrono::high_resolution_clock::now();
     s_a = std::chrono::high_resolution_clock::now();
     // submit the content to the queue for execution
     auto event = my_gpu_queue.submit([&](handler& h) {
       
       // Parallel Computation      
       h.parallel_for(range{N}, [=](id<1> item) {
         device_mem[item] *= 2;
       });

     });
     // wait the computation done
     my_gpu_queue.wait();
     e_b = std::chrono::high_resolution_clock::now();
     duration_gpu_b =  std::chrono::duration<float, std::milli>(e_b - s_b).count();

     duration_gpu_a =
      (event.get_profiling_info<info::event_profiling::command_end>() -
      event.get_profiling_info<info::event_profiling::command_start>()) /1000.0f/1000.0f;
   
     // Copy back from GPU to CPU
     my_gpu_queue.memcpy(host_mem, device_mem, N * sizeof(int)).wait();
     e_c = std::chrono::high_resolution_clock::now();
     duration_gpu_c =  std::chrono::duration<float, std::milli>(e_c - s_c).count();

     printf("\n GPU Computation, GPU Time A = %lf \n", duration_gpu_a);
     printf("\n GPU Computation, GPU Time B = %lf \n", duration_gpu_b);
     printf("\n GPU Computation, GPU Time C = %lf \n", duration_gpu_c);

     printf("\nTask Done!\n");
   
     return 0;
   }

