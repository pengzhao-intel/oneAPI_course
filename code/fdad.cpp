   // patric zhao, patric.zhao@intel.com
   // show SLM usage by Finite Difference Approximating Derivatives (fdad)  
   #include <CL/sycl.hpp>
   #include <iostream>
   using namespace sycl;

   #define random_float() (rand() / double(RAND_MAX))
   #define BLOCK 256 
   #define CheckResult 0
   
   constexpr int64_t N = 256 * 256 * 256 + 2;
   constexpr float delta = 0.001f;

   void verify(float *gpu, float *cpu, int N) {
       int error = 0;
       for(int i = 0; i < N; i++) {
           if(std::fabs(gpu[i] - cpu[i]) > 10e-3) {
               printf("\nError at %d GPU = %f, CPU = %f\n", i, gpu[i], cpu[i]);
               error++;
           }
           if(error > 20) break;
       }
       return;
   }
   
   int main() {
     
     // Enable queue profiling  
     auto propList = cl::sycl::property_list {cl::sycl::property::queue::enable_profiling()};
     queue my_gpu_queue(gpu_selector{}, propList);
   
     std::cout << "Selected GPU device: " <<
       my_gpu_queue.get_device().get_info<info::device::name>() << "\n";

     float *input        = malloc_host<float>(N, my_gpu_queue);
     float *output_P_cpu = malloc_host<float>(N-2, my_gpu_queue);

     float *input_Q = malloc_device<float>(N, my_gpu_queue); 
     float *output_P = malloc_device<float>(N-2, my_gpu_queue); 

     float *output_P_gpu   = malloc_host<float>(N-2, my_gpu_queue);
   
     // Init CPU data
     for(int64_t i = 0; i < N; i++) {
        input[i] = random_float();
     }

     // CPU compuatation
     printf("\n Start Computation, Number of Elems = %ld \n", N);
     for(int64_t i = 0; i < N-2; i++) {
         output_P_cpu[i] = (input[i+2] - input[i]) / (2.0f * delta);
     }

     float duration_gpu_a = 0.0;
     float duration_gpu_b = 0.0;
     
     // Copy from host(CPU) to device(GPU)
     my_gpu_queue.memcpy(input_Q, input, N * sizeof(float)).wait();

     int warmup = 10;
     int iteration = 50;
     for(int i = 0; i < iteration + warmup; i++) {

         // read/write global memory directly
         auto event1 = my_gpu_queue.submit([&](handler& h) {
             h.parallel_for(nd_range<1>{N-2, BLOCK}, [=](nd_item<1> item) {
                  auto global_id = item.get_global_id(0);
                  output_P[global_id] = (input_Q[global_id +2] - input_Q[global_id]) / (2.0f * delta);
             });
         });
         // wait the computation done
         my_gpu_queue.wait();

         if (i >= warmup) {
             duration_gpu_a +=
              (event1.get_profiling_info<info::event_profiling::command_end>() -
              event1.get_profiling_info<info::event_profiling::command_start>()) /1000.0f/1000.0f;
         }
   
         if (CheckResult) {
             my_gpu_queue.memcpy(output_P_gpu, output_P, (N - 2) * sizeof(float)).wait();
             verify(output_P_gpu, output_P_gpu, N);
         }

         // read data to SLM and then computaiton w/ SLM read
         // finally write back to global memory
         auto event2 = my_gpu_queue.submit([&](handler& h) {

           // Define SLM size per work-group      
           sycl::accessor<float, 1, sycl::access::mode::read_write, 
                                    sycl::access::target::local>
                                    slm_buffer(BLOCK + 2, h);


           h.parallel_for(nd_range<1>(N-2, BLOCK), [=](nd_item<1> item) {

                auto local_id  = item.get_local_id(0);
                auto global_id = item.get_global_id(0);

                slm_buffer[local_id] = input_Q[global_id];
                if(local_id == BLOCK-1) {
                    slm_buffer[BLOCK  ] = input_Q[global_id +1];
                    slm_buffer[BLOCK+1] = input_Q[global_id +2];
                }
                item.barrier(sycl::access::fence_space::local_space);

                output_P[global_id] = (slm_buffer[local_id +2] - slm_buffer[local_id]) / (2.0f * delta);
           });

         });
         my_gpu_queue.wait();

         if (i >= warmup) {
             duration_gpu_b +=
             (event2.get_profiling_info<info::event_profiling::command_end>() -
             event2.get_profiling_info<info::event_profiling::command_start>()) /1000.0f/1000.0f;
         }

         if (CheckResult) {
             my_gpu_queue.memcpy(output_P_gpu, output_P, (N - 2) * sizeof(float)).wait();
             verify(output_P_gpu, output_P_gpu, N);
         }

     }

     printf("\n GPU Computation, GPU Time w/o SLM = %lf \n", duration_gpu_a / iteration);
     printf("\n GPU Computation, GPU Time w/  SLM = %lf \n", duration_gpu_b / iteration);

     printf("\nTask Done!\n");
   
     free(input_Q, my_gpu_queue);
     free(output_P, my_gpu_queue);
     free(output_P_cpu, my_gpu_queue);
     free(output_P_gpu, my_gpu_queue);
     free(input, my_gpu_queue);

     return 0;
   }

