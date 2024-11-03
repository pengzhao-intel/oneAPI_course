// Patric Zhao:  patric.zhao@intel.com

#include <CL/sycl.hpp>
#include <iostream>
using namespace sycl;

int main() {
  queue my_gpu_queue( sycl::cpu_selector_v);

  std::cout << "Selected CPU device: " <<
    my_gpu_queue.get_device().get_info<info::device::name>() << "\n";

  std::cout << "max_compute_units: " <<
    my_gpu_queue.get_device().get_info<info::device::max_compute_units>() << "\n";

  std::cout << "max_work_item_dimensions: " <<
    my_gpu_queue.get_device().get_info<info::device::max_work_item_dimensions>() << "\n";

  std::cout << "max_work_group_size: " <<
    my_gpu_queue.get_device().get_info<info::device::max_work_group_size>() << "\n";

  std::cout << "max_num_sub_groups: " <<
    my_gpu_queue.get_device().get_info<info::device::max_num_sub_groups>() << "\n";

  std::cout << "supported sub_group_sizes: ";
    for(const auto&  num : my_gpu_queue.get_device().get_info<info::device::sub_group_sizes>() )
            std::cout << num << " ";
  std::cout << "\n";

  std::cout << "max_mem_alloc_size: " <<
    my_gpu_queue.get_device().get_info<info::device::max_mem_alloc_size>() << "\n";

  std::cout << "global_mem_size: " <<
    my_gpu_queue.get_device().get_info<info::device::global_mem_size>() << "\n";

  std::cout << "local_mem_size: " <<
    my_gpu_queue.get_device().get_info<info::device::local_mem_size>() << "\n";

  return 0;
}
