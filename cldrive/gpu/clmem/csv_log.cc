// Copyright (c) 2016-2020 Chris Cummins.
// This file is part of clmem.
//
// clmem is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// clmem is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with clmem.  If not, see <https://www.gnu.org/licenses/>.
#include "gpu/clmem/csv_log.h"

#include "labm8/cpp/logging.h"

#include <iostream>

namespace gpu {
namespace clmem {

// Print a value only if it is not zero.
template <typename T>
std::ostream& NullIfZero(std::ostream& stream, const T& value) {
  if (value != 0) {
    stream << value;
  }
  return stream;
}

// Print a value only if it is not empty().
template <typename T>
std::ostream& NullIfEmpty(std::ostream& stream, const T& value) {
  if (!value.empty()) {
    stream << value;
  }
  return stream;
}

// Print a value only if it is not less than zero.
template <typename T>
std::ostream& NullIfNegative(std::ostream& stream, const T& value) {
  if (value >= 0) {
    stream << value;
  }
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const CsvLogHeader& header) {
  stream << "instance,device,build_opts,kernel,work_item_local_mem_size,"
         << "work_item_private_mem_size,global_size,local_size,outcome,"
         << "transferred_bytes,transfer_time_ns,kernel_time_ns\n";
  return stream;
}

CsvLog::CsvLog(int instance_id)
    : instance_id_(instance_id),
      work_item_local_mem_size_(-1),
      work_item_private_mem_size_(-1),
      global_size_(-1),
      local_size_(-1),
      transferred_bytes_(-1),
      transfer_time_ns_(-1),
      kernel_time_ns_(-1) {
  CHECK(instance_id >= 0) << "Negative instance ID not allowed";
}

std::ostream& operator<<(std::ostream& stream, const CsvLog& log) {
  stream << log.instance_id_ << "," << log.device_ << "," << log.build_opts_
         << ",";
  NullIfEmpty(stream, log.kernel_) << ",";
  NullIfNegative(stream, log.work_item_local_mem_size_) << ",";
  NullIfNegative(stream, log.work_item_private_mem_size_) << ",";
  NullIfNegative(stream, log.global_size_) << ",";
  NullIfNegative(stream, log.local_size_) << "," << log.outcome_ << ",";
  NullIfNegative(stream, log.transferred_bytes_) << ",";
  NullIfNegative(stream, log.transfer_time_ns_) << ",";
  NullIfNegative(stream, log.kernel_time_ns_) << std::endl;
  return stream;
}

/*static*/ CsvLog CsvLog::FromProtos(
    int instance_id, const ClmemInstance* const instance,
    const ClmemKernelInstance* const kernel_instance,
    const ClmemKernelRun* const run,
    const gpu::libcecl::OpenClKernelInvocation* const log) {
  CsvLog csv(instance_id);

  CHECK(instance) << "ClmemInstance pointer cannot be null";
  csv.device_ = instance->device().name();
  csv.build_opts_ = instance->build_opts();

  csv.outcome_ = ClmemInstance::InstanceOutcome_Name(instance->outcome());
  if (kernel_instance) {
    csv.kernel_ = kernel_instance->name();
    csv.work_item_local_mem_size_ =
        kernel_instance->work_item_local_mem_size_in_bytes();
    csv.work_item_private_mem_size_ =
        kernel_instance->work_item_private_mem_size_in_bytes();

    csv.outcome_ = ClmemKernelInstance::KernelInstanceOutcome_Name(
        kernel_instance->outcome());
    if (run) {
      csv.outcome_ = ClmemKernelRun::KernelRunOutcome_Name(run->outcome());
      if (log) {
        csv.global_size_ = log->global_size();
        csv.local_size_ = log->local_size();
        if (log->transferred_bytes() >= 0) {
          csv.outcome_ = "PASS";
          csv.kernel_time_ns_ = log->kernel_time_ns();
          csv.transfer_time_ns_ = log->transfer_time_ns();
          csv.transferred_bytes_ = log->transferred_bytes();
        }
      }
    }
  }

  return csv;
}

}  // namespace clmem
}  // namespace gpu
