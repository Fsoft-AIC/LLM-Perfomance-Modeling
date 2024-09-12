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
#pragma once

#include "gpu/clmem/proto/clmem.pb.h"
#include "labm8/cpp/port.h"
#include "labm8/cpp/string.h"

#include <vector>

namespace gpu {
namespace clmem {

// A class which prints the header values for a CSV row.
//
// Usage:
//    std::cout << CsvLogHeader();
class CsvLogHeader {
  // Format CSV header to output stream.
  friend std::ostream& operator<<(std::ostream& stream,
                                  const CsvLogHeader& log);
};

// A class which formats a CSV row.
//
// Usage:
//    CsvLog::FromProtos log(...);
//    std::cout << log;
class CsvLog {
 public:
  CsvLog(int instance_id);

  // Create a log from proto messages.
  static CsvLog FromProtos(
      int instance_id, const ClmemInstance* const instance,
      const ClmemKernelInstance* const kernel_instance,
      const ClmemKernelRun* const run,
      const gpu::libcecl::OpenClKernelInvocation* const log);

  // Format CSV to output stream.
  friend std::ostream& operator<<(std::ostream& stream, const CsvLog& log);

 private:
  // Begin CSV columns (in order) -----------------------------------

  // From ClmemInstance.opencl_src field.
  int instance_id_;

  // From ClmemInstance.device.name field.
  string device_;

  // From ClmemInstance.build_opts field.
  string build_opts_;

  // The OpenCL kernel name, from ClmemKernelInstance.name
  // field. If ClmemInstance.outcome != PASS, this will be empty.
  string kernel_;

  // From ClmemKernelInstance.work_item_local_mem_size_in_bytes field. If
  // ClmemInstance.outcome != PASS, this will be empty.
  int work_item_local_mem_size_;

  // From ClmemKernelInstance.work_item_private_mem_size_in_bytes field. If
  // ClmemInstance.outcome != PASS, this will be empty.
  int work_item_private_mem_size_;

  // From ClmemInstance.dynamic_params.global_size_x field. If
  // ClmemInstance.outcome != PASS, this will be empty.
  int global_size_;

  // From ClmemInstance.dynamic_params.local_size_x field. If
  // ClmemInstance.outcome != PASS, this will be empty.
  int local_size_;

  // A stringified enum value. Either ClmemInstance.outcome if
  // ClmemInstance.outcome != PASS, else ClmemKernelInstance.outcome if
  // ClmemKernelInstance.outcome != PASS, else ClmemKernelRun.outcome.
  string outcome_;

  // From ClmemKernelRun.log. If outcome != PASS, these will be empty.
  labm8::int64 transferred_bytes_;
  labm8::int64 transfer_time_ns_;
  labm8::int64 kernel_time_ns_;

  // End CSV columns (in order) -----------------------------------
};

//
std::ostream& operator<<(std::ostream& stream, const CsvLogHeader& log);
std::ostream& operator<<(std::ostream& stream, const CsvLog& log);

}  // namespace clmem
}  // namespace gpu
