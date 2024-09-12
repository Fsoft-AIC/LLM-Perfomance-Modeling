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

#include "gpu/clmem/kernel_arg_set.h"
#include "gpu/clmem/logger.h"
#include "gpu/clmem/proto/clmem.pb.h"
#include "labm8/cpp/statusor.h"
#include "labm8/cpp/string.h"
#include "third_party/opencl/cl.hpp"

namespace gpu {
namespace clmem {

class KernelDriver {
 public:
  KernelDriver(const cl::Context& context, const cl::CommandQueue& queue,
               const cl::Kernel& kernel, ClmemInstance* instance,
               int instance_num);

  void RunOrDie(Logger& logger);

  labm8::StatusOr<ClmemKernelRun> RunDynamicParams(
      const DynamicParams& dynamic_params, Logger& logger, KernelArgValuesSet &inputs);

  // Run the kernel with the given dynamic parameters. Any error here will
  // result in the programming terminating. If flush is true, the result is
  // logged immediately. Else, the result is buffered, to be logged at a
  // later call to logger.FlushLogs().
  gpu::libcecl::OpenClKernelInvocation RunOnceOrDie(
      const DynamicParams& dynamic_params, KernelArgValuesSet& inputs,
      KernelArgValuesSet* outputs, const ClmemKernelRun* const run,
      Logger& logger, bool flush = true);

 private:
  // Private helper to public RunDynamicParams() method that doesn't catch
  // OpenCL exceptions.
  labm8::Status RunDynamicParams(const DynamicParams& dynamic_params,
                                 Logger& logger, ClmemKernelRun* run, 
                                 KernelArgValuesSet &inputs);

  cl::Context context_;
  cl::CommandQueue queue_;
  cl::Device device_;
  cl::Kernel kernel_;
  const ClmemInstance& instance_;
  int instance_num_;
  ClmemKernelInstance* kernel_instance_;
  string name_;
  KernelArgSet args_set_;
};

}  // namespace clmem
}  // namespace gpu
