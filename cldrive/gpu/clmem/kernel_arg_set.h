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

#include "gpu/clmem/kernel_arg.h"
#include "gpu/clmem/kernel_arg_values_set.h"
#include "gpu/clmem/proto/clmem.pb.h"
#include "labm8/cpp/status.h"
#include "third_party/opencl/cl.hpp"

namespace gpu {
namespace clmem {

class KernelArgSet {
 public:
  KernelArgSet(cl::Kernel* kernel);

  ClmemKernelInstance::KernelInstanceOutcome LogErrorOutcome(
      const ClmemKernelInstance::KernelInstanceOutcome& outcome);

  ClmemKernelInstance::KernelInstanceOutcome Init();

  labm8::Status SetRandom(const cl::Context& context,
                          const DynamicParams& dynamic_params,
                          KernelArgValuesSet* values);
  labm8::Status SetRandom(const cl::Context& context,
                                      const std::vector<long long>& args_values,
                                      KernelArgValuesSet* values);

  labm8::Status SetOnes(const cl::Context& context,
                        const DynamicParams& dynamic_params,
                        KernelArgValuesSet* values);

  std::vector<KernelArg> args() const { return args_; }

 private:
  cl::Kernel* kernel_;
  std::vector<KernelArg> args_;
};

}  // namespace clmem
}  // namespace gpu
