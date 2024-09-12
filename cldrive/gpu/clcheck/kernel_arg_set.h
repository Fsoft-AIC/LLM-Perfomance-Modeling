// Copyright (c) 2016-2020 Chris Cummins.
// This file is part of clcheck.
//
// clcheck is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// clcheck is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with clcheck.  If not, see <https://www.gnu.org/licenses/>.
#pragma once

#include "gpu/clcheck/kernel_arg.h"
#include "gpu/clcheck/kernel_arg_values_set.h"
#include "gpu/clcheck/proto/clcheck.pb.h"
#include "labm8/cpp/status.h"
#include "third_party/opencl/cl.hpp"

namespace gpu {
namespace clcheck {

class KernelArgSet {
 public:
  KernelArgSet(cl::Kernel* kernel);

  CldriveKernelInstance::KernelInstanceOutcome LogErrorOutcome(
      const CldriveKernelInstance::KernelInstanceOutcome& outcome);

  CldriveKernelInstance::KernelInstanceOutcome Init();

  labm8::Status SetRandom(const cl::Context& context,
                          const DynamicParams& dynamic_params,
                          KernelArgValuesSet* values);
  labm8::Status SetRandom(const cl::Context& context,
                                      const std::vector<long long>& args_array_bound,
                                      KernelArgValuesSet* values);

  labm8::Status SetOnes(const cl::Context& context,
                        const DynamicParams& dynamic_params,
                        KernelArgValuesSet* values);
  const std::vector<KernelArg>& args() const;
  string ToStringWithValue(const KernelArgValuesSet& values) const;

 private:
  cl::Kernel* kernel_;
  std::vector<KernelArg> args_;
};

}  // namespace clcheck
}  // namespace gpu
