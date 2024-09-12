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
#include "gpu/clmem/kernel_arg_set.h"

#include "gpu/clmem/opencl_util.h"

#include "labm8/cpp/logging.h"
#include "labm8/cpp/status_macros.h"

namespace gpu {
namespace clmem {

KernelArgSet::KernelArgSet(cl::Kernel* kernel) : kernel_(kernel) {}

ClmemKernelInstance::KernelInstanceOutcome KernelArgSet::Init() {
  size_t num_args = kernel_->getInfo<CL_KERNEL_NUM_ARGS>();
  if (!num_args) {
    LOG(WARNING) << "Kernel '" << util::GetOpenClKernelName(*kernel_)
                 << "' has no arguments";
    return ClmemKernelInstance::NO_ARGUMENTS;
  }

  // Create args.
  int num_mutable_args = 0;
  for (size_t i = 0; i < num_args; ++i) {
    KernelArg arg_driver;
    if (!arg_driver.Init(kernel_, i).ok()) {
      LOG(WARNING) << "Skipping kernel with no mutable arguments: '"
                   << util::GetOpenClKernelName(*kernel_) << "'";
      return ClmemKernelInstance::UNSUPPORTED_ARGUMENTS;
    }
    if (arg_driver.IsGlobal()) {
      ++num_mutable_args;
    }
    args_.push_back(std::move(arg_driver));
  }

  if (!num_mutable_args) {
    LOG(WARNING) << "Skipping kernel with no mutable arguments: '"
                 << util::GetOpenClKernelName(*kernel_) << "'";
    return ClmemKernelInstance::NO_MUTABLE_ARGUMENTS;
  }

  return ClmemKernelInstance::PASS;
}

labm8::Status KernelArgSet::SetRandom(const cl::Context& context,
                                      const DynamicParams& dynamic_params,
                                      KernelArgValuesSet* values) {
  values->Clear();
  for (auto& arg : args_) {
    auto value = (arg.IsPointer()) ? arg.TryToCreateRandomValue(context, /*size=*/dynamic_params.global_size_x())
                                   : arg.TryToCreateConstValue(context, /*size=*/1, /*value=*/dynamic_params.global_size_x());
    if (value) {
      values->AddKernelArgValue(std::move(value));
    } else {
      // TryToCreateRandomValue() returns nullptr if the argument is not
      // supported.
      return labm8::Status(labm8::error::Code::INVALID_ARGUMENT,
                           "Unsupported argument type.");
    }
  }
  return labm8::Status::OK;
}

labm8::Status KernelArgSet::SetRandom(const cl::Context& context,
                                      const std::vector<long long>& args_values,
                                      KernelArgValuesSet* values) {
  values->Clear();
  int i = 0;
  for (auto& arg : args_) {
    auto value = (arg.IsPointer()) ? arg.TryToCreateRandomValue(context, /*size=*/args_values[i])
                                   : arg.TryToCreateConstValue(context, /*size=*/1, /*value=*/args_values[i]);
    if (value) {
      values->AddKernelArgValue(std::move(value));
    } else {
      // TryToCreateRandomValue() returns nullptr if the argument is not
      // supported.
      return labm8::Status(labm8::error::Code::INVALID_ARGUMENT,
                           "Unsupported argument type.");
    }
    ++i;
  }
  return labm8::Status::OK;
}

labm8::Status KernelArgSet::SetOnes(const cl::Context& context,
                                    const DynamicParams& dynamic_params,
                                    KernelArgValuesSet* values) {
  values->Clear();
  for (auto& arg : args_) {
    auto value = (arg.IsPointer()) ? arg.TryToCreateConstValue(context, /*size=*/dynamic_params.global_size_x(),/*value=*/ 1)
                                   : arg.TryToCreateConstValue(context, /*size=*/1, /*value=*/1);
    if (value) {
      values->AddKernelArgValue(std::move(value));
    } else {
      // TryToCreateRandomValue() returns nullptr if the argument is not
      // supported.
      return labm8::Status(labm8::error::Code::INVALID_ARGUMENT,
                           "Unsupported argument type.");
    }
  }
  return labm8::Status::OK;
}

}  // namespace clmem
}  // namespace gpu
