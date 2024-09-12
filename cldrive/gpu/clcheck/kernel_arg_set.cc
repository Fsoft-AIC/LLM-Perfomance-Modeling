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
#include "gpu/clcheck/kernel_arg_set.h"

#include "gpu/clcheck/opencl_util.h"

#include "labm8/cpp/logging.h"
#include "labm8/cpp/status_macros.h"

namespace gpu {
namespace clcheck {

KernelArgSet::KernelArgSet(cl::Kernel* kernel) : kernel_(kernel) {}

CldriveKernelInstance::KernelInstanceOutcome KernelArgSet::Init() {
  size_t num_args = kernel_->getInfo<CL_KERNEL_NUM_ARGS>();
  if (!num_args) {
    LOG(WARNING) << "Kernel '" << util::GetOpenClKernelName(*kernel_)
                 << "' has no arguments";
    return CldriveKernelInstance::NO_ARGUMENTS;
  }

  // Create args.
  int num_mutable_args = 0;
  for (size_t i = 0; i < num_args; ++i) {
    KernelArg arg_driver;
    if (!arg_driver.Init(kernel_, i).ok()) {
      LOG(WARNING) << "Skipping kernel with no mutable arguments: '"
                   << util::GetOpenClKernelName(*kernel_) << "'";
      return CldriveKernelInstance::UNSUPPORTED_ARGUMENTS;
    }
    if (arg_driver.IsGlobal()) {
      ++num_mutable_args;
    }
    args_.push_back(std::move(arg_driver));
  }

  if (!num_mutable_args) {
    LOG(WARNING) << "Skipping kernel with no mutable arguments: '"
                 << util::GetOpenClKernelName(*kernel_) << "'";
    return CldriveKernelInstance::NO_MUTABLE_ARGUMENTS;
  }

  return CldriveKernelInstance::PASS;
}

labm8::Status KernelArgSet::SetRandom(const cl::Context& context,
                                      const DynamicParams& dynamic_params,
                                      KernelArgValuesSet* values) {
  values->Clear();
  for (auto& arg : args_) {
    auto value = arg.TryToCreateRandomValue(context, dynamic_params);
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
                                      const std::vector<long long>& args_array_bound,
                                      KernelArgValuesSet* values) {
  values->Clear();
  int i = 0;
  for (auto& arg : args_) {
    DynamicParams dynamic_params;
    dynamic_params.set_global_size_x(args_array_bound[i]);
    auto value = arg.TryToCreateRandomValue(context, dynamic_params);
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
    auto value = arg.TryToCreateOnesValue(context, dynamic_params);
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

string KernelArgSet::ToStringWithValue(const KernelArgValuesSet& arg_values) const {
  string s = "[";
  for (size_t i = 0; i < arg_values.values().size(); ++i) {
    // if arg is a pointer, save the size of the array instead
    string value = arg_values.values()[i]->ToString();
    if (args_[i].IsPointer()) {
      value = std::to_string(arg_values.values()[i]->Size());
    }
    else {
      value = arg_values.values()[i]->ToString();
    }

    string qualifier;
    if (args_[i].IsGlobal()) {
      qualifier = "global";
    }
    else if (args_[i].IsLocal()) {
      qualifier = "local";
    }
    else if (args_[i].IsConstant()) {
      qualifier = "constant";
    }
    else if (args_[i].IsPrivate()) {
      qualifier = "private";
    }

    absl::StrAppend(&s, absl::StrFormat("{id: %d, name: %s, type: %s, value: %s, qualifier: %s},",
                                        i, 
                                        args_[i].name(),
                                        args_[i].type_name(),
                                        value,
                                        qualifier
                                        ));
  }
  absl::StrAppend(&s, "]");
  
  return s;
}

}  // namespace clcheck
}  // namespace gpu
