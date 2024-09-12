// Copyright (c) 2016-2020 Chris Cummins.
// This file is part of cldrive.
//
// cldrive is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// cldrive is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with cldrive.  If not, see <https://www.gnu.org/licenses/>.
#include "gpu/cldrive/kernel_arg_set.h"

#include "gpu/cldrive/opencl_util.h"

#include "labm8/cpp/logging.h"
#include "labm8/cpp/status_macros.h"

namespace gpu {
namespace cldrive {

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

void KernelArgSet::GetScalarArgsIndexes(std::vector<int>* indexes) const {
  indexes->clear();
  for (size_t i = 0; i < args_.size(); ++i) {
    if (!args_[i].IsPointer()) {
      indexes->push_back(i);
    }
  }
}

string KernelArgSet::ToStringWithValue(const KernelArgValuesSet& arg_values) const {
  string s = "[";
  for (size_t i = 0; i < arg_values.values().size(); ++i) {
    // if arg is a pointer, save the size of the array instead
    string value = "";
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
    string is_pointer = args_[i].IsPointer() ? "true" : "false";

    absl::StrAppend(&s, absl::StrFormat("{\"id\": %d, \"name\": \"%s\", \"type\": \"%s\", \"value\": \"%s\", \"qualifier\": \"%s\", \"is_pointer\": %s},",
                                        i, 
                                        args_[i].name(),
                                        args_[i].type_name(),
                                        value,
                                        qualifier,
                                        is_pointer
                                        ));
  }
  s.back() = ']';
  
  return s;
}

string KernelArgSet::ToString() const {
  string s = "[";
  for (size_t i = 0; i < args_.size(); ++i) {
    // if arg is a pointer, save the size of the array instead
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

    string is_pointer = args_[i].IsPointer() ? "true" : "false";

    absl::StrAppend(&s, absl::StrFormat("{\"id\": %d, \"name\": \"%s\", \"type\": \"%s\", \"qualifier\": \"%s\", \"is_pointer\": %s},",
                                        i, 
                                        args_[i].name(),
                                        args_[i].type_name(),
                                        qualifier,
                                        is_pointer
                                        ));
  }
  s.back() = ']';
  
  return s;
}

}  // namespace cldrive
}  // namespace gpu
