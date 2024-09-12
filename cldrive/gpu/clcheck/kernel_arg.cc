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
#include "gpu/clcheck/kernel_arg.h"

#include "gpu/clcheck/global_memory_arg_value.h"
#include "gpu/clcheck/opencl_type.h"
#include "gpu/clcheck/opencl_type_util.h"
#include "gpu/clcheck/opencl_util.h"
#include "gpu/clcheck/scalar_kernel_arg_value.h"

#include "labm8/cpp/status_macros.h"

#include <cstdlib>

namespace gpu {
namespace clcheck {

labm8::Status KernelArg::Init(cl::Kernel* kernel, size_t arg_index) {
  name_ = util::GetKernelArgName(*kernel, arg_index);

  address_ = kernel->getArgInfo<CL_KERNEL_ARG_ADDRESS_QUALIFIER>(arg_index);
  CHECK(IsGlobal() || IsLocal() || IsConstant() || IsPrivate());

  // Access qualifier is one of:
  //   CL_KERNEL_ARG_ACCESS_READ_ONLY
  //   CL_KERNEL_ARG_ACCESS_WRITE_ONLY
  //   CL_KERNEL_ARG_ACCESS_READ_WRITE
  //   CL_KERNEL_ARG_ACCESS_NONE
  //
  // If argument is not an image type, CL_KERNEL_ARG_ACCESS_NONE is returned.
  // If argument is an image type, the access qualifier specified or the
  // default access qualifier is returned.
  auto access_qualifier =
      kernel->getArgInfo<CL_KERNEL_ARG_ACCESS_QUALIFIER>(arg_index);
  if (access_qualifier != CL_KERNEL_ARG_ACCESS_NONE) {
    LOG(WARNING) << "Argument " << arg_index << " is an unsupported image type";
    return labm8::Status(labm8::error::Code::INVALID_ARGUMENT,
                         "Unsupported argument");
  }

  type_name_ = util::GetKernelArgTypeName(*kernel, arg_index);

  is_pointer_ = type_name_.back() == '*';

  // Strip the trailing '*' on pointer types.
  if (is_pointer_) {
    type_name_.resize(type_name_.size() - 1);
  }

  auto type_or = OpenClTypeFromString(type_name_);
  if (!type_or.ok()) {
    LOG(WARNING) << "Argument " << arg_index << " of kernel '"
                 << util::GetOpenClKernelName(*kernel)
                 << "' is of unknown type: " << type_name_;
    return type_or.status();
  }
  type_ = type_or.ValueOrDie();

  // Check for invalid private pointer arguments.
  if (is_pointer_ && IsPrivate()) {
    LOG(WARNING) << "Pointer to private argument is not allowed";
    return labm8::Status(labm8::error::Code::INVALID_ARGUMENT,
                         "Unsupported argument");
  }

  return labm8::Status::OK;
}

const OpenClType& KernelArg::type() const { return type_; }
const string& KernelArg::name() const { return name_; }
const string& KernelArg::type_name() const { return type_name_; }

std::unique_ptr<KernelArgValue> KernelArg::TryToCreateRandomValue(
    const cl::Context& context, const DynamicParams& dynamic_params) const {
  return TryToCreateKernelArgValue(context, dynamic_params,
                                   /*rand_values=*/true);
}

std::unique_ptr<KernelArgValue> KernelArg::TryToCreateOnesValue(
    const cl::Context& context, const DynamicParams& dynamic_params) const {
  return TryToCreateKernelArgValue(context, dynamic_params,
                                   /*rand_values=*/false);
}

bool KernelArg::IsGlobal() const {
  return address_ == CL_KERNEL_ARG_ADDRESS_GLOBAL;
}

bool KernelArg::IsLocal() const {
  return address_ == CL_KERNEL_ARG_ADDRESS_LOCAL;
}

bool KernelArg::IsConstant() const {
  return address_ == CL_KERNEL_ARG_ADDRESS_CONSTANT;
}

bool KernelArg::IsPrivate() const {
  return address_ == CL_KERNEL_ARG_ADDRESS_PRIVATE;
}

bool KernelArg::IsPointer() const { return is_pointer_; }

std::unique_ptr<KernelArgValue> KernelArg::TryToCreateKernelArgValue(
    const cl::Context& context, const DynamicParams& dynamic_params,
    bool rand_values) const {
  CHECK(type() != OpenClType::DEFAULT_UNKNOWN);

  if (IsPointer() && IsGlobal()) {
    return util::CreateGlobalMemoryArgValue(
        type(), context,
        /*size=*/dynamic_params.global_size_x(),
        /*value=*/1, rand_values);
  } else if (IsPointer() && IsLocal()) {
    return util::CreateLocalMemoryArgValue(
        type(),
        /*size=*/dynamic_params.global_size_x());
  } else if (!IsPointer()) {
    return util::CreateScalarArgValue(type(),
                                      /*value=*/dynamic_params.global_size_x());
  } else {
    return std::unique_ptr<KernelArgValue>(nullptr);
  }
}

}  // namespace clcheck
}  // namespace gpu
