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

#include "gpu/clcheck/kernel_arg_value.h"
#include "gpu/clcheck/opencl_type.h"
#include "gpu/clcheck/proto/clcheck.pb.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/statusor.h"
#include "opencl_type.h"
#include "third_party/opencl/cl.hpp"

namespace gpu {
namespace clcheck {

class KernelArg {
 public:
  KernelArg() : type_(OpenClType::DEFAULT_UNKNOWN) {}

  labm8::Status Init(cl::Kernel *kernel, size_t arg_index);

  // Create a random value for this argument. If the argument is not supported,
  // returns nullptr.
  std::unique_ptr<KernelArgValue> TryToCreateRandomValue(
      const cl::Context &context, const DynamicParams &dynamic_params) const;

  // Create a "ones" value for this argument. If the argument is not supported,
  // returns nullptr.
  std::unique_ptr<KernelArgValue> TryToCreateOnesValue(
      const cl::Context &context, const DynamicParams &dynamic_params) const;

  // Address qualifier accessors.

  bool IsGlobal() const;

  bool IsLocal() const;

  bool IsConstant() const;

  bool IsPrivate() const;

  bool IsPointer() const;

  const OpenClType &type() const;
  const string &name() const;
  const string &type_name() const;

 private:
  std::unique_ptr<KernelArgValue> TryToCreateKernelArgValue(
      const cl::Context &context, const DynamicParams &dynamic_params,
      bool rand_values) const;

  OpenClType type_;
  cl_kernel_arg_address_qualifier address_;
  bool is_pointer_;
  string name_;
  string type_name_;
};

}  // namespace clcheck
}  // namespace gpu
