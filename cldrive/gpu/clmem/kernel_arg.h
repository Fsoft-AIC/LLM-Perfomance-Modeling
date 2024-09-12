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

#include "gpu/clmem/kernel_arg_value.h"
#include "gpu/clmem/opencl_type.h"
#include "gpu/clmem/proto/clmem.pb.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/statusor.h"
#include "opencl_type.h"
#include "third_party/opencl/cl.hpp"

namespace gpu {
namespace clmem {

class KernelArg {
 public:
  KernelArg() : type_(OpenClType::DEFAULT_UNKNOWN) {}

  labm8::Status Init(cl::Kernel *kernel, size_t arg_index);

  // Create a random value for this argument. If the argument is not supported,
  // returns nullptr.
  std::unique_ptr<KernelArgValue> TryToCreateRandomValue(
      const cl::Context &context, const int& size) const;

  // Create a "ones" value for this argument. If the argument is not supported,
  // returns nullptr.
  std::unique_ptr<KernelArgValue> TryToCreateConstValue(
      const cl::Context &context, const int& size, const int& value) const;

  // Address qualifier accessors.

  bool IsGlobal() const;

  bool IsLocal() const;

  bool IsConstant() const;

  bool IsPrivate() const;

  bool IsPointer() const;

  const OpenClType &type() const;

 private:
  std::unique_ptr<KernelArgValue> TryToCreateKernelArgValueRandom(
    const cl::Context& context, const int& size) const;
  std::unique_ptr<KernelArgValue> TryToCreateKernelArgValueConst(
    const cl::Context& context, const int& size, const int& value) const;

  OpenClType type_;
  cl_kernel_arg_address_qualifier address_;
  bool is_pointer_;
};

}  // namespace clmem
}  // namespace gpu
