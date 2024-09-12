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

#include "gpu/clcheck/opencl_type.h"
#include "gpu/clcheck/opencl_util.h"
#include "gpu/clcheck/profiling_data.h"

#include "third_party/opencl/cl.hpp"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "labm8/cpp/string.h"
#include "opencl_type.h"

namespace gpu {
namespace clcheck {

// Abstract base class.
class KernelArgValue {
 public:
  virtual ~KernelArgValue(){};

  virtual void CopyToDevice(const cl::CommandQueue &queue,
                            ProfilingData *profiling) = 0;

  virtual std::unique_ptr<KernelArgValue> CopyFromDevice(
      const cl::CommandQueue &queue, ProfilingData *profiling) = 0;

  virtual void SetAsArg(cl::Kernel *kernel, size_t arg_index) = 0;

  virtual bool operator==(const KernelArgValue *const rhs) const = 0;

  virtual bool operator!=(const KernelArgValue *const rhs) const = 0;

  virtual string ToString() const = 0;

  virtual size_t SizeInBytes() const = 0;

  virtual size_t Size() const = 0;
};

}  // namespace clcheck
}  // namespace gpu
