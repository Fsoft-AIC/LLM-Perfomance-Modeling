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

#include "gpu/clmem/profiling_data.h"
#include "labm8/cpp/logging.h"
#include "third_party/opencl/cl.hpp"

namespace gpu {
namespace clmem {
namespace util {

// Blocking host to device copy operation between iterators and a buffer.
// Returns the elapsed nanoseconds.
void CopyHostToDevice(const cl::CommandQueue &queue, void *host_pointer,
                      const cl::Buffer &buffer, size_t buffer_size,
                      ProfilingData *profiling);

// Blocking host to device copy operation between iterators and a buffer.
// Returns the elapsed nanoseconds.
void CopyDeviceToHost(const cl::CommandQueue &queue, const cl::Buffer &buffer,
                      void *host_pointer, size_t buffer_size,
                      ProfilingData *profiling);

// Get the name of a kernel.
string GetOpenClKernelName(const cl::Kernel &kernel);

// Get the type name of a kernel argument.
string GetKernelArgTypeName(const cl::Kernel &kernel, size_t arg_index);

}  // namespace util
}  // namespace clmem
}  // namespace gpu
