// This file performs the translation from OpenClType enum value to templated
// classes.
//
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

#include "third_party/opencl/cl.hpp"

#include <cstdlib>

namespace gpu {
namespace clmem {
namespace util {

std::unique_ptr<KernelArgValue> CreateGlobalMemoryArgValue(
    const OpenClType& type, const cl::Context& context, size_t size,
    const int& value, bool rand_values);

std::unique_ptr<KernelArgValue> CreateLocalMemoryArgValue(
    const OpenClType& type, size_t size);

std::unique_ptr<KernelArgValue> CreateScalarArgValue(const OpenClType& type,
                                                     const int& value);

}  // namespace util
}  // namespace clmem
}  // namespace gpu
