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

#include "gpu/clmem/logger.h"
#include "gpu/clmem/proto/clmem.pb.h"

#include "third_party/opencl/cl.hpp"

namespace gpu {
namespace clmem {

class Clmem {
 public:
  Clmem(ClmemInstance* instance, int instance_num = 0);

  void RunOrDie(Logger& logger);

 private:
  void DoRunOrDie(Logger& logger);

  ClmemInstance* instance_;
  int instance_num_;
  cl::Device device_;
};

void ProcessClmemInstancesOrDie(ClmemInstances* instance);

}  // namespace clmem
}  // namespace gpu
