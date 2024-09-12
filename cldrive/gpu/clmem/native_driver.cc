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
#include "gpu/clmem/libclmem.h"

#include "labm8/cpp/pbutil.h"

namespace gpu {
namespace clmem {

void ProcessClmemInstancesOrDie(ClmemInstances* instances) {
  ProtocolBufferLogger logger(std::cout, instances, /*text_format=*/false);
  for (int i = 0; i < instances->instance_size(); ++i) {
    logger.StartNewInstance();
    Clmem(instances->mutable_instance(i), i).RunOrDie(logger);
  }
}

}  // namespace clmem
}  // namespace gpu

PBUTIL_INPLACE_PROCESS_MAIN(gpu::clmem::ProcessClmemInstancesOrDie,
                            gpu::clmem::ClmemInstances);
