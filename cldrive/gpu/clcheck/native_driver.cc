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
#include "gpu/clcheck/libclcheck.h"

#include "labm8/cpp/pbutil.h"

namespace gpu {
namespace clcheck {

void ProcessCldriveInstancesOrDie(CldriveInstances* instances) {
  ProtocolBufferLogger logger(std::cout, instances, /*text_format=*/false);
  for (int i = 0; i < instances->instance_size(); ++i) {
    logger.StartNewInstance();
    Cldrive(instances->mutable_instance(i), i).RunOrDie(logger);
  }
}

}  // namespace clcheck
}  // namespace gpu

PBUTIL_INPLACE_PROCESS_MAIN(gpu::clcheck::ProcessCldriveInstancesOrDie,
                            gpu::clcheck::CldriveInstances);
