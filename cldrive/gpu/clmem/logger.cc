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
#include "gpu/clmem/logger.h"
#include "labm8/cpp/logging.h"

namespace gpu {
namespace clmem {

Logger::Logger(std::ostream& ostream, const ClmemInstances* const instances)
    : ostream_(ostream), instances_(instances), instance_num_(-1) {}

/*virtual*/ labm8::Status Logger::StartNewInstance() {
  ++instance_num_;
  return labm8::Status::OK;
}

/*virtual*/ labm8::Status Logger::RecordLog(
    const ClmemInstance* const instance,
    const ClmemKernelInstance* const kernel_instance,
    const ClmemKernelRun* const run,
    const gpu::libcecl::OpenClKernelInvocation* const log, bool flush) {
  CHECK(instance_num() >= 0);
  return labm8::Status::OK;
}

void Logger::PrintAndClearBuffer() {
  ostream_ << buffer_.str();
  ClearBuffer();
}

void Logger::ClearBuffer() {
  buffer_.clear();
  buffer_.str(string());
}

const ClmemInstances* Logger::instances() { return instances_; }

std::ostream& Logger::ostream(bool flush) {
  if (flush) {
    return ostream_;
  } else {
    return buffer_;
  }
}

int Logger::instance_num() const { return instance_num_; }

ProtocolBufferLogger::ProtocolBufferLogger(
    std::ostream& ostream, const ClmemInstances* const instances,
    bool text_format)
    : Logger(ostream, instances), text_format_(text_format) {}

/*virtual*/ ProtocolBufferLogger::~ProtocolBufferLogger() {
  if (text_format_) {
    ostream(/*flush=*/true) << "# File: //gpu/clmem/proto/clmem.proto\n"
                            << "# Proto: gpu.clmem.ClmemInstances\n"
                            << instances()->DebugString();
  } else {
    instances()->SerializeToOstream(&ostream(/*flush=*/true));
  }
}
NULLLogger::NULLLogger(std::ostream& ostream,
                     const ClmemInstances* const instances)
    : Logger(ostream, instances) {
  // this->ostream(/*flush=*/true) << CsvLogHeader();
  // do nothing
}

/*virtual*/ labm8::Status NULLLogger::RecordLog(
    const ClmemInstance* const instance,
    const ClmemKernelInstance* const kernel_instance,
    const ClmemKernelRun* const run,
    const gpu::libcecl::OpenClKernelInvocation* const log, bool flush) {
  // do nothing
  // ostream(flush) << CsvLog::FromProtos(instance_num(), instance,
  //                                      kernel_instance, run, log);
  return labm8::Status::OK;
}

}  // namespace clmem
}  // namespace gpu
