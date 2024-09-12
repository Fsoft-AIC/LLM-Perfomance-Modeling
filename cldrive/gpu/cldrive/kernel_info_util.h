#pragma once

#include "labm8/cpp/logging.h"
#include "labm8/cpp/status.h"
#include "labm8/cpp/statusor.h"

namespace gpu {
namespace cldrive {
namespace util {

labm8::StatusOr<cl::Program> BuildOpenClProgram(
    const std::string& opencl_kernel, const cl::Context& context,
    const string& cl_build_opts);

string GetKernelInfoOrDie(std::string opencl_src, string build_opts="", cl::Device device=cl::Device());
}
}
}

