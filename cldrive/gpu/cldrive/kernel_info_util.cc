#include "gpu/cldrive/kernel_arg_set.h"

#include "gpu/clinfo/libclinfo.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/strip.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace gpu {
namespace cldrive {
namespace util {

labm8::StatusOr<cl::Program> BuildOpenClProgram(
    const std::string& opencl_kernel, const cl::Context& context,
    const string& cl_build_opts) {
  auto start_time = absl::Now();
  try {
    // Assemble the build options. We need -cl-kernel-arg-info so that we can
    // read the kernel signatures.
    string all_build_opts = "-cl-kernel-arg-info ";
    absl::StrAppend(&all_build_opts, cl_build_opts);
    labm8::TrimRight(all_build_opts);

    cl::Program program(context, opencl_kernel);
    program.build(context.getInfo<CL_CONTEXT_DEVICES>(),
                  all_build_opts.c_str());
    auto end_time = absl::Now();
    auto duration = (end_time - start_time) / absl::Milliseconds(1);
    LOG(INFO) << "clBuildProgram() with options '" << all_build_opts
              << "' completed in " << duration << " ms";
    return program;
  } catch (cl::Error e) {
    LOG(WARNING) << "OpenCL exception: " << e.what() << ", error: " \
             << labm8::gpu::clinfo::OpenClErrorString(e.err());
    return labm8::Status(labm8::error::Code::INVALID_ARGUMENT,
                         "clBuildProgram failed");;
  }
}

string GetKernelInfoOrDie(std::string opencl_src, string build_opts="", cl::Device device=cl::Device()) {
  cl::Context context(device);
  cl::CommandQueue queue(context,
                         /*devices=*/context.getInfo<CL_CONTEXT_DEVICES>()[0]);

  // Compile program or fail.
  labm8::StatusOr<cl::Program> program_or = BuildOpenClProgram(
      opencl_src, context, build_opts);
  if (!program_or.ok()) {
    LOG(ERROR) << "OpenCL program compilation failed!";
    return "";
  }
  cl::Program program = program_or.ValueOrDie();

  std::vector<cl::Kernel> kernels;
  program.createKernels(&kernels);

  if (!kernels.size()) {
    LOG(ERROR) << "OpenCL program contains no kernels!";
    return "";
  }

  string res = "{";
  for (auto& kernel : kernels) {
    KernelArgSet args_set_(&kernel);
    args_set_.Init();
    res += '\"' + kernel.getInfo<CL_KERNEL_FUNCTION_NAME>() + "\": ";
    res += args_set_.ToString();
    res += ',';
  }
  res.pop_back(); // remove last comma
  res += "}";

  // TODO: explain
  for (auto kernel : kernels) {
    cl_kernel k = *(cl_kernel*)&kernel;
    ::clReleaseKernel(k);
  }
  return res;
}
}
}
}

