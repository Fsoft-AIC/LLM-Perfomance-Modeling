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

#include "gpu/clmem/kernel_driver.h"

#include "gpu/clmem/logger.h"
#include "gpu/clmem/opencl_util.h"
#include "gpu/clinfo/libclinfo.h"

#include "labm8/cpp/logging.h"
#include "labm8/cpp/status_macros.h"

#define MAX_ARRAY_SIZE 1000000

namespace gpu {
namespace clmem {

KernelDriver::KernelDriver(const cl::Context& context,
                           const cl::CommandQueue& queue,
                           const cl::Kernel& kernel, ClmemInstance* instance,
                           int instance_num)
    : context_(context),
      queue_(queue),
      device_(context.getInfo<CL_CONTEXT_DEVICES>()[0]),
      kernel_(kernel),
      instance_(*instance),
      instance_num_(instance_num),
      kernel_instance_(instance->add_kernel()),
      name_(util::GetOpenClKernelName(kernel)),
      args_set_(&kernel_) {}

void KernelDriver::RunOrDie(Logger& logger) {
  kernel_instance_->set_name(name_);
  kernel_instance_->set_work_item_local_mem_size_in_bytes(
      kernel_.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(device_));
  kernel_instance_->set_work_item_private_mem_size_in_bytes(
      kernel_.getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(device_));
  
  kernel_instance_->set_outcome(args_set_.Init());
  if (kernel_instance_->outcome() != ClmemKernelInstance::PASS) {
    LOG(WARNING) << "Skipping kernel with unsupported arguments: '" << name_
                 << "'";
    logger.RecordLog(&instance_, kernel_instance_, /*run=*/nullptr,
                     /*log=*/nullptr);
    return;
  }
  
  // auto max_work_group_size = device_.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  // run experiment for all dynamic params, record the results
  for (int i = 0; i < instance_.dynamic_params_size(); ++i) {
    // preallocate the buffer for kernel arguments, 
    // since the size is fixed for all dynamic params 
    // (we does not know the size of the buffer before analysis)
    std::vector<long long> args_values(instance_.args_values().begin(), instance_.args_values().end());
    // for (auto& arg : args_set_.args()) {
    //   if (arg.IsPointer()) {
    //     if (arg.IsGlobal()) {
    //       args_values.push_back(MAX_ARRAY_SIZE);
    //     } else {
    //       args_values.push_back(max_work_group_size);
    //     }
    //   } else {
    //     args_values.push_back(instance_.dynamic_params(i).global_size_x());
    //   }
    // }
    KernelArgValuesSet inputs;
    auto args_status = args_set_.SetRandom(context_, args_values, &inputs);
    if (!args_status.ok()) {
      LOG(WARNING) << "Unsupported params for kernel: '" << name_ << "'";
      logger.RecordLog(&instance_, kernel_instance_, /*run=*/nullptr, 
                      /*log=*/nullptr);
      return;
    }
    auto run = RunDynamicParams(instance_.dynamic_params(i), logger, inputs);
    if (run.ok()) {
      *kernel_instance_->add_run() = run.ValueOrDie();
    } else {
      kernel_instance_->clear_run();
      kernel_instance_->set_outcome(
          ClmemKernelInstance::UNSUPPORTED_ARGUMENTS);
    }
  }
}

labm8::StatusOr<ClmemKernelRun> KernelDriver::RunDynamicParams(
    const DynamicParams& dynamic_params, Logger& logger, KernelArgValuesSet& inputs) {
  ClmemKernelRun run;

  try {
    RunDynamicParams(dynamic_params, logger, &run, inputs);
  } catch (cl::Error error) {
    LOG(WARNING) << "Error code " << error.err() << " ("
                 << labm8::gpu::clinfo::OpenClErrorString(error.err()) << ") "
                 << "raised by " << error.what() << "() while driving kernel: '"
                 << name_ << "'";
    run.set_outcome(ClmemKernelRun::CL_ERROR);
    logger.RecordLog(&instance_, kernel_instance_, &run, /*log=*/nullptr);
  }

  return run;
}

namespace {

gpu::libcecl::OpenClKernelInvocation DynamicParamsToLog(
    const DynamicParams& dynamic_params) {
  gpu::libcecl::OpenClKernelInvocation invocation;
  invocation.set_global_size_x(dynamic_params.global_size_x());
  invocation.set_local_size_x(dynamic_params.local_size_x());
  invocation.set_local_size_y(dynamic_params.local_size_y());
  invocation.set_local_size_z(dynamic_params.local_size_z());
  // Negative values indicate null.
  invocation.set_kernel_time_ns(-1);
  invocation.set_transfer_time_ns(-1);
  invocation.set_transferred_bytes(-1);
  return invocation;
}

}  // anonymous namespace

labm8::Status KernelDriver::RunDynamicParams(
    const DynamicParams& dynamic_params, Logger& logger,
    ClmemKernelRun* run, KernelArgValuesSet &inputs) {
  // Create a log message with just the dynamic params so that we can log the
  // global and local sizes on error.
  gpu::libcecl::OpenClKernelInvocation log = DynamicParamsToLog(dynamic_params);

  // Check that the dynamic params are within legal range.
  auto max_work_group_size = device_.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  if (static_cast<int>(max_work_group_size) < dynamic_params.local_size_x()) {
    run->set_outcome(ClmemKernelRun::INVALID_DYNAMIC_PARAMS);
    LOG(WARNING) << "Unsupported dynamic params to kernel '" << name_
                 << "' (local size " << dynamic_params.local_size_x()
                 << " exceeds maximum device work group size "
                 << max_work_group_size << ")";
    logger.RecordLog(&instance_, kernel_instance_, run, &log);
    return labm8::Status(labm8::error::Code::INVALID_ARGUMENT,
                         "Unsupported dynamic params");
  }
  KernelArgValuesSet output_b;
  inputs.SetAsArgs(&kernel_);
  *run->add_log() = RunOnceOrDie(dynamic_params, inputs, &output_b, run, logger,
                                 /*flush=*/false);

  // We've passed the point of rejecting the kernel. Flush the buffered logs
  // from the preliminary runs.
  logger.PrintAndClearBuffer();

  run->set_outcome(ClmemKernelRun::PASS);
  return labm8::Status::OK;
}

gpu::libcecl::OpenClKernelInvocation KernelDriver::RunOnceOrDie(
    const DynamicParams& dynamic_params, KernelArgValuesSet& inputs,
    KernelArgValuesSet* outputs, const ClmemKernelRun* const run,
    Logger& logger, bool flush) {
  gpu::libcecl::OpenClKernelInvocation log;
  ProfilingData profiling;
  cl::Event event;

  size_t global_size = dynamic_params.global_size_x();
  size_t local_size_x = dynamic_params.local_size_x();
  size_t local_size_y = dynamic_params.local_size_y();
  size_t local_size_z = dynamic_params.local_size_z();

  log.set_global_size_x(global_size);
  log.set_local_size_x(local_size_x);
  log.set_local_size_y(local_size_y);
  log.set_local_size_z(local_size_z);
  log.set_kernel_name(name_);

  inputs.CopyToDevice(queue_, &profiling);
  inputs.SetAsArgs(&kernel_);

  queue_.enqueueNDRangeKernel(kernel_, /*offset=*/cl::NullRange,
                              /*global=*/cl::NDRange(global_size),
                              /*local=*/cl::NDRange(local_size_x, local_size_y, local_size_z),
                              /*events=*/nullptr, /*event=*/&event);
  profiling.kernel_nanoseconds += GetElapsedNanoseconds(event);

  inputs.CopyFromDeviceToNewValueSet(queue_, outputs, &profiling);

  // Set run proto fields.
  log.set_kernel_time_ns(profiling.kernel_nanoseconds);
  log.set_transfer_time_ns(profiling.transfer_nanoseconds);
  log.set_transferred_bytes(profiling.transferred_bytes);

  logger.RecordLog(&instance_, kernel_instance_, run, &log, flush);

  return log;
}

}  // namespace clmem
}  // namespace gpu
