// Main entry point for clmem command line executable.
//
// Usage summary:
//   clmem --srcs=<opencl_sources> --envs=<opencl_devices>
//       --gsize=<gsize> --lsize=<lsize> --output_format=(txt|pb|pbtxt)
//
// Run with `--help` argument to see full usage options.
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
#include "gpu/clmem/libclmem.h"

#include "gpu/clmem/logger.h"
#include "gpu/clmem/proto/clmem.pb.h"
#include "gpu/clinfo/libclinfo.h"

#include "labm8/cpp/app.h"
#include "labm8/cpp/logging.h"

#include "absl/strings/str_split.h"
#include "absl/strings/numbers.h"
#include "boost/filesystem.hpp"
#include "boost/filesystem/fstream.hpp"
#include "gflags/gflags.h"

#include <sstream>

namespace {

// Split a string into a vector of comma separated strings, e.g.
//     'a,b' -> 'a', 'b'
//     'ab' -> 'ab'
std::vector<string> SplitCommaSeparated(const string& str) {
  std::vector<absl::string_view> str_paths =
      absl::StrSplit(str, ',', absl::SkipEmpty());
  return std::vector<string>(str_paths.begin(), str_paths.end());
}

template <typename int_type>
void SplitCommaSeparatedInt(const string& str, std::vector<int_type>& out) {
  CHECK(!str.empty()) << "Empty argument to SplitCommaSeparatedInt()";
  std::vector<absl::string_view> str_paths =
      absl::StrSplit(str, ',', absl::SkipEmpty());
  for (auto str_path : str_paths) {
    int_type val;
    CHECK(absl::SimpleAtoi(str_path, &val))
        << "Failed to parse integer: '" << string(str_path) << "'";
    out.push_back(val);
  }
}

// Read the entire contents of a file to string or abort.
string ReadFileOrDie(const string& path) {
  const boost::filesystem::path fs_path(path);
  CHECK(boost::filesystem::is_regular_file(fs_path))
      << "Not a regular file: '" << path << "'";
  boost::filesystem::ifstream istream(fs_path);
  CHECK(istream.is_open()) << "Failed to open: '" << path << "'";

  std::stringstream buffer;
  buffer << istream.rdbuf();
  return buffer.str();
}

}  // anonymous namespace

// Flag definitions ------------------------------------

DEFINE_string(srcs, "", "A comma separated list of OpenCL source files.");
static bool ValidateSrcs(const char* flagname, const string& value) {
  for (auto str_path : SplitCommaSeparated(value)) {
    // string str_path(str_path_view);
    boost::filesystem::path path(str_path);
    if (!boost::filesystem::is_regular_file(path)) {
      LOG(FATAL) << "File not found: " << value;
    }
  }

  return true;
}
DEFINE_validator(srcs, &ValidateSrcs);

DEFINE_string(envs, "",
              "A comma separated list of OpenCL devices to use. Use "
              "'--clinfo' argument to print a list of available devices. If "
              "not provided, all available OpenCL devices will be used.");
static bool ValidateEnvs(const char* flagname, const string& value) {
  for (auto env : SplitCommaSeparated(value)) {
    try {
      labm8::gpu::clinfo::GetOpenClDevice(env);
    } catch (std::invalid_argument e) {
      LOG(ERROR) << "Available OpenCL environments:";
      auto devices = labm8::gpu::clinfo::GetOpenClDevices();
      for (int i = 0; i < devices.device_size(); ++i) {
        LOG(ERROR) << "    " << devices.device(i).name();
      }
      LOG(FATAL) << "OpenCL environment '" << env << "' not found";
    }
  }
  return true;
}
DEFINE_validator(envs, &ValidateEnvs);

DEFINE_string(output_format, "null",
              "The output format. One of: {pb,pbtxt,null}.");
static bool ValidateOutputFormat(const char* flagname, const string& value) {
  if (value.compare("pb") && value.compare("pbtxt") && value.compare("null")) {
    LOG(FATAL) << "Illegal value for --" << flagname << ". Must be one of: "
               << "{pb,pbtxt,null}";
  }
  return true;
}
DEFINE_validator(output_format, &ValidateOutputFormat);

DEFINE_int32(gsize, 1024,
             "The global size to drive each kernel with. Buffers of this size "
             "are allocated and transferred for array arguments, and this many "
             "work items are instantiated.");
DEFINE_int32(lsize_x, 128, "The local (work group) size in first dimension. lsize_x*lsize_y*lsize_z must be <= gsize.");
DEFINE_int32(lsize_y, 1, "The local (work group) size in second dimension. lsize_x*lsize_y*lsize_z must be <= gsize.");
DEFINE_int32(lsize_z, 1, "The local (work group) size in third dimension. lsize_x*lsize_y*lsize_z must be <= gsize.");

DEFINE_string(args_values, "",
              "A comma separated list of values to use for each kernel "
              "argument. Must be the same length as the number of arguments");

DEFINE_string(cl_build_opt, "", "Build options passed to clBuildProgram().");
DEFINE_int32(num_runs, 30, "The number of runs per kernel.");
DEFINE_bool(clinfo, false, "List the available devices and exit.");

// End flag definitions ------------------------------------

namespace gpu {
namespace clmem {

std::unique_ptr<Logger> MakeLoggerFromFlags(
    std::ostream& ostream, const ClmemInstances* const instances) {
  if (!FLAGS_output_format.compare("pb")) {
    return std::make_unique<ProtocolBufferLogger>(std::cout, instances,
                                                  /*text_format=*/false);
  } else if (!FLAGS_output_format.compare("pbtxt")) {
    return std::make_unique<ProtocolBufferLogger>(std::cout, instances,
                                                  /*text_format=*/true);
  }
  else if (!FLAGS_output_format.compare("null")) {
    return std::make_unique<NULLLogger>(std::cout, instances);
  } else  {
    CHECK(false) << "unreachable!";
    return nullptr;
  }
}

}  // namespace clmem
}  // namespace gpu

namespace {

// Look up OpenCL devices from a comma separated list of names. If the string
// is empty, all available devices are returned.
std::vector<::gpu::clinfo::OpenClDevice> GetDevicesFromCommaSeparatedString(
    const string& str) {
  std::vector<::gpu::clinfo::OpenClDevice> devices;

  if (FLAGS_envs.empty()) {
    auto devices_proto = labm8::gpu::clinfo::GetOpenClDevices();
    for (int i = 0; i < devices_proto.device_size(); ++i) {
      devices.push_back(devices_proto.device(i));
    }
  } else {
    for (auto device_name : SplitCommaSeparated(FLAGS_envs)) {
      devices.push_back(
          labm8::gpu::clinfo::GetOpenClDeviceProto(device_name).ValueOrDie());
    }
  }

  return devices;
}

}  // namespace

int main(int argc, char** argv) {
  labm8::InitApp(&argc, &argv, "Drive arbitrary OpenCL kernels.");

  // Special case handling for --clinfo argument which prints to stdout then
  // quits.
  if (FLAGS_clinfo) {
    auto devices = labm8::gpu::clinfo::GetOpenClDevices();
    for (int i = 0; i < devices.device_size(); ++i) {
      std::cout << devices.device(i).name() << std::endl;
    }
    return 0;
  }

  // Check that required flags are set. We can't check this in the flag
  // validator functions as they are only required if the early-exit flags
  // above are not set.
  if (FLAGS_srcs.empty()) {
    LOG(FATAL) << "Flag --srcs must be set";
  }

  auto devices = GetDevicesFromCommaSeparatedString(FLAGS_envs);

  // Create instances proto.
  gpu::clmem::ClmemInstances instances;
  gpu::clmem::ClmemInstance* instance = instances.add_instance();
  instance->set_build_opts(FLAGS_cl_build_opt);
  auto dp = instance->add_dynamic_params();
  if ((int64_t)FLAGS_gsize < (int64_t) FLAGS_lsize_x * FLAGS_lsize_y * FLAGS_lsize_z) {
    LOG(FATAL) << "Global size must be greater than or equal to local size. Got: "
                << "gsize: " << FLAGS_gsize << ", lsize_x*lsize_y*lsize_z: " <<  
                (int64_t) FLAGS_lsize_x * FLAGS_lsize_y * FLAGS_lsize_z;
  }
  dp->set_global_size_x(FLAGS_gsize);
  dp->set_local_size_x(FLAGS_lsize_x);
  dp->set_local_size_y(FLAGS_lsize_y);
  dp->set_local_size_z(FLAGS_lsize_z);
  instance->set_min_runs_per_kernel(FLAGS_num_runs);

  // Parse logger flag.
  std::unique_ptr<gpu::clmem::Logger> logger =
      gpu::clmem::MakeLoggerFromFlags(std::cout, &instances);

  int instance_num = 0;
  std::vector<long long> args_values;
  SplitCommaSeparatedInt(FLAGS_args_values, args_values);
  for (auto arg_value : args_values) {
    instance->add_args_values(arg_value);
  }

  for (auto path : SplitCommaSeparated(FLAGS_srcs)) {
    logger->StartNewInstance();
    instance->set_opencl_src(ReadFileOrDie(path));

    for (size_t i = 0; i < devices.size(); ++i) {
      // Reset fields from previous loop iterations.
      instance->clear_outcome();
      instance->clear_kernel();
      
      *instance->mutable_device() = devices[i];

      gpu::clmem::Clmem(instance, instance_num).RunOrDie(*logger);
    }

    ++instance_num;
  }

  return 0;
}
