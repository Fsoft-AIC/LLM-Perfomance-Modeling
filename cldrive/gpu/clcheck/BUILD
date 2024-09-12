# Cldrive drives arbitrary OpenCL kernels.
#
# Copyright (c) 2016-2020 Chris Cummins.
# This file is part of clcheck.
#
# clcheck is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clcheck is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clcheck.  If not, see <https://www.gnu.org/licenses/>.

load("@cec_exports_repo//tools/bzl:exports_repo.bzl", "exports_repo")
load("@io_bazel_rules_docker//cc:image.bzl", "cc_image")

licenses(["restricted"])  # GPL v3

exports_files([
    "README.md",
    "LICENSE",
])

exports_repo(
    name = "export",
    path_rename = {
        "gpu/clcheck/README.md": "README.md",
        "gpu/clcheck/LICENSE": "LICENSE",
        "gpu/clcheck/travis.yml": ".travis.yml",
    },
    remote = "git@github.com:ChrisCummins/clcheck.git",
    tags = ["manual"],
    targets = [
        "//gpu/clcheck/...",
    ],
)

py_library(
    name = "api",
    srcs = ["api.py"],
    data = [
        ":native_csv_driver",
        ":native_driver",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//gpu/clcheck/legacy:env",
        "//gpu/clcheck/proto:clcheck_pb_py",
        "//gpu/oclgrind",
        "//labm8/py:app",
        "//labm8/py:bazelutil",
        "//labm8/py:pbutil",
        "//third_party/py/numpy",
        "//third_party/py/pandas",
    ],
)

py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    deps = [
        ":api",
        "//labm8/py:app",
        "//labm8/py:test",
        "//third_party/py/numpy",
        "//third_party/py/pytest",
    ],
)

cc_binary(
    name = "clcheck",
    srcs = ["clcheck.cc"],
    linkstatic = False,  # Needed for Oclgrind support.
    visibility = ["//visibility:public"],
    deps = [
        ":csv_log",
        ":libclcheck",
        ":mem_analysis_util",
        "//gpu/clinfo:libclinfo",
        "//labm8/cpp:app",
        "//labm8/cpp:logging",
        "@boost//:filesystem",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/strings",
    ],
)

cc_image(
    name = "clcheck_image",
    srcs = ["clcheck.cc"],
    deps = [
        # TODO(cec): This is a duplicate of the dependencies of :clcheck.
        ":csv_log",
        ":libclcheck",
        ":mem_analysis_util",
        "//gpu/clinfo:libclinfo",
        "//labm8/cpp:app",
        "//labm8/cpp:logging",
        "@boost//:filesystem",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "mem_analysis_util",
    srcs = ["mem_analysis_util.cc"],
    hdrs = ["mem_analysis_util.h"],
    deps = [
        "//labm8/cpp:logging",
        "@boost//:filesystem",
        "@com_github_jsoncpp//:jsoncpp",
    ],
)

cc_library(
    name = "csv_log",
    srcs = ["csv_log.cc"],
    hdrs = ["csv_log.h"],
    deps = [
        "//gpu/clcheck/proto:clcheck_py_cc",
        "//labm8/cpp:logging",
        "//labm8/cpp:port",
        "//labm8/cpp:string",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "global_memory_arg_value",
    hdrs = ["global_memory_arg_value.h"],
    deps = [
        ":kernel_arg_value",
        "//labm8/cpp:logging",
        "//labm8/cpp:string",
        "//third_party/opencl",
    ],
)

cc_test(
    name = "global_memory_arg_value_test",
    srcs = ["global_memory_arg_value_test.cc"],
    deps = [
        ":global_memory_arg_value",
        ":testutil",
        "//labm8/cpp:port",
        "//labm8/cpp:test",
    ],
)

cc_library(
    name = "kernel_arg",
    srcs = ["kernel_arg.cc"],
    hdrs = ["kernel_arg.h"],
    deps = [
        ":global_memory_arg_value",
        ":kernel_arg_value",
        ":opencl_type",
        ":opencl_type_util",
        ":opencl_util",
        ":scalar_kernel_arg_value",
        "//gpu/clcheck/proto:clcheck_py_cc",
        "//labm8/cpp:status",
        "//labm8/cpp:status_macros",
        "//labm8/cpp:statusor",
        "//third_party/opencl",
    ],
)

cc_test(
    name = "kernel_arg_test",
    srcs = ["kernel_arg_test.cc"],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    linkstatic = False,  # Needed for oclgrind support.
    deps = [
        ":global_memory_arg_value",
        ":kernel_arg",
        ":testutil",
        "//gpu/clcheck/proto:clcheck_py_cc",
        "//labm8/cpp:test",
        "//third_party/opencl",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)

cc_library(
    name = "kernel_arg_set",
    srcs = ["kernel_arg_set.cc"],
    hdrs = ["kernel_arg_set.h"],
    deps = [
        ":kernel_arg",
        ":kernel_arg_values_set",
        ":opencl_util",
        "//gpu/clcheck/proto:clcheck_py_cc",
        "//labm8/cpp:logging",
        "//labm8/cpp:status",
        "//labm8/cpp:status_macros",
        "//third_party/opencl",
    ],
)

cc_test(
    name = "kernel_arg_set_test",
    srcs = ["kernel_arg_set_test.cc"],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    linkstatic = False,  # Needed for oclgrind support.
    deps = [
        ":kernel_arg_set",
        "//labm8/cpp:test",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)

cc_library(
    name = "kernel_arg_value",
    srcs = ["kernel_arg_value.cc"],
    hdrs = ["kernel_arg_value.h"],
    deps = [
        ":opencl_type",
        ":opencl_util",
        ":profiling_data",
        "//labm8/cpp:logging",
        "//labm8/cpp:string",
        "//third_party/opencl",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "kernel_arg_values_test",
    srcs = ["kernel_arg_values_test.cc"],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    linkstatic = False,  # Needed for oclgrind support.
    deps = [
        ":kernel_arg_value",
        "//labm8/cpp:port",
        "//labm8/cpp:test",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)

cc_library(
    name = "kernel_arg_values_set",
    srcs = ["kernel_arg_values_set.cc"],
    hdrs = ["kernel_arg_values_set.h"],
    deps = [
        ":kernel_arg_value",
        "//labm8/cpp:logging",
        "//third_party/opencl",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_test(
    name = "kernel_arg_values_set_test",
    srcs = ["kernel_arg_values_set_test.cc"],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    linkstatic = False,  # Needed for oclgrind support.
    deps = [
        ":kernel_arg_values_set",
        "//labm8/cpp:test",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)

cc_library(
    name = "kernel_driver",
    srcs = ["kernel_driver.cc"],
    hdrs = ["kernel_driver.h"],
    deps = [
        ":kernel_arg_set",
        ":logger",
        ":opencl_util",
        ":mem_analysis_util",
        "//gpu/clcheck/proto:clcheck_py_cc",
        "//gpu/clinfo:libclinfo",
        "//labm8/cpp:logging",
        "//labm8/cpp:status_macros",
        "//labm8/cpp:statusor",
        "//labm8/cpp:string",
        "//third_party/opencl",
    ],
)

cc_test(
    name = "kernel_driver_test",
    srcs = ["kernel_driver_test.cc"],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    linkstatic = False,  # Needed for oclgrind support.
    deps = [
        ":kernel_driver",
        "//labm8/cpp:test",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)

cc_library(
    name = "libclcheck",
    srcs = ["libclcheck.cc"],
    hdrs = ["libclcheck.h"],
    deps = [
        ":kernel_arg_set",
        ":kernel_arg_value",
        ":kernel_arg_values_set",
        ":kernel_driver",
        ":logger",
        "//gpu/clcheck/proto:clcheck_py_cc",
        "//gpu/clinfo:libclinfo",
        "//labm8/cpp:common",
        "//labm8/cpp:logging",
        "//labm8/cpp:macros",
        "//labm8/cpp:statusor",
        "//third_party/opencl",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

cc_test(
    name = "libclcheck_test",
    srcs = ["libclcheck_test.cc"],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    linkstatic = False,  # Needed for oclgrind support.
    deps = [
        ":libclcheck",
        "//labm8/cpp:test",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)

cc_library(
    name = "local_memory_arg_value",
    hdrs = ["local_memory_arg_value.h"],
    deps = [
        ":kernel_arg_value",
        ":profiling_data",
        "//labm8/cpp:string",
        "//third_party/opencl",
    ],
)

cc_library(
    name = "logger",
    srcs = ["logger.cc"],
    hdrs = ["logger.h"],
    deps = [
        ":csv_log",
        "//gpu/clcheck/proto:clcheck_py_cc",
        "//labm8/cpp:logging",
        "//labm8/cpp:status",
    ],
)

cc_binary(
    name = "native_driver",
    srcs = ["native_driver.cc"],
    linkstatic = False,  # Needed for Oclgrind support.
    deps = [
        ":libclcheck",
        "//gpu/clcheck/proto:clcheck_py_cc",
        "//labm8/cpp:pbutil",
    ],
)

cc_binary(
    name = "native_csv_driver",
    srcs = ["native_csv_driver.cc"],
    linkstatic = False,  # Needed for Oclgrind support.
    deps = [
        ":libclcheck",
        "//gpu/clcheck/proto:clcheck_py_cc",
    ],
)

cc_library(
    name = "opencl_type",
    srcs = ["opencl_type.cc"],
    hdrs = ["opencl_type.h"],
    deps = [
        "//labm8/cpp:logging",
        "//labm8/cpp:status_macros",
        "//labm8/cpp:statusor",
        "//labm8/cpp:string",
        "//third_party/opencl",
        "@boost//:variant",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "opencl_type_util",
    srcs = ["opencl_type_util.cc"],
    hdrs = ["opencl_type_util.h"],
    deps = [
        ":global_memory_arg_value",
        ":kernel_arg_value",
        ":local_memory_arg_value",
        ":scalar_kernel_arg_value",
        "//third_party/opencl",
    ],
)

cc_library(
    name = "opencl_util",
    srcs = ["opencl_util.cc"],
    hdrs = ["opencl_util.h"],
    deps = [
        ":profiling_data",
        "//labm8/cpp:logging",
        "//third_party/opencl",
    ],
)

cc_test(
    name = "opencl_util_test",
    srcs = ["opencl_util_test.cc"],
    deps = [
        ":opencl_util",
        ":testutil",
        "//labm8/cpp:test",
    ],
)

cc_library(
    name = "profiling_data",
    srcs = ["profiling_data.cc"],
    hdrs = ["profiling_data.h"],
    deps = [
        "//labm8/cpp:port",
        "//third_party/opencl",
    ],
)

cc_test(
    name = "profiling_data_test",
    srcs = ["profiling_data_test.cc"],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    linkstatic = False,  # Needed for oclgrind support.
    deps = [
        ":profiling_data",
        "//labm8/cpp:test",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)

cc_library(
    name = "scalar_kernel_arg_value",
    srcs = ["scalar_kernel_arg_value.cc"],
    hdrs = ["scalar_kernel_arg_value.h"],
    deps = [
        ":kernel_arg_value",
        "//labm8/cpp:logging",
        "//labm8/cpp:string",
        "//third_party/opencl",
    ],
)

cc_test(
    name = "scalar_kernel_arg_value_test",
    srcs = ["scalar_kernel_arg_value_test.cc"],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    linkstatic = False,  # Needed for oclgrind support.
    deps = [
        ":scalar_kernel_arg_value",
        "//labm8/cpp:test",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)

cc_library(
    name = "testutil",
    testonly = 1,
    srcs = ["testutil.cc"],
    hdrs = ["testutil.h"],
    linkopts = ["-ldl"] + select({
        "//:darwin": ["-framework OpenCL"],
        "//conditions:default": [],
    }),
    deps = [
        ":global_memory_arg_value",
        ":kernel_arg_value",
        ":scalar_kernel_arg_value",
        "//gpu/clcheck/proto:clcheck_py_cc",
        "//third_party/opencl",
    ] + select({
        "//:darwin": [],
        "//conditions:default": ["@libopencl//:libOpenCL"],
    }),
)
