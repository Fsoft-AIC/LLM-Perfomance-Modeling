# cldrive - Run arbitrary OpenCL kernels
`cldrive` is a tool for running
arbitrary OpenCL kernels to record their runtimes and outputs. It reads OpenCL
kernels from an input file, and for each, generates random inputs
(parameterized by a given size), runs the kernel and records its execution time
and outputs. It was developed as part of Chris Cummins's work on
[Deep Learning benchmark synthesis](https://github.com/ChrisCummins/clgen).

## Setup

### Docker setup
We provide a pre-built docker image with all dependencies installed at `ghcr.io/minhkhoi1026/cldrive/cldrive:latest`. Run the container using `docker compose`:
```sh
$ cd docker
$ docker compose up -d
```

### Linux setup
1. Create a conda env with python 3.6
```sh
$ conda create -n cldrive python=3.6
$ conda activate cldrive
```
2. Install `bazel` version 3.7.2 
```sh
$ chmod +x bazel-3.7.2-installer-linux-x86_64.sh
$ ./bazel-3.7.2-installer-linux-x86_64.sh --user
```
Then build cldrive using:

```sh
$ bazel build -c opt //gpu/cldrive
```
This will build an optimized `cldrive` binary and print its path.

For more details, see [INSTALL.md](https://github.com/ChrisCummins/phd/blob/master/INSTALL.md) for instructions on setting up the build environment.

## Usage

```sh
$ cldrive --srcs=<opencl_sources> --envs=<opencl_devices>
```

Where `<opencl_sources>` if a comma separated list of absolute paths to OpenCL
source files, and `<opencl_devices>` is a comma separated list of
fully-qualified OpenCL device names. To list the available device names use
`--clinfo`. Use `--help` to see the full list of options.

### Example

For example, given a file:

```sh
$ cat kernel.cl
kernel void my_kernel(global int* a, global int* b) {
    int tid = get_global_id(0);
    a[tid] += 1;
    b[tid] = a[tid] * 2;
}
```

and available OpenCL devices:

```sh
$ cldrive --clinfo
GPU|NVIDIA|GeForce_GTX_1080|396.37|1.2
CPU|Intel|Intel_Xeon_CPU_E5-2620_v4_@_2.10GHz|1.2.0.25|2.0
```

To run the kernel 5 times on both devices using 4096 work items divided into
work groups of size 1024:

```sh
$ cldrive --srcs=$PWD/kernel.cl --num_runs=5 \
    --gsize=4096 --lsize=1024 \
    --envs='GPU|NVIDIA|GeForce_GTX_1080|396.37|1.2','CPU|Intel|Intel_Xeon_CPU_E5-2620_v4_@_2.10GHz|1.2.0.25|2.0'
OpenCL Device, Kernel Name, Global Size, Local Size, Transferred Bytes, Runtime (ns)
I 2019-02-26 09:54:10 [gpu/cldrive/libcldrive.cc:59] clBuildProgram() with options '-cl-kernel-arg-info' completed in 1851 ms
GPU|NVIDIA|GeForce_GTX_1080|396.37|1.2, my_kernel, 4096, 1024, 65536, 113344
GPU|NVIDIA|GeForce_GTX_1080|396.37|1.2, my_kernel, 4096, 1024, 65536, 57984
GPU|NVIDIA|GeForce_GTX_1080|396.37|1.2, my_kernel, 4096, 1024, 65536, 64096
GPU|NVIDIA|GeForce_GTX_1080|396.37|1.2, my_kernel, 4096, 1024, 65536, 73696
GPU|NVIDIA|GeForce_GTX_1080|396.37|1.2, my_kernel, 4096, 1024, 65536, 73632
I 2019-02-26 09:54:11 [gpu/cldrive/libcldrive.cc:59] clBuildProgram() with options '-cl-kernel-arg-info' completed in 76 ms
CPU|Intel|Intel_Xeon_CPU_E5-2620_v4_@_2.10GHz|1.2.0.25|2.0, my_kernel, 4096, 1024, 65536, 105440
CPU|Intel|Intel_Xeon_CPU_E5-2620_v4_@_2.10GHz|1.2.0.25|2.0, my_kernel, 4096, 1024, 65536, 55936
CPU|Intel|Intel_Xeon_CPU_E5-2620_v4_@_2.10GHz|1.2.0.25|2.0, my_kernel, 4096, 1024, 65536, 63296
CPU|Intel|Intel_Xeon_CPU_E5-2620_v4_@_2.10GHz|1.2.0.25|2.0, my_kernel, 4096, 1024, 65536, 56192
CPU|Intel|Intel_Xeon_CPU_E5-2620_v4_@_2.10GHz|1.2.0.25|2.0, my_kernel, 4096, 1024, 65536, 55680
```

By default, cldrive prints a CSV summary of kernel stats and runtimes to
stdout, and logging information to stderr. The raw information produced by
cldrive is described in a set of protocol buffers
[//gpu/cldrive/proto:cldrive.proto](/gpu/cldrive/proto/cldrive.proto). To print
`cldrive.Instances` protos to stdout, use argumet `--output_format=pbtxt`
to print text format protos, or `--output_format=pb` for binary format.
