# Memory access analysis to find array size

## Environment setup

We provide a pre-built Docker image for the environment setup. You can run `docker compose` in this folder to start the container:
```bash
docker compose up -d
```

Execute the following command to enter the container:
```bash
docker exec -it <container-name> bash
```

## Build the tool
**Note**: all the commands below are executed inside the container.

Copy all source code at folder [here](./insert-mem-hook) to `/home/clang-llvm/llvm-project/clang-tools-extra/insert-mem-hook` folder in the container.

Add the following line to `/home/clang-llvm/llvm-project/clang-tools-extra/CMakeLists.txt` file. This helps CMake add our tool to the build list.

```cmake
add_subdirectory(insert-mem-hook)
```

Run the following commands to build the project.

```bash
cd /home/clang-llvm/build
ninja insert-mem-hook
```

## Use the tool
You can use the tool to add hook function to array access in a OpenCL kernel. The hook function will print the argument index of the array and index passed to that array access.

Example kernel file in `kernel.cl`:

```C
kernel void A(global int* a, int x) {
  int g = get_global_id(0);
  a[g] += x;
}
```

Add hook to the kernel file using tool:

```bash
/home/clang-llvm/build/bin/insert-mem-hook kernel.cl -save_dir=/home/ --
```

Result kernel file in `/home/clang-llvm/kernels-modified/kernel.cl`:

```C
//{"a":0,"x":1}
int hook(int argId, int id) {
	printf("%d,%d\n", argId, id);
	return id;
}
kernel void A(global int* a, int x) {
  int g = get_global_id(0);
  a[hook(0, g)] += x;
}
```

## Add memory hook for all kernels in a folder
Use the code at [here](./insert_hook_folder.cpp) to add memory hook for all kernels in a folder. You can build and run it using the Makefile at [here](./Makefile):

```bash
make all
```

Inside the Makefile, there are several things you can change:
- `CXX = clang++`: compiler
- `CXXFLAGS = -std=c++17 -Wall -Wextra`: compiler flags
- `SRC = insert_hook_folder.cpp`: source file
- `FOLDER = /home/clang-llvm/kernels`: source folder containing kernels
- `SAVE_FOLDER = /home/clang-llvm/kernels-modified`: target folder to save modified kernels
- `EXECUTABLE = insert_hook_folder`: name of this program
- `COMMAND = insert-mem-hook`: command to process the kernel

## Use the resulting kernels
For further steps on analyzing hook-inserted kernel, you can use an analysis in [cldrive](cldrive/) folder. The analysis will run the kernel and print the output of the hook function.
