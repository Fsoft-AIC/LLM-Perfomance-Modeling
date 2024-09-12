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
#include "gpu/clcheck/opencl_util.h"

#include "gpu/clcheck/testutil.h"

#include "labm8/cpp/test.h"

namespace gpu {
namespace clcheck {
namespace util {
namespace {

using ::gpu::clcheck::test::CreateClKernel;

TEST(GetOpenClKernelName, SingleCharacterName) {
  auto kernel = CreateClKernel("kernel void A(global int *a) {}");
  EXPECT_EQ(GetOpenClKernelName(kernel), "A");
}

TEST(GetOpenClKernelName, MultiCharacterName) {
  auto kernel = CreateClKernel("kernel void FooBar(global int *a) {}");
  EXPECT_EQ(GetOpenClKernelName(kernel), "FooBar");
}

TEST(GetKernelArgTypeName, IntPointer) {
  auto kernel = CreateClKernel("kernel void A(global int *a) {}");
  EXPECT_EQ(GetKernelArgTypeName(kernel, 0), "int*");
}

TEST(GetKernelArgTypeName, IntScalar) {
  auto kernel = CreateClKernel("kernel void A(const int a) {}");
  EXPECT_EQ(GetKernelArgTypeName(kernel, 0), "int");
}

TEST(GetKernelArgTypeName, FloatVectorPointer) {
  auto kernel = CreateClKernel("kernel void A(local float8 *a) {}");
  EXPECT_EQ(GetKernelArgTypeName(kernel, 0), "float8*");
}

TEST(GetKernelArgTypeName, SecondArg) {
  auto kernel =
      CreateClKernel("kernel void A(const int a, local float8 *b) {}");
  EXPECT_EQ(GetKernelArgTypeName(kernel, 1), "float8*");
}

}  // anonymous namespace
}  // namespace util
}  // namespace clcheck
}  // namespace gpu

TEST_MAIN();
