set(LLVM_LINK_COMPONENTS support)

add_clang_executable(insert-mem-hook
  MemHookInsertion.cpp
  )
target_link_libraries(insert-mem-hook
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  )
