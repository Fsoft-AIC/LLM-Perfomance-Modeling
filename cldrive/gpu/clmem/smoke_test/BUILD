# Smoke test for clmem.
load("@com_github_chriscummins_rules_bats//:bats.bzl", "bats_test")

bats_test(
    name = "smoke_test",
    srcs = ["smoke_test.bats"],
    data = [
        "//gpu/clmem",
        "//labm8/sh:test",
        "//third_party/oclgrind",
    ],
)
