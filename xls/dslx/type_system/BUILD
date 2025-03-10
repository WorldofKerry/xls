# Copyright 2023 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Frontend type-system and associated data structures.

# pytype tests are present in this file
# cc_proto_library is used in this file

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//xls:xls_internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "concrete_type",
    srcs = ["concrete_type.cc"],
    hdrs = ["concrete_type.h"],
    deps = [
        ":parametric_expression",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/dslx:channel_direction",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/ir:bits_ops",
    ],
)

cc_test(
    name = "concrete_type_test",
    srcs = ["concrete_type_test.cc"],
    deps = [
        ":concrete_type",
        ":parametric_expression",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/ir:bits",
    ],
)

cc_library(
    name = "deduce_ctx",
    srcs = ["deduce_ctx.cc"],
    hdrs = ["deduce_ctx.h"],
    deps = [
        ":concrete_type",
        ":parametric_env",
        ":parametric_expression",
        ":type_and_parametric_env",
        ":type_info",
        ":type_mismatch_error_data",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:import_data",
        "//xls/dslx:import_routines",
        "//xls/dslx:interp_value",
        "//xls/dslx:warning_collector",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
    ],
)

cc_library(
    name = "deduce",
    srcs = [
        "deduce.cc",
        "deduce_enum_def.cc",
        "deduce_enum_def.h",
        "deduce_expr.cc",
        "deduce_expr.h",
        "deduce_invocation.cc",
        "deduce_invocation.h",
    ],
    hdrs = ["deduce.h"],
    deps = [
        ":concrete_type",
        ":concrete_type_zero_value",
        ":deduce_ctx",
        ":deduce_utils",
        ":parametric_constraint",
        ":parametric_env",
        ":parametric_expression",
        ":parametric_instantiator",
        ":type_and_parametric_env",
        ":type_info",
        ":unwrap_meta_type",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//xls/common:casts",
        "//xls/common:visitor",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:channel_direction",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:errors",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_bindings",
        "//xls/dslx:interp_value",
        "//xls/dslx:warning_kind",
        "//xls/dslx/bytecode",
        "//xls/dslx/bytecode:bytecode_emitter",
        "//xls/dslx/bytecode:bytecode_interpreter",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_node",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:token_utils",
        "//xls/ir:bits",
        "//xls/ir:format_strings",
    ],
)

cc_library(
    name = "parametric_constraint",
    srcs = ["parametric_constraint.cc"],
    hdrs = ["parametric_constraint.h"],
    deps = [
        ":concrete_type",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
    ],
)

cc_library(
    name = "parametric_instantiator_internal",
    srcs = ["parametric_instantiator_internal.cc"],
    hdrs = ["parametric_instantiator_internal.h"],
    deps = [
        ":concrete_type",
        ":deduce_ctx",
        ":parametric_bind",
        ":parametric_constraint",
        ":parametric_env",
        ":parametric_expression",
        ":type_and_parametric_env",
        ":type_info",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:errors",
        "//xls/dslx:interp_value",
        "//xls/dslx:warning_kind",
        "//xls/dslx/bytecode",
        "//xls/dslx/bytecode:bytecode_emitter",
        "//xls/dslx/bytecode:bytecode_interpreter",
        "//xls/dslx/bytecode:bytecode_interpreter_options",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
    ],
)

cc_library(
    name = "parametric_instantiator",
    srcs = ["parametric_instantiator.cc"],
    hdrs = ["parametric_instantiator.h"],
    deps = [
        ":concrete_type",
        ":deduce_ctx",
        ":parametric_constraint",
        ":parametric_instantiator_internal",
        ":type_and_parametric_env",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
    ],
)

cc_library(
    name = "type_mismatch_error_data",
    hdrs = ["type_mismatch_error_data.h"],
    deps = [
        ":concrete_type",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
    ],
)

cc_library(
    name = "maybe_explain_error",
    srcs = ["maybe_explain_error.cc"],
    hdrs = ["maybe_explain_error.h"],
    deps = [
        ":type_mismatch_error_data",
        "@com_google_absl//absl/status",
        "//xls/dslx:errors",
    ],
)

cc_library(
    name = "instantiate_parametric_function",
    srcs = ["instantiate_parametric_function.cc"],
    hdrs = ["instantiate_parametric_function.h"],
    deps = [
        ":concrete_type",
        ":deduce_ctx",
        ":parametric_constraint",
        ":parametric_instantiator",
        ":type_and_parametric_env",
        ":type_info",
        ":unwrap_meta_type",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:errors",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
    ],
)

cc_library(
    name = "typecheck_invocation",
    srcs = ["typecheck_invocation.cc"],
    hdrs = ["typecheck_invocation.h"],
    deps = [
        ":concrete_type",
        ":deduce",
        ":deduce_ctx",
        ":instantiate_parametric_function",
        ":parametric_constraint",
        ":parametric_env",
        ":type_and_parametric_env",
        ":type_info",
        ":unwrap_meta_type",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "//xls/common:casts",
        "//xls/common:visitor",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:dslx_builtins",
        "//xls/dslx:errors",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
        "//xls/dslx/bytecode",
        "//xls/dslx/bytecode:bytecode_emitter",
        "//xls/dslx/bytecode:bytecode_interpreter",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_utils",
        "//xls/dslx/frontend:builtins_metadata",
        "//xls/dslx/frontend:pos",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_library(
    name = "typecheck",
    srcs = ["typecheck.cc"],
    hdrs = ["typecheck.h"],
    deps = [
        ":concrete_type",
        ":deduce",
        ":deduce_ctx",
        ":maybe_explain_error",
        ":parametric_env",
        ":scoped_fn_stack_entry",
        ":type_info",
        ":type_mismatch_error_data",
        ":typecheck_invocation",
        ":unwrap_meta_type",
        ":warn_on_defined_but_unused",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:channel_direction",
        "//xls/dslx:constexpr_evaluator",
        "//xls/dslx:errors",
        "//xls/dslx:import_data",
        "//xls/dslx:import_routines",
        "//xls/dslx:interp_value",
        "//xls/dslx:warning_collector",
        "//xls/dslx:warning_kind",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/dslx/frontend:proc",
    ],
)

cc_library(
    name = "typecheck_test_utils",
    testonly = True,
    srcs = ["typecheck_test_utils.cc"],
    hdrs = ["typecheck_test_utils.h"],
    deps = [
        ":type_info_to_proto",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/logging:log_lines",
        "//xls/common/status:status_macros",
        "//xls/dslx:command_line_utils",
        "//xls/dslx:create_import_data",
        "//xls/dslx:parse_and_typecheck",
    ],
)

cc_test(
    name = "typecheck_test",
    srcs = ["typecheck_test.cc"],
    deps = [
        ":type_info",
        ":typecheck",
        ":typecheck_test_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/common/status:status_macros",
        "//xls/dslx:create_import_data",
        "//xls/dslx:error_printer",
        "//xls/dslx:error_test_utils",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
    ],
)

cc_test(
    name = "proc_typecheck_test",
    srcs = ["proc_typecheck_test.cc"],
    deps = [
        ":typecheck",
        ":typecheck_test_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:error_printer",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx/frontend:ast",
    ],
)

cc_library(
    name = "parametric_expression",
    srcs = ["parametric_expression.cc"],
    hdrs = ["parametric_expression.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:pos",
    ],
)

cc_test(
    name = "parametric_expression_test",
    srcs = ["parametric_expression_test.cc"],
    deps = [
        ":parametric_expression",
        "@com_google_absl//absl/container:flat_hash_set",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
    ],
)

cc_library(
    name = "concrete_type_zero_value",
    srcs = ["concrete_type_zero_value.cc"],
    hdrs = ["concrete_type_zero_value.h"],
    deps = [
        ":concrete_type",
        "//xls/common:symbolized_stacktrace",
        "//xls/common/status:ret_check",
        "//xls/dslx:errors",
        "//xls/dslx:import_data",
        "//xls/dslx:interp_value",
    ],
)

cc_library(
    name = "parametric_env",
    srcs = ["parametric_env.cc"],
    hdrs = ["parametric_env.h"],
    deps = [
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/dslx:interp_value",
    ],
)

proto_library(
    name = "type_info_proto",
    srcs = ["type_info.proto"],
)

cc_proto_library(
    name = "type_info_cc_proto",
    deps = [":type_info_proto"],
)

cc_library(
    name = "type_info_to_proto",
    srcs = ["type_info_to_proto.cc"],
    hdrs = ["type_info_to_proto.h"],
    deps = [
        ":type_info",
        ":type_info_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common:proto_adaptor_utils",
        "//xls/dslx:import_data",
        "//xls/dslx/frontend:ast_node",
    ],
)

cc_test(
    name = "type_info_to_proto_test",
    srcs = ["type_info_to_proto_test.cc"],
    data = glob(["testdata/*.txt"]),
    deps = [
        ":type_info_to_proto",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common:golden_files",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx:create_import_data",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
    ],
)

cc_library(
    name = "type_and_parametric_env",
    hdrs = ["type_and_parametric_env.h"],
    deps = [
        ":concrete_type",
        ":parametric_env",
    ],
)

cc_library(
    name = "type_info",
    srcs = ["type_info.cc"],
    hdrs = ["type_info.h"],
    deps = [
        ":concrete_type",
        ":parametric_env",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
    ],
)

cc_test(
    name = "type_info_test",
    srcs = ["type_info_test.cc"],
    deps = [
        ":type_info",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx/frontend:module",
    ],
)

cc_library(
    name = "parametric_bind",
    srcs = ["parametric_bind.cc"],
    hdrs = ["parametric_bind.h"],
    deps = [
        ":concrete_type",
        ":deduce_ctx",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/logging",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:pos",
    ],
)

cc_test(
    name = "parametric_bind_test",
    srcs = ["parametric_bind_test.cc"],
    deps = [
        ":parametric_bind",
        "@com_google_absl//absl/status:statusor",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
    ],
)

cc_library(
    name = "unwrap_meta_type",
    srcs = ["unwrap_meta_type.cc"],
    hdrs = ["unwrap_meta_type.h"],
    deps = [
        ":concrete_type",
        "@com_google_absl//absl/status:statusor",
        "//xls/common/status:ret_check",
        "//xls/dslx:errors",
    ],
)

# -- binary for invoking type checking subsystem

cc_binary(
    name = "typecheck_main",
    srcs = ["typecheck_main.cc"],
    deps = [
        ":type_info_to_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "//xls/common:exit_status",
        "//xls/common:init_xls",
        "//xls/common/file:filesystem",
        "//xls/dslx:command_line_utils",
        "//xls/dslx:create_import_data",
        "//xls/dslx:default_dslx_stdlib_path",
        "//xls/dslx:import_data",
        "//xls/dslx:parse_and_typecheck",
        "//xls/dslx:warning_kind",
    ],
)

py_test(
    name = "typecheck_main_test",
    srcs = ["typecheck_main_test.py"],
    data = [
        ":typecheck_main",
        "//xls/dslx/tests:mod_const_enum_importer.x",
        "//xls/dslx/tests:mod_simple_const_enum.x",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@com_google_absl_py//absl/testing:absltest",
        "//xls/common:runfiles",
    ],
)

cc_library(
    name = "scoped_fn_stack_entry",
    srcs = ["scoped_fn_stack_entry.cc"],
    hdrs = ["scoped_fn_stack_entry.h"],
    deps = [
        ":deduce_ctx",
        ":parametric_env",
        "//xls/common/logging",
        "//xls/dslx/frontend:ast",
    ],
)

cc_library(
    name = "warn_on_defined_but_unused",
    srcs = ["warn_on_defined_but_unused.cc"],
    hdrs = ["warn_on_defined_but_unused.h"],
    deps = [
        ":concrete_type",
        ":deduce_ctx",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//xls/common/status:ret_check",
        "//xls/common/status:status_macros",
        "//xls/dslx:warning_kind",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:ast_utils",
    ],
)

cc_library(
    name = "deduce_utils",
    srcs = ["deduce_utils.cc"],
    hdrs = ["deduce_utils.h"],
    deps = [
        ":concrete_type",
        ":deduce_ctx",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "//xls/common/logging",
        "//xls/common/status:status_macros",
        "//xls/dslx:errors",
        "//xls/dslx:interp_value",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:module",
        "//xls/dslx/frontend:pos",
        "//xls/ir:bits",
        "//xls/ir:bits_ops",
        "//xls/ir:format_preference",
    ],
)

cc_test(
    name = "deduce_utils_test",
    srcs = ["deduce_utils_test.cc"],
    deps = [
        ":concrete_type",
        ":deduce_utils",
        "@com_google_absl//absl/status",
        "//xls/common:casts",
        "//xls/common:xls_gunit",
        "//xls/common:xls_gunit_main",
        "//xls/common/status:matchers",
        "//xls/dslx/frontend:ast",
        "//xls/dslx/frontend:bindings",
        "//xls/dslx/frontend:parser",
        "//xls/dslx/frontend:scanner",
    ],
)
