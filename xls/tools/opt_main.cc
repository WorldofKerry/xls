// Copyright 2020 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Takes in an IR file and produces an IR file that has been run through the
// standard optimization pipeline.

#include <cstdint>
#include <filesystem>  // NOLINT
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/flags/flag.h"
#include "absl/log/globals.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/ram_rewrite.pb.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/passes/pass_base.h"
#include "xls/tools/opt.h"

const char kUsage[] = R"(
Takes in an IR file and produces an IR file that has been run through the
standard optimization pipeline.

Successfully optimized IR is printed to stdout.

Expected invocation:
  opt_main <IR file>
where:
  - <IR file> is the path to the input IR file. '-' denotes stdin as input.

Example invocation:
  opt_main path/to/file.ir
)";

ABSL_FLAG(std::string, output_path, "-",
          "Output path for the optimized IR file; '-' denotes stdout.");
ABSL_FLAG(std::optional<std::string>, alsologto, std::nullopt,
          "Path to write logs to, in addition to stderr.");
// LINT.IfChange
ABSL_FLAG(std::string, top, "", "Top entity to optimize.");
ABSL_FLAG(std::string, ir_dump_path, "",
          "Dump all intermediate IR files to the given directory");
ABSL_FLAG(std::vector<std::string>, skip_passes, {},
          "If specified, passes in this comma-separated list of (short) "
          "pass names are skipped.");
ABSL_FLAG(int64_t, convert_array_index_to_select, -1,
          "If specified, convert array indexes with fewer than or "
          "equal to the given number of possible indices (by range analysis) "
          "into chains of selects. Otherwise, this optimization is skipped, "
          "since it can sometimes reduce output quality.");
ABSL_FLAG(int64_t, opt_level, xls::kMaxOptLevel,
          absl::StrFormat("Optimization level. Ranges from 1 to %d.",
                          xls::kMaxOptLevel));
ABSL_FLAG(bool, inline_procs, false,
          "Whether to inline all procs by calling the proc inlining pass.");
ABSL_FLAG(std::string, ram_rewrites_pb, "",
          "Path to protobuf describing ram rewrites.");
ABSL_FLAG(bool, use_context_narrowing_analysis, false,
          "Use context sensitive narrowing analysis. This is somewhat slower "
          "but might produce better results in some circumstances by using "
          "usage context to narrow values more aggressively.");
// LINT.ThenChange(//xls/build_rules/xls_ir_rules.bzl)
ABSL_FLAG(
    std::optional<std::string>, passes, std::nullopt,
    absl::StrFormat(
        "Explicit list of passes to run in a specific order. Passes are named "
        "by 'short_name' and if they have non-opt-level arguments these are "
        "placed in (). Fixed point sets of passes can be put within []. Pass "
        "names are separated based on spaces. For example a simple pipeline "
        "might be \"dfe dce [ ident_remove const_fold dce canon dce arith dce "
        "comparison_simp ] loop_unroll map_inline\". This should not be used "
        "with --skip_passes. If this is given the standard optimization "
        "pipeline is ignored entierly, care should be taken to ensure the "
        "given pipeline will run in reasonable amount of time. See the map in "
        "passes/optimization_pass_pipeline.cc for pass mappings. Available "
        "passes: %s",
        xls::GetOptimizationPipelineGenerator(xls::kMaxOptLevel)
            .GetAvailablePassesStr()));
ABSL_FLAG(std::optional<int64_t>, passes_bisect_limit, std::nullopt,
          "Number of passes to allow to execute. This can be used as compiler "
          "fuel to ensure the compiler finishes at a particular point.");

namespace xls::tools {
namespace {

class FileStderrLogSink final : public absl::LogSink {
 public:
  explicit FileStderrLogSink(std::filesystem::path path)
      : path_(std::move(path)) {
    XLS_CHECK_OK(SetFileContents(path_, ""));
  }

  ~FileStderrLogSink() override = default;

  void Send(const absl::LogEntry& entry) override {
    if (entry.log_severity() < absl::StderrThreshold()) {
      return;
    }

    if (!entry.stacktrace().empty()) {
      XLS_CHECK_OK(AppendStringToFile(path_, entry.stacktrace()));
    } else {
      XLS_CHECK_OK(AppendStringToFile(
          path_, entry.text_message_with_prefix_and_newline()));
    }
  }

 private:
  const std::filesystem::path path_;
};

absl::Status RealMain(std::string_view input_path) {
  if (input_path == "-") {
    input_path = "/dev/stdin";
  }

  std::string output_path = absl::GetFlag(FLAGS_output_path);

  std::optional<std::string> alsologto = absl::GetFlag(FLAGS_alsologto);
  std::unique_ptr<absl::LogSink> log_file_sink;
  if (alsologto.has_value()) {
    log_file_sink = std::make_unique<FileStderrLogSink>(*alsologto);
    absl::AddLogSink(log_file_sink.get());
  }
  absl::Cleanup log_file_sink_cleanup = [&log_file_sink] {
    if (log_file_sink) {
      absl::RemoveLogSink(log_file_sink.get());
    }
  };

  int64_t opt_level = absl::GetFlag(FLAGS_opt_level);
  std::string top = absl::GetFlag(FLAGS_top);
  std::string ir_dump_path = absl::GetFlag(FLAGS_ir_dump_path);
  std::vector<std::string> skip_passes = absl::GetFlag(FLAGS_skip_passes);
  int64_t convert_array_index_to_select =
      absl::GetFlag(FLAGS_convert_array_index_to_select);
  bool inline_procs = absl::GetFlag(FLAGS_inline_procs);
  std::string ram_rewrites_pb = absl::GetFlag(FLAGS_ram_rewrites_pb);
  bool use_context_narrowing_analysis =
      absl::GetFlag(FLAGS_use_context_narrowing_analysis);
  std::optional<std::string> pass_list = absl::GetFlag(FLAGS_passes);
  std::optional<int64_t> bisect_limit =
      absl::GetFlag(FLAGS_passes_bisect_limit);

  XLS_ASSIGN_OR_RETURN(
      std::string opt_ir,
      tools::OptimizeIrForTop(
          /*input_path=*/input_path, /*opt_level=*/opt_level,
          /*top=*/top,
          /*ir_dump_path=*/ir_dump_path,
          /*skip_passes=*/skip_passes,
          /*convert_array_index_to_select=*/convert_array_index_to_select,
          /*inline_procs=*/inline_procs,
          /*ram_rewrites_pb=*/ram_rewrites_pb,
          /*use_context_narrowing_analysis=*/use_context_narrowing_analysis,
          /*pass_list=*/pass_list,
          /*bisect_limit=*/bisect_limit));

  if (output_path == "-") {
    std::cout << opt_ir;
    return absl::OkStatus();
  }
  return SetFileContents(output_path, opt_ir);
}

}  // namespace
}  // namespace xls::tools

int main(int argc, char **argv) {
  std::vector<std::string_view> positional_arguments =
      xls::InitXls(kUsage, argc, argv);

  if (positional_arguments.empty()) {
    XLS_LOG(QFATAL) << absl::StreamFormat("Expected invocation: %s <path>",
                                          argv[0]);
  }

  return xls::ExitStatus(xls::tools::RealMain(positional_arguments[0]));
}
