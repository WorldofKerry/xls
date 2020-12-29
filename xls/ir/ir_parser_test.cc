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

#include "xls/ir/ir_parser.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "xls/common/source_location.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/number_parser.h"

namespace xls {

using status_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

// EXPECTS that the two given strings are similar modulo extra whitespace.
void ExpectStringsSimilar(
    absl::string_view a, absl::string_view b,
    xabsl::SourceLocation loc = xabsl::SourceLocation::current()) {
  std::string a_string(a);
  std::string b_string(b);
  testing::ScopedTrace trace(loc.file_name(), loc.line(),
                             "ExpectStringsSimilar failed");

  // After dumping remove any extra leading, trailing, and consecutive internal
  // whitespace verify that strings are the same.
  absl::RemoveExtraAsciiWhitespace(&a_string);
  absl::RemoveExtraAsciiWhitespace(&b_string);

  EXPECT_EQ(a_string, b_string);
}

// Parses the given string as a function, dumps the IR and compares that the
// dumped string and input string are the same modulo whitespace.
void ParseFunctionAndCheckDump(
    absl::string_view in,
    xabsl::SourceLocation loc = xabsl::SourceLocation::current()) {
  testing::ScopedTrace trace(loc.file_name(), loc.line(),
                             "ParseFunctionAndCheckDump failed");
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(auto function, Parser::ParseFunction(in, &p));
  ExpectStringsSimilar(function->DumpIr(), in, loc);
}

// Parses the given string as a package, dumps the IR and compares that the
// dumped string and input string are the same modulo whitespace.
void ParsePackageAndCheckDump(
    absl::string_view in,
    xabsl::SourceLocation loc = xabsl::SourceLocation::current()) {
  testing::ScopedTrace trace(loc.file_name(), loc.line(),
                             "ParsePackageAndCheckDump failed");
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(in));
  ExpectStringsSimilar(package->DumpIr(), in, loc);
}

TEST(IrParserTest, ParseBitsLiteral) {
  ParseFunctionAndCheckDump(R"(fn f() -> bits[37] {
  ret literal.1: bits[37] = literal(value=42, id=1)
})");
}

TEST(IrParserTest, ParseWideLiteral) {
  ParseFunctionAndCheckDump(R"(fn f() -> bits[96] {
  ret literal.1: bits[96] = literal(value=0xaaaa_bbbb_1234_5678_90ab_cdef, id=1)
})");
}

TEST(IrParserTest, ParseVariousBitsLiterals) {
  const char tmplate[] = R"(fn f() -> bits[$0] {
  ret literal.1: bits[$0] = literal(value=$1)
})";
  struct TestCase {
    int64 width;
    std::string literal;
    Bits expected;
  };
  for (const TestCase& test_case :
       {TestCase{1, "-1", UBits(1, 1)}, TestCase{8, "-1", UBits(0xff, 8)},
        TestCase{8, "-128", UBits(0x80, 8)},
        TestCase{32, "0xffffffff", UBits(0xffffffffULL, 32)},
        TestCase{32, "-0x80000000", UBits(0x80000000ULL, 32)},
        TestCase{32, "0x80000000", UBits(0x80000000ULL, 32)}}) {
    Package p("my_package");
    XLS_ASSERT_OK_AND_ASSIGN(
        auto function,
        Parser::ParseFunction(
            absl::Substitute(tmplate, test_case.width, test_case.literal), &p));
    EXPECT_EQ(function->return_value()->As<Literal>()->value().bits(),
              test_case.expected);
  }
}

TEST(IrParserTest, ParseTupleLiterals) {
  std::string text = R"(fn f() -> (bits[16], bits[96]) {
  ret literal.1: (bits[16], bits[96]) = literal(value=(1234, 0xdeadbeefdeadbeefdeadbeef))
})";
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(auto function, Parser::ParseFunction(text, &p));
  Bits deadbeef = UBits(0xdeadbeefULL, 32);
  EXPECT_EQ(
      function->return_value()->As<Literal>()->value(),
      Value::Tuple({Value(UBits(1234, 16)),
                    Value(bits_ops::Concat({deadbeef, deadbeef, deadbeef}))}));
}

TEST(IrParserTest, ParseVariousLiteralsTooFewBits) {
  const char tmplate[] = R"(fn f() -> bits[$0] {
  ret literal.1: bits[$0] = literal(value=$1)
})";
  struct TestCase {
    int64 width;
    std::string literal;
  };
  for (const TestCase& test_case :
       {TestCase{1, "-2"}, TestCase{3, "42"}, TestCase{3, "-5"},
        TestCase{8, "-129"}, TestCase{64, "0x1_ffff_ffff_ffff_ffff"},
        TestCase{32, "-0x80000001"}}) {
    Package p("my_package");
    EXPECT_THAT(
        Parser::ParseFunction(
            absl::Substitute(tmplate, test_case.width, test_case.literal), &p)
            .status(),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("is not representable")));
  }
}

TEST(IrParserTest, DuplicateKeywordArgs) {
  Package p("my_package");
  const std::string input =
      R"(fn f() -> bits[37] {
  ret literal.1: bits[37] = literal(value=42, value=123)
})";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Duplicate keyword argument 'value'")));
}

TEST(IrParserTest, WrongDeclaredNodeType) {
  Package p("my_package");
  const std::string input =
      R"(
fn less_than(a: bits[32], b: bits[32]) -> bits[1] {
  ret ult.3: bits[32] = ult(a, b)
})";
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Declared type bits[32] does not match expected type bits[1]")));
}

TEST(IrParserTest, WrongFunctionReturnType) {
  Package p("my_package");
  const std::string input =
      R"(
fn less_than(a: bits[32], b: bits[32]) -> bits[32] {
  ret ult.3: bits[1] = ult(a, b)
})";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Type of return value bits[1] does not match "
                                 "declared function return type bits[32]")));
}

TEST(IrParserTest, MissingMandatoryKeyword) {
  Package p("my_package");
  const std::string input =
      R"(fn f() -> bits[37] {
  ret literal.1: bits[37] = literal()
})";
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Mandatory keyword argument 'value' not found")));
}

TEST(IrParserTest, ParsePosition) {
  ParseFunctionAndCheckDump(
      R"(
fn f(x: bits[42], y: bits[42]) -> bits[42] {
  ret and.1: bits[42] = and(x, y, id=1, pos=0,1,3)
}
)");
}

TEST(IrParserTest, UndefinedOperand) {
  Package p("my_package");
  std::string input =
      R"(
fn f(x: bits[42]) -> bits[42] {
  ret and.1: bits[42] = and(x, z)
}
)";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("was not previously defined: \"z\"")));
}

TEST(IrParserTest, InvalidOp) {
  Package p("my_package");
  std::string input =
      R"(
fn f(x: bits[42]) -> bits[42] {
  ret foo_op.1: bits[42] = foo_op(x, z)
}
)";
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Unknown operation for string-to-op conversion: foo_op")));
}

TEST(IrParserTest, PositionalArgumentAfterKeywordArgument) {
  Package p("my_package");
  std::string input =
      R"(
fn f(x: bits[42], y: bits[42]) -> bits[42] {
  ret and.1: bits[42] = and(x, pos=0,1,3, y)
}
)";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \"=\"")));
}

TEST(IrParserTest, ExtraOperands) {
  Package p("my_package");
  const std::string input =
      R"(
fn f(x: bits[42], y: bits[42], z: bits[42]) -> bits[42] {
  ret add.1: bits[42] = add(x, y, z)
}
})";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected 2 operands, got 3")));
}

TEST(IrParserTest, TooFewOperands) {
  Package p("my_package");
  const std::string input =
      R"(
fn f(x: bits[42]) -> bits[42] {
  ret add.1: bits[42] = add(x, id=1)
}
})";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected 2 operands, got 1")));
}

TEST(IrParserTest, DuplicateName) {
  Package p("my_package");
  const std::string input =
      R"(
fn f(x: bits[42]) -> bits[42] {
 and.1: bits[42] = and(x, x, id=1)
 ret and.1: bits[42] = and(and.1, and.1)
}
})";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Name 'and.1' has already been defined")));
}

TEST(IrParserTest, ParseNode) {
  ParseFunctionAndCheckDump(
      R"(
fn f() -> bits[32] {
  literal.1: bits[32] = literal(value=3, id=1, pos=0,3,11)
  ret sub.2: bits[32] = sub(literal.1, literal.1, id=2)
})");
}

TEST(IrParserTest, ParseFunction) {
  ParseFunctionAndCheckDump(
      R"(
fn simple_arith(a: bits[32], b: bits[32]) -> bits[32] {
  ret sub.3: bits[32] = sub(a, b, id=3)
})");
}

TEST(IrParserTest, ParseULessThan) {
  ParseFunctionAndCheckDump(
      R"(
fn less_than(a: bits[32], b: bits[32]) -> bits[1] {
  ret ult.3: bits[1] = ult(a, b, id=3)
})");
}

TEST(IrParserTest, ParseSLessThan) {
  ParseFunctionAndCheckDump(
      R"(
fn less_than(a: bits[32], b: bits[32]) -> bits[1] {
  ret slt.3: bits[1] = slt(a, b, id=3)
})");
}

TEST(IrParserTest, ParseTwoPlusTwo) {
  std::string program = R"(
fn two_plus_two() -> bits[32] {
  literal.1: bits[32] = literal(value=2, id=1)
  literal.2: bits[32] = literal(value=2, id=2)
  ret add.3: bits[32] = add(literal.1, literal.2, id=3)
}
)";
  ParseFunctionAndCheckDump(program);
}

TEST(IrParserTest, ParseTwoPlusThreeCustomIdentifiers) {
  std::string program = R"(
fn two_plus_two() -> bits[32] {
  literal.1: bits[32] = literal(value=2, id=1)
  literal.2: bits[32] = literal(value=3, id=2)
  ret add.3: bits[32] = add(literal.1, literal.2, id=3)
}
)";
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(auto function, Parser::ParseFunction(program, &p));
  // The nodes are given canonical names when we dump because we don't note the
  // original names.
  ExpectStringsSimilar(function->DumpIr(), R"(
fn two_plus_two() -> bits[32] {
  literal.1: bits[32] = literal(value=2, id=1)
  literal.2: bits[32] = literal(value=3, id=2)
  ret add.3: bits[32] = add(literal.1, literal.2, id=3)
})");
}

TEST(IrParserTest, CountedFor) {
  std::string program = R"(
package CountedFor

fn body(x: bits[11], y: bits[11]) -> bits[11] {
  ret add.3: bits[11] = add(x, y, id=3)
}

fn main() -> bits[11] {
  literal.4: bits[11] = literal(value=0, id=4)
  ret counted_for.5: bits[11] = counted_for(literal.4, trip_count=7, stride=1, body=body, id=5)
}
)";
  ParsePackageAndCheckDump(program);
}

TEST(IrParserTest, CountedForMissingBody) {
  std::string program = R"(
package CountedForMissingBody

fn body(i: bits[11], x: bits[11], y: bits[11]) -> bits[11] {
  ret add.3: bits[11] = add(x, y, id=3)
}

fn main() -> bits[11] {
  literal.4: bits[11] = literal(value=0, id=4)
  ret counted_for.5: bits[11] = counted_for(literal.4, trip_count=7, stride=1)
}
)";
  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Mandatory keyword argument 'body' not found")));
}

TEST(IrParserTest, CountedForInvariantArgs) {
  std::string program = R"(
package CountedFor

fn body(i: bits[11], x: bits[11], y: bits[11]) -> bits[11] {
  ret add.3: bits[11] = add(x, y, id=3)
}

fn main() -> bits[11] {
  literal.4: bits[11] = literal(value=0, id=4)
  literal.5: bits[11] = literal(value=1, id=5)
  ret counted_for.6: bits[11] = counted_for(literal.4, trip_count=7, stride=1, body=body, invariant_args=[literal.5], id=6)
}
)";
  ParsePackageAndCheckDump(program);
}
TEST(IrParserTest, CountedForBodyParamCountTooMany0) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[16], x: bits[16], y: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)
  literal.3: bits[16] = literal(value=3)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=2, body=loop_fn,
      invariant_args=[literal.2])
}
)";

  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("counted_for body should have 3 parameters, "
                                 "got 4 instead")));
}

TEST(IrParserTest, CountedForBodyParamCountTooMany1) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[16], x: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=2, body=loop_fn)
}
)";

  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("counted_for body should have 2 parameters, "
                                 "got 3 instead")));
}

TEST(IrParserTest, CountedForBodyParamCountTooFew0) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[16], x: bits[16], y: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)
  literal.3: bits[16] = literal(value=3, id=3)
  literal.4: bits[16] = literal(value=4)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=2, body=loop_fn,
      invariant_args=[literal.2, literal.3, literal.4])
}
)";

  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("counted_for body should have 5 parameters, "
                                 "got 4 instead")));
}

TEST(IrParserTest, CountedForBodyParamCountTooFew1) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=2, body=loop_fn,
      invariant_args=[literal.2])
}
)";

  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("counted_for body should have 3 parameters, "
                                 "got 2 instead")));
}

TEST(IrParserTest, CountedForBodyParamCountTooFew2) {
  std::string program = R"(
package test

fn loop_fn() -> bits[16] {
  ret literal.10: bits[16] = literal(value=0)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=2, body=loop_fn)
}
)";

  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("counted_for body should have 2 parameters, "
                                 "got 0 instead")));
}

TEST(IrParserTest, CountedForBodyBitWidthSufficient0) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0, id=10)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=16, stride=1,
                                              body=loop_fn, id=100)
}
)";

  ParsePackageAndCheckDump(program);
}

TEST(IrParserTest, CountedForBodyBitWidthZeroIteration) {
  std::string program = R"(
package test

fn loop_fn(i: bits[1], data: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0, id=10)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=0, stride=1,
                                              body=loop_fn, id=100)
}
)";

  ParsePackageAndCheckDump(program);
}

TEST(IrParserTest, CountedForBodyBitWidthOneIteration) {
  std::string program = R"(
package test

fn loop_fn(i: bits[1], data: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0, id=10)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=1, stride=1,
                                              body=loop_fn, id=100)
}
)";

  ParsePackageAndCheckDump(program);
}

TEST(IrParserTest, CountedForBodyBitWidthInsufficient) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=17, body=loop_fn, id=100)
}
)";

  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("counted_for body should have bits[N] type, where "
                         "N >= 5, got bits[4]")));
}

TEST(IrParserTest, CountedForBodyBitWidthInsufficientWithStride) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0, id=10)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=16, stride=2,
                                              body=loop_fn)
}
)";

  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("counted_for body should have bits[N] type, where "
                         "N >= 5, got bits[4]")));
}

TEST(IrParserTest, CountedForBodyBitWidthTypeMismatch0) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4][1], data: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0, id=10)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=0, body=loop_fn, id=100)
}
)";

  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("counted_for body should have bits[N] type, where "
                         "N >= 1, got bits[4][1]")));
}

TEST(IrParserTest, CountedForBodyBitWidthTypeMismatch1) {
  std::string program = R"(
package test

fn loop_fn(i: (bits[4]), data: bits[16]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0, id=10)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=1, body=loop_fn, id=100)
}
)";

  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("counted_for body should have bits[N] type, where "
                         "N >= 1, got (bits[4])")));
}

TEST(IrParserTest, CountedForBodyDataTypeMismatch) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[13]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0, id=10)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=1, body=loop_fn, id=100)
}
)";

  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("should have bits[16] type, got bits[13] instead")));
}

TEST(IrParserTest, CountedForReturnTypeMismatch) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[16]) -> bits[15] {
  ret literal.10: bits[15] = literal(value=0, id=10)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[16] = literal(value=2, id=2)

  ret counted_for.100: bits[16] = counted_for(x, trip_count=1, body=loop_fn, id=100)
}
)";

  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("should have bits[16] type, got bits[15] instead")));
}

TEST(IrParserTest, CountedForBodyInvariantArgTypeMismatch0) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[16],
           x: bits[4], y: (bits[4], bits[4]), z: bits[4][1]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0, id=10)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[4] = literal(value=2, id=2)
  literal.3: bits[4] = literal(value=3, id=3)
  literal.4: bits[4] = literal(value=4, id=4)

  literal.112: bits[4] = literal(value=1, id=112)
  tuple.113: (bits[4], bits[4]) = tuple(literal.2, literal.3, id=113)
  array.114: bits[4][1] = array(literal.4, id=114)

  ret counted_for.200: bits[16] = counted_for(x, trip_count=2, body=loop_fn,
      invariant_args=[literal.112, array.114, array.114])
}
)";

  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Parameter 3 (y) of function loop_fn used as "
                         "counted_for body should have bits[4][1] type")));
}

TEST(IrParserTest, CountedForBodyInvariantArgTypeMismatch1) {
  std::string program = R"(
package test

fn loop_fn(i: bits[4], data: bits[16],
           x: bits[4], y: (bits[4], bits[4]), z: bits[4][1]) -> bits[16] {
  ret literal.10: bits[16] = literal(value=0)
}

fn f(x: bits[16]) -> bits[16] {
  literal.2: bits[4] = literal(value=2, id=2)
  literal.3: bits[4] = literal(value=3, id=3)
  literal.4: bits[4] = literal(value=4)

  literal.112: bits[4] = literal(value=1, id=112)
  tuple.113: (bits[4], bits[4]) = tuple(literal.2, literal.3, id=113)
  array.114: bits[4][1] = array(literal.4)

  ret counted_for.200: bits[16] = counted_for(x, trip_count=2, body=loop_fn,
      invariant_args=[literal.112, tuple.113, literal.112])
}
)";

  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Parameter 4 (z) of function loop_fn used as "
                                 "counted_for body should have bits[4] type")));
}

TEST(IrParserTest, ParseBitSlice) {
  std::string input = R"(
fn bitslice(x: bits[32]) -> bits[14] {
  ret bit_slice.1: bits[14] = bit_slice(x, start=7, width=14, id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseDynamicBitSlice) {
  std::string input = R"(
fn dynamicbitslice(x: bits[32], y: bits[32]) -> bits[14] {
  ret dynamic_bit_slice.1: bits[14] = dynamic_bit_slice(x, y, width=14, id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseAfterAllEmpty) {
  std::string input = R"(
fn after_all_func() -> token {
  ret after_all.1: token = after_all(id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseAfterAllMany) {
  std::string input = R"(
fn after_all_func() -> token {
  after_all.1: token = after_all(id=1)
  after_all.2: token = after_all(id=2)
  ret after_all.3: token = after_all(after_all.1, after_all.2, id=3)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseAfterAllNonToken) {
  Package p("my_package");
  std::string input = R"(
fn after_all_func() -> token {
  after_all.1: token = after_all(id=1)
  after_all.2: token = after_all(id=2)
  ret after_all.3: bits[2] = after_all(after_all.1, after_all.2, id=3)
}
)";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token type @")));
}

TEST(IrParserTest, ParseArray) {
  std::string input = R"(
fn array_and_array(x: bits[32], y: bits[32], z: bits[32]) -> bits[32][3] {
  ret array.1: bits[32][3] = array(x, y, z, id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseReverse) {
  std::string input = R"(
fn reverse(x: bits[32]) -> bits[32] {
  ret reverse.1: bits[32] = reverse(x, id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseArrayOfTuples) {
  std::string input = R"(
fn array_and_array(x: (bits[32], bits[1]), y: (bits[32], bits[1])) -> (bits[32], bits[1])[3] {
  ret array.1: (bits[32], bits[1])[3] = array(x, y, x, id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseNestedBitsArrayIndex) {
  std::string input = R"(
fn array_and_array(p: bits[2][5][4][42], q: bits[32], r: bits[2]) -> bits[2][5] {
  ret array_index.1: bits[2][5] = array_index(p, indices=[q, r], id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseNestedBitsArrayUpdate) {
  std::string input = R"(
fn array_and_array(p: bits[2][5][4][42], q: bits[32], v: bits[2][5][4]) -> bits[2][5][4][42] {
  ret array_update.1: bits[2][5][4][42] = array_update(p, v, indices=[q], id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, DifferentWidthMultiplies) {
  std::string input = R"(
fn multiply(x: bits[32], y: bits[7]) -> bits[42] {
  ret umul.1: bits[42] = umul(x, y, id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, EmptyBitsBounds) {
  Package p("my_package");
  std::string input = R"(fn f() -> bits[] {
  ret literal.1: bits[] = literal(value=0, id=1)
})";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \"literal\"")));
}

TEST(IrParserTest, ParseSingleEmptyPackage) {
  std::string input = R"(package EmptyPackage)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(input));
  EXPECT_EQ(package->name(), "EmptyPackage");
  EXPECT_EQ(0, package->functions().size());

  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseSingleFunctionPackage) {
  std::string input = R"(package SingleFunctionPackage

fn two_plus_two() -> bits[32] {
  literal.1: bits[32] = literal(value=2, id=1)
  literal.2: bits[32] = literal(value=2, id=2)
  ret add.3: bits[32] = add(literal.1, literal.2, id=3)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(input));
  EXPECT_EQ(package->name(), "SingleFunctionPackage");
  EXPECT_EQ(1, package->functions().size());
  Function* func = package->functions().front().get();
  EXPECT_EQ(func->name(), "two_plus_two");

  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseMultiFunctionPackage) {
  std::string input = R"(package MultiFunctionPackage

fn two_plus_two() -> bits[32] {
  literal.1: bits[32] = literal(value=2, id=1)
  literal.2: bits[32] = literal(value=2, id=2)
  ret add.3: bits[32] = add(literal.1, literal.2, id=3)
}

fn seven_and_five() -> bits[32] {
  literal.4: bits[32] = literal(value=7, id=4)
  literal.5: bits[32] = literal(value=5, id=5)
  ret and.6: bits[32] = and(literal.4, literal.5, id=6)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(input));
  EXPECT_EQ(package->name(), "MultiFunctionPackage");
  EXPECT_EQ(2, package->functions().size());
  EXPECT_EQ(package->functions()[0]->name(), "two_plus_two");
  EXPECT_EQ(package->functions()[1]->name(), "seven_and_five");

  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParsePackageWithError) {
  std::string input = R"(package MultiFunctionPackage

Garbage
)";
  EXPECT_THAT(Parser::ParsePackage(input).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected fn, proc, or chan definition")));
}

TEST(IrParserTest, ParseEmptyStringAsPackage) {
  EXPECT_THAT(Parser::ParsePackage("").status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token, but found EOF")));
}

TEST(IrParserTest, ParsePackageWithMissingPackageLine) {
  std::string input = R"(fn two_plus_two() -> bits[32] {
  literal.1: bits[32] = literal(value=2, id=1)
  literal.2: bits[32] = literal(value=2, id=2)
  ret add.3: bits[32] = add(literal.1, literal.2, id=3)
}
)";
  absl::Status status = Parser::ParsePackage(input).status();
  EXPECT_THAT(Parser::ParsePackage(input).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected 'package' keyword")));
}

TEST(IrParserTest, ParseBinaryConcat) {
  std::string input = R"(package p
fn concat_wrapper(x: bits[31], y: bits[1]) -> bits[32] {
  ret concat.1: bits[32] = concat(x, y, id=1)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(input));
  ASSERT_EQ(1, p->functions().size());
  std::unique_ptr<Function>& f = p->functions()[0];
  EXPECT_EQ(f->return_value()->op(), Op::kConcat);
  EXPECT_FALSE(f->return_value()->Is<BinOp>());
  EXPECT_TRUE(f->return_value()->Is<Concat>());
  EXPECT_EQ(p->GetBitsType(32), f->return_value()->GetType());
}

TEST(IrParserTest, ParseNaryConcat) {
  std::string input = R"(package p
fn concat_wrapper(x: bits[31], y: bits[1]) -> bits[95] {
  ret concat.1: bits[95] = concat(x, y, x, x, y, id=1)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(input));
  ASSERT_EQ(1, p->functions().size());
  std::unique_ptr<Function>& f = p->functions()[0];
  EXPECT_EQ(f->return_value()->op(), Op::kConcat);
  EXPECT_TRUE(f->return_value()->Is<Concat>());
  EXPECT_EQ(p->GetBitsType(95), f->return_value()->GetType());
}

TEST(IrParserTest, ParseMap) {
  std::string input = R"(
package SimpleMap

fn to_apply(element: bits[42]) -> bits[1] {
  literal.2: bits[42] = literal(value=10, id=2)
  ret ult.3: bits[1] = ult(element, literal.2, id=3)
}

fn top(input: bits[42][123]) -> bits[1][123] {
  ret map.5: bits[1][123] = map(input, to_apply=to_apply, id=5)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseBinarySel) {
  const std::string input = R"(
package ParseSel

fn sel_wrapper(x: bits[1], y: bits[32], z: bits[32]) -> bits[32] {
  ret sel.1: bits[32] = sel(x, cases=[y, z], id=1)
}
  )";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  ASSERT_EQ(pkg->functions().size(), 1);
  Function& f = *(pkg->functions()[0]);
  EXPECT_EQ(f.return_value()->op(), Op::kSel);
  EXPECT_FALSE(f.return_value()->Is<BinOp>());
  EXPECT_TRUE(f.return_value()->Is<Select>());
  EXPECT_EQ(f.return_value()->GetType(), pkg->GetBitsType(32));

  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseTernarySelectWithDefault) {
  const std::string input = R"(
package ParseSel

fn sel_wrapper(p: bits[2], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
  literal.1: bits[32] = literal(value=0, id=1)
  ret sel.2: bits[32] = sel(p, cases=[x, y, z], default=literal.1, id=2)
}
  )";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  ASSERT_EQ(pkg->functions().size(), 1);
  Function& f = *(pkg->functions()[0]);
  EXPECT_EQ(f.return_value()->op(), Op::kSel);
  EXPECT_FALSE(f.return_value()->Is<BinOp>());
  EXPECT_TRUE(f.return_value()->Is<Select>());
  EXPECT_EQ(f.return_value()->GetType(), pkg->GetBitsType(32));

  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseOneHotLsbPriority) {
  const std::string input = R"(
package ParseOneHot

fn sel_wrapper(x: bits[42]) -> bits[43] {
  ret one_hot.1: bits[43] = one_hot(x, lsb_prio=true, id=1)
}
  )";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseOneHotMsbPriority) {
  const std::string input = R"(
package ParseOneHot

fn sel_wrapper(x: bits[42]) -> bits[43] {
  ret one_hot.1: bits[43] = one_hot(x, lsb_prio=false, id=1)
}
  )";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseOneHotSelect) {
  const std::string input = R"(
package ParseOneHotSel

fn sel_wrapper(p: bits[3], x: bits[32], y: bits[32], z: bits[32]) -> bits[32] {
  ret one_hot_sel.1: bits[32] = one_hot_sel(p, cases=[x, y, z], id=1)
}
  )";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseParamReturn) {
  std::string input = R"(
package ParseParamReturn

fn simple_neg(x: bits[2]) -> bits[2] {
  ret x: bits[2] = param(name=x)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseInvoke) {
  const std::string input = R"(package foobar

fn bar(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y, id=1)
}

fn foo(x: bits[32]) -> bits[32] {
  literal.2: bits[32] = literal(value=5, id=2)
  ret invoke.3: bits[32] = invoke(x, literal.2, to_apply=bar, id=3)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseSimpleProc) {
  const std::string input = R"(package test

chan ch(data: bits[32], id=0, kind=streaming, ops=send_receive, metadata="""""")

proc my_proc(my_token: token, my_state: bits[32], init=42) {
  send.1: token = send(my_token, data=[my_state], channel_id=0, id=1)
  receive.2: (token, bits[32]) = receive(send.1, channel_id=0, id=2)
  tuple_index.3: token = tuple_index(receive.2, index=0, id=3)
  next (tuple_index.3, my_state)
}
)";
  ParsePackageAndCheckDump(input);
}

TEST(IrParserTest, ParseArrayIndex) {
  const std::string input = R"(
fn foo(x: bits[32][6]) -> bits[32] {
  literal.1: bits[32] = literal(value=5, id=1)
  ret array_index.2: bits[32] = array_index(x, indices=[literal.1], id=2)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseArrayUpdate) {
  const std::string input = R"(
fn foo(array: bits[32][3], idx: bits[32], newval: bits[32]) -> bits[32][3] {
  ret array_update.4: bits[32][3] = array_update(array, newval, indices=[idx], id=4)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseArrayUpdateNonArary) {
  const std::string input = R"(
fn foo(array: bits[32], idx: bits[32], newval: bits[32]) -> bits[32][3] {
  ret array_update.4: bits[32][3] = array_update(array, newval, indices=[idx],  id=4)
}
)";
  Package p("my_package");
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Too many indices (1) to index into array of type bits[32]")));
}

TEST(IrParserTest, ParseArrayUpdateIncompatibleTypes) {
  const std::string input = R"(
fn foo(array: bits[32][3], idx: bits[32], newval: bits[64]) -> bits[32][3] {
  ret array_update.4: bits[32][3] = array_update(array, newval, indices=[idx], id=4)
}
)";
  Package p("my_package");
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected update value to have type bits[32]; "
                                 "has type bits[64]")));
}

TEST(IrParserTest, ParseArrayConcat0) {
  const std::string input = R"(
fn foo(a0: bits[32][3], a1: bits[32][1]) -> bits[32][4] {
  ret array_concat.3: bits[32][4] = array_concat(a0, a1, id=3)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseArrayConcat1) {
  const std::string input = R"(
fn foo(a0: bits[32][0], a1: bits[32][1]) -> bits[32][1] {
  ret array_concat.3: bits[32][1] = array_concat(a0, a1, id=3)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseArrayConcatMixedOperands) {
  const std::string input = R"(
fn f(a0: bits[32][2], a1: bits[32][3], a2: bits[32][1]) -> bits[32][7] {
  array_concat.4: bits[32][1] = array_concat(a2, id=4)
  array_concat.5: bits[32][2] = array_concat(array_concat.4, array_concat.4, id=5)
  array_concat.6: bits[32][7] = array_concat(a0, array_concat.5, a1, id=6)
  ret array_concat.7: bits[32][7] = array_concat(array_concat.6, id=7)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseArrayConcatNonArrayType) {
  const std::string input = R"(
fn foo(a0: bits[16], a1: bits[16][1]) -> bits[16][2] {
  ret array_concat.3: bits[16][2] = array_concat(a0, a1, id=3)
}
)";
  Package p("my_package");
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot array-concat node a0 because it has "
                                 "non-array type bits[16]")));
}

TEST(IrParserTest, ParseArrayIncompatibleElementType) {
  const std::string input = R"(
fn foo(a0: bits[16][1], a1: bits[32][1]) -> bits[16][2] {
  ret array_concat.3: bits[16][2] = array_concat(a0, a1, id=3)
}
)";
  Package p("my_package");
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Cannot array-concat node a1 because it has element "
                         "type bits[32] but expected bits[16]")));
}

TEST(IrParserTest, ParseArrayIncompatibleReturnType) {
  const std::string input = R"(
fn foo(a0: bits[16][1], a1: bits[16][1]) -> bits[16][3] {
  ret array_concat.3: bits[16][3] = array_concat(a0, a1, id=3)
}
)";
  Package p("my_package");
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Declared type bits[16][3] does not match "
                                 "expected type bits[16][2]")));
}

TEST(IrParserTest, ParseTupleIndex) {
  const std::string input = R"(
fn foo(x: bits[42]) -> bits[33] {
  literal.1: bits[32] = literal(value=5, id=1)
  literal.2: bits[33] = literal(value=123, id=2)
  tuple.3: (bits[42], bits[32], bits[33]) = tuple(x, literal.1, literal.2, id=3)
  ret tuple_index.4: bits[33] = tuple_index(tuple.3, index=2, id=4)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseIdentity) {
  const std::string input = R"(
fn foo(x: bits[32]) -> bits[32] {
  ret identity.2: bits[32] = identity(x, id=2)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseUnsignedInequalities) {
  std::string program = R"(
fn parse_inequalities() -> bits[1] {
  literal.1: bits[32] = literal(value=2, id=1)
  literal.2: bits[32] = literal(value=2, id=2)
  uge.3: bits[1] = uge(literal.1, literal.2, id=3)
  ugt.4: bits[1] = ugt(literal.1, literal.2, id=4)
  ule.5: bits[1] = ule(literal.1, literal.2, id=5)
  ult.6: bits[1] = ult(literal.1, literal.2, id=6)
  ret eq.7: bits[1] = eq(literal.1, literal.2, id=7)
}
)";
  ParseFunctionAndCheckDump(program);
}

TEST(IrParserTest, ParseSignedInequalities) {
  std::string program = R"(
fn parse_inequalities() -> bits[1] {
  literal.1: bits[32] = literal(value=2, id=1)
  literal.2: bits[32] = literal(value=2, id=2)
  sge.3: bits[1] = sge(literal.1, literal.2, id=3)
  sgt.4: bits[1] = sgt(literal.1, literal.2, id=4)
  sle.5: bits[1] = sle(literal.1, literal.2, id=5)
  slt.6: bits[1] = slt(literal.1, literal.2, id=6)
  ret eq.7: bits[1] = eq(literal.1, literal.2, id=7)
}
)";
  ParseFunctionAndCheckDump(program);
}

TEST(IrParserTest, StandAloneRet) {
  const std::string input = R"(package foobar

fn foo(x: bits[32]) -> bits[32] {
  identity.2: bits[32] = identity(x, id=2)
  ret identity.2
}
)";

  EXPECT_THAT(Parser::ParsePackage(input).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \":\"")));
}

TEST(IrParserTest, ParseEndOfLineComment) {
  const std::string input = R"(// top comment
package foobar
// a comment

fn foo(x: bits[32]) -> bits[32] {  // another comment
  ret identity.2: bits[32] = identity(x)  // yep, another one

// comment

}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  ASSERT_EQ(pkg->functions().size(), 1);
}

TEST(IrParserTest, ParseTupleType) {
  const std::string input = R"(
    package foobar

    fn foo(x: bits[32]) -> (bits[32], bits[32]) {
       ret tuple.1: (bits[32], bits[32]) = tuple(x, x, id=1)
    }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  ASSERT_EQ(pkg->functions().size(), 1);
  Type* t = pkg->functions()[0]->return_value()->GetType();
  EXPECT_TRUE(t->IsTuple());
  EXPECT_EQ(t->AsTupleOrDie()->size(), 2);
  EXPECT_TRUE(t->AsTupleOrDie()->element_type(0)->IsBits());
  EXPECT_TRUE(t->AsTupleOrDie()->element_type(1)->IsBits());
}

TEST(IrParserTest, ParseEmptyTuple) {
  const std::string input = R"(
    package foobar

    fn foo(x: bits[32]) -> () {
       ret tuple.1: () = tuple(id=1)
    }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  ASSERT_EQ(pkg->functions().size(), 1);
  Type* t = pkg->functions()[0]->return_value()->GetType();
  EXPECT_TRUE(t->IsTuple());
  EXPECT_EQ(t->AsTupleOrDie()->size(), 0);
}

TEST(IrParserTest, ParseNestedTuple) {
  const std::string input = R"(
    package foobar

    fn foo(x: bits[32]) -> ((bits[32], bits[32]), bits[32]) {
       tuple.1: (bits[32], bits[32]) = tuple(x, x, id=1)
       ret tuple.2: ((bits[32], bits[32]), bits[32]) = tuple(tuple.1, x, id=2)
    }
  )";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  ASSERT_EQ(pkg->functions().size(), 1);
  Type* t = pkg->functions()[0]->return_value()->GetType();
  EXPECT_TRUE(t->IsTuple());
  EXPECT_EQ(t->AsTupleOrDie()->size(), 2);
  EXPECT_TRUE(t->AsTupleOrDie()->element_type(0)->IsTuple());
  EXPECT_EQ(t->AsTupleOrDie()->element_type(0)->AsTupleOrDie()->size(), 2);
  EXPECT_TRUE(t->AsTupleOrDie()->element_type(1)->IsBits());
}

TEST(IrParserTest, ParseArrayLiterals) {
  const std::string input = R"(
fn foo(x: bits[32]) -> bits[32] {
  literal.1: bits[32][2] = literal(value=[0, 1], id=1)
  literal.2: bits[3] = literal(value=1, id=2)
  ret array_index.3: bits[32] = array_index(literal.1, indices=[literal.2], id=3)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseNestedArrayLiterals) {
  const std::string input = R"(
fn foo() -> bits[32][2][3][1] {
  ret literal.1: bits[32][2][3][1] = literal(value=[[[0, 1], [2, 3], [4, 5]]], id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseArrayLiteralWithInsufficientBits) {
  Package p("my_package");
  const std::string input = R"(
fn foo() -> bits[7][2] {
  ret literal.1: bits[7][2] = literal(value=[0, 12345], id=1)
}
)";
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Value 12345 is not representable in 7 bits")));
}

TEST(IrParserTest, ReturnArrayLiteral) {
  const std::string input = R"(
package foobar

fn foo(x: bits[32]) -> bits[32][2] {
  ret literal.1: bits[32][2] = literal(value=[0, 1], id=1)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  ASSERT_EQ(pkg->functions().size(), 1);
  Type* t = pkg->functions()[0]->return_value()->GetType();
  EXPECT_TRUE(t->IsArray());
}

TEST(IrParserTest, ReturnArrayOfTuplesLiteral) {
  const std::string input = R"(
package foobar

fn foo() -> (bits[32], bits[3])[2] {
  ret literal.1: (bits[32], bits[3])[2] = literal(value=[(2, 2), (0, 1)], id=1)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           Parser::ParsePackage(input));
  ASSERT_EQ(pkg->functions().size(), 1);
  Type* t = pkg->functions()[0]->return_value()->GetType();
  EXPECT_TRUE(t->IsArray());
}

TEST(IrParserTest, ArrayValueInBitsLiteral) {
  Package p("my_package");
  const std::string input = R"(
fn foo() -> bits[42] {
  ret literal.1: bits[42] = literal(value=[0, 123], id=1)
}
)";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \"literal\"")));
}

TEST(IrParserTest, BitsValueInArrayLiteral) {
  Package p("my_package");
  const std::string input = R"(
fn foo() -> bits[7][42] {
  ret literal.1: bits[7][42] = literal(value=123], id=1)
}
)";
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \"[\"")));
}

TEST(IrParserTest, ParseTupleLiteral) {
  const std::string input = R"(
fn foo() -> (bits[32][2], bits[1]) {
  ret literal.1: (bits[32][2], bits[1]) = literal(value=([123, 456], 0), id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseNestedTupleLiteral) {
  const std::string input = R"(
fn foo() -> (bits[32][2], bits[1], (), (bits[44])) {
  ret literal.1: (bits[32][2], bits[1], (), (bits[44])) = literal(value=([123, 456], 0, (), (10)), id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseNaryXor) {
  const std::string input = R"(
fn foo(x: bits[8]) -> bits[8] {
  ret xor.2: bits[8] = xor(x, x, x, x, id=2)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseExtendOps) {
  const std::string input = R"(
fn foo(x: bits[8]) -> bits[32] {
  zero_ext.1: bits[32] = zero_ext(x, new_bit_count=32, id=1)
  sign_ext.2: bits[32] = sign_ext(x, new_bit_count=32, id=2)
  ret xor.3: bits[32] = xor(zero_ext.1, sign_ext.2, id=3)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseInconsistentExtendOp) {
  const std::string input = R"(
fn foo(x: bits[8]) -> bits[32] {
  ret zero_ext.1: bits[33] = zero_ext(x, new_bit_count=32, id=1)
}
)";
  Package p("my_package");
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("differs from its new_bit_count annotation 32")));
}

TEST(IrParserTest, ParseDecode) {
  const std::string input = R"(
fn foo(x: bits[8]) -> bits[256] {
  decode.1: bits[42] = decode(x, width=42, id=1)
  ret decode.2: bits[256] = decode(x, width=256, id=2)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ParseEncode) {
  const std::string input = R"(
fn foo(x: bits[16]) -> bits[4] {
  ret encode.1: bits[4] = encode(x, id=1)
}
)";
  ParseFunctionAndCheckDump(input);
}

TEST(IrParserTest, ArrayIndexOfTuple) {
  const std::string input = R"(
fn foo(x: (bits[8])) -> bits[32] {
  literal.1: bits[32] = literal(value=0, id=1)
  ret array_index.2: bits[8] = array_index(x, indices=[literal.1], id=2)
}
)";
  Package p("my_package");
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Too many indices (1) to index into array of type (bits[8])")));
}

TEST(IrParserTest, TupleIndexOfArray) {
  const std::string input = R"(
fn foo(x: bits[8][5]) -> bits[8] {
  ret tuple_index.1: bits[8] = tuple_index(x, index=0, id=1)
}
)";
  Package p("my_package");
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("tuple_index operand is not a tuple")));
}

TEST(IrParserTest, NicerErrorOnEmptyString) {
  const std::string input = "";
  EXPECT_THAT(
      Parser::ParsePackage(input).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Expected keyword 'package': Expected token, but found EOF.")));
}

TEST(IrParserTest, ParsesComplexValue) {
  const std::string input = "(0xf00, [0xba5, 0xba7], [0])";
  Package p("test_package");
  auto* u32 = p.GetBitsType(32);
  auto* u12 = p.GetBitsType(12);
  auto* u1 = p.GetBitsType(1);
  auto* array_1xu1 = p.GetArrayType(1, u1);
  auto* array_2xu12 = p.GetArrayType(2, u12);
  auto* overall = p.GetTupleType({u32, array_2xu12, array_1xu1});
  XLS_ASSERT_OK_AND_ASSIGN(Value v, Parser::ParseValue(input, overall));
  Value expected = Value::Tuple({
      Value(UBits(0xf00, /*bit_count=*/32)),
      Value::ArrayOrDie({
          Value(UBits(0xba5, /*bit_count=*/12)),
          Value(UBits(0xba7, /*bit_count=*/12)),
      }),
      Value::ArrayOrDie({Value(UBits(0, /*bit_count=*/1))}),
  });
  EXPECT_EQ(expected, v);
}

TEST(IrParserTest, ParsesComplexValueWithEmbeddedTypes) {
  const std::string input =
      "(bits[32]:0xf00, [bits[12]:0xba5, bits[12]:0xba7], [bits[1]:0])";
  XLS_ASSERT_OK_AND_ASSIGN(Value v, Parser::ParseTypedValue(input));
  Value expected = Value::Tuple({
      Value(UBits(0xf00, /*bit_count=*/32)),
      Value::ArrayOrDie({
          Value(UBits(0xba5, /*bit_count=*/12)),
          Value(UBits(0xba7, /*bit_count=*/12)),
      }),
      Value::ArrayOrDie({Value(UBits(0, /*bit_count=*/1))}),
  });
  EXPECT_EQ(expected, v);
}

TEST(IrParserTest, ParsesTokenType) {
  const std::string input = "token";
  XLS_ASSERT_OK_AND_ASSIGN(Value v, Parser::ParseTypedValue(input));
  Value expected = Value::Token();
  EXPECT_EQ(expected, v);
}

TEST(IrParserTest, ParsesComplexValueWithEmbeddedTokens) {
  const std::string input =
      "(bits[32]:0xf00, [bits[12]:0xba5, bits[12]:0xba7], [token, token], "
      "[bits[1]:0], token)";
  XLS_ASSERT_OK_AND_ASSIGN(Value v, Parser::ParseTypedValue(input));
  Value expected = Value::Tuple({
      Value(UBits(0xf00, /*bit_count=*/32)),
      Value::ArrayOrDie({
          Value(UBits(0xba5, /*bit_count=*/12)),
          Value(UBits(0xba7, /*bit_count=*/12)),
      }),
      Value::ArrayOrDie({Value::Token(), Value::Token()}),
      Value::ArrayOrDie({Value(UBits(0, /*bit_count=*/1))}),
      Value::Token(),
  });
  EXPECT_EQ(expected, v);
}

// TODO(leary): 2019-08-01 Figure out if we want to reify the type into the
// empty array Value.
TEST(IrParserTest, DISABLED_ParsesEmptyArray) {
  const std::string input = "[]";
  Package p("test_package");
  auto* u1 = p.GetBitsType(1);
  auto* array_0xu1 = p.GetArrayType(0, u1);
  XLS_ASSERT_OK_AND_ASSIGN(Value v, Parser::ParseValue(input, array_0xu1));
  Value expected = Value::ArrayOrDie({});
  EXPECT_EQ(expected, v);
}

TEST(IrParserTest, BigOrdinalAnnotation) {
  std::string program = R"(
package test

fn main() -> bits[1] {
  ret literal.1000: bits[1] = literal(value=0, id=1000)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  EXPECT_GT(package->next_node_id(), 1000);
}

TEST(IrParserTest, TrivialProc) {
  std::string program = R"(
package test

proc foo(my_token: token, my_state: bits[32], init=42) {
  next (my_token, my_state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  EXPECT_EQ(package->functions().size(), 0);
  EXPECT_EQ(package->procs().size(), 1);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("foo"));
  EXPECT_EQ(proc->node_count(), 2);
  EXPECT_EQ(proc->params().size(), 2);
  EXPECT_EQ(proc->InitValue().ToString(), "bits[32]:42");
  EXPECT_EQ(proc->StateType()->ToString(), "bits[32]");
  EXPECT_EQ(proc->TokenParam()->GetName(), "my_token");
  EXPECT_EQ(proc->StateParam()->GetName(), "my_state");
}

TEST(IrParserTest, FunctionAndProc) {
  std::string program = R"(
package test

fn my_function() -> bits[1] {
  ret literal.1: bits[1] = literal(value=0, id=1)
}

proc my_proc(my_token: token, my_state: bits[32], init=42) {
  next (my_token, my_state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  EXPECT_EQ(package->functions().size(), 1);
  EXPECT_EQ(package->procs().size(), 1);
  XLS_ASSERT_OK_AND_ASSIGN(Function * function,
                           package->GetFunction("my_function"));
  EXPECT_EQ(function->name(), "my_function");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));
  EXPECT_EQ(proc->name(), "my_proc");
}

TEST(IrParserTest, ProcWrongParameterCount) {
  std::string program = R"(
package test

proc foo(my_token: token, my_state: bits[32], total_garbage: bits[1], init=42) {
  ret tuple.1: (token, bits[32]) = tuple(my_token, my_state, id=1)
}
)";
  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected 'init' attribute")));
}

TEST(IrParserTest, ProcWrongTokenType) {
  std::string program = R"(
package test

proc foo(my_token: bits[1], my_state: bits[32], init=42) {
  ret tuple.1: (token, bits[32]) = tuple(my_token, my_state, id=1)
}
)";
  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Expected second argument of proc to be token type")));
}

TEST(IrParserTest, ProcWrongInitValueType) {
  std::string program = R"(
package test

proc foo(my_token: token, my_state: bits[32], init=(1, 2, 3)) {
  ret tuple.1: (token, bits[32]) = tuple(my_token, my_state, id=1)
}
)";
  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \"literal\"")));
}

TEST(IrParserTest, ProcWrongReturnType) {
  std::string program = R"(
package test

proc foo(my_token: token, my_state: bits[32], init=42) {
  literal.1: bits[32] = literal(value=123, id=1)
  next (literal.1, my_state)
}
)";
  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Recurrent token of proc must be token type, is: bits[32]")));
}

TEST(IrParserTest, ProcWithRet) {
  std::string program = R"(
package test

proc foo(my_token: token, my_state: bits[32], init=42) {
  ret literal.1: bits[32] = literal(value=123, id=1)
}
)";
  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("ret keyword only supported in functions")));
}

TEST(IrParserTest, FunctionWithNext) {
  std::string program = R"(
package test

fn foo(x: bits[32]) -> bits[32] {
  next (x, x)
}
)";
  EXPECT_THAT(Parser::ParsePackage(program).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("next keyword only supported in procs")));
}

TEST(IrParserTest, ProcWithBogusNextToken) {
  std::string program = R"(
package test

proc foo(my_token: token, my_state: bits[32], init=42) {
  next (foobar, my_state)
}
)";
  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Proc next token name @ 5:9  was not previously defined")));
}

TEST(IrParserTest, ProcWithBogusNextState) {
  std::string program = R"(
package test

proc foo(my_token: token, my_state: bits[32], init=42) {
  next (my_token, sfsdfsfd)
}
)";
  EXPECT_THAT(
      Parser::ParsePackage(program).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr(
                   "Proc next state name @ 5:19  was not previously defined")));
}

TEST(IrParserTest, ParseSendReceiveChannel) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch,
                           Parser::ParseChannel(
                               R"(chan foo(foo_data: bits[32], id=42, kind=port,
                      ops=send_receive,
                      metadata="module_port { flopped: true }"))",
                               &p));
  EXPECT_EQ(ch->name(), "foo");
  EXPECT_EQ(ch->id(), 42);
  EXPECT_EQ(ch->supported_ops(), Channel::SupportedOps::kSendReceive);
  EXPECT_TRUE(ch->IsPort());
  EXPECT_EQ(ch->data_elements().size(), 1);
  EXPECT_EQ(ch->data_elements().front().name, "foo_data");
  EXPECT_EQ(ch->data_elements().front().type, p.GetBitsType(32));
  EXPECT_TRUE(ch->data_elements().front().initial_values.empty());
  EXPECT_EQ(ch->metadata().channel_oneof_case(),
            ChannelMetadataProto::kModulePort);
  EXPECT_TRUE(ch->metadata().module_port().flopped());
}

TEST(IrParserTest, ParseSendReceiveChannelWithInitialValues) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch,
      Parser::ParseChannel(
          R"(chan foo(foo_data: bits[32] = {2, 4, 5}, id=42, kind=streaming, ops=send_receive,
                         metadata="module_port { flopped: true }"))",
          &p));
  EXPECT_EQ(ch->name(), "foo");
  EXPECT_EQ(ch->id(), 42);
  EXPECT_EQ(ch->supported_ops(), Channel::SupportedOps::kSendReceive);
  EXPECT_TRUE(ch->IsStreaming());
  EXPECT_EQ(ch->data_elements().size(), 1);
  EXPECT_EQ(ch->data_elements().front().name, "foo_data");
  EXPECT_EQ(ch->data_elements().front().type, p.GetBitsType(32));
  EXPECT_THAT(ch->data_elements().front().initial_values,
              ElementsAre(Value(UBits(2, 32)), Value(UBits(4, 32)),
                          Value(UBits(5, 32))));
  EXPECT_EQ(ch->metadata().channel_oneof_case(),
            ChannelMetadataProto::kModulePort);
  EXPECT_TRUE(ch->metadata().module_port().flopped());
}

TEST(IrParserTest, ParseSendReceiveChannelWithEmptyInitialValues) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch,
      Parser::ParseChannel(
          R"(chan foo(foo_data: bits[32] = {}, id=42, kind=register,
                      ops=send_receive,
                      metadata="module_port { flopped: true }"))",
          &p));
  EXPECT_EQ(ch->name(), "foo");
  EXPECT_EQ(ch->data_elements().size(), 1);
  EXPECT_TRUE(ch->IsRegister());
  EXPECT_TRUE(ch->data_elements().front().initial_values.empty());
}

TEST(IrParserTest, ParseSendReceiveChannelWithTupleType) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch,
      Parser::ParseChannel(
          R"(chan foo(foo_data: (bits[32], bits[1]) = {(123, 1), (42, 0)},
                      id=42, kind=streaming,  ops=send_receive,
                      metadata="module_port { flopped: true }"))",
          &p));
  EXPECT_EQ(ch->name(), "foo");
  EXPECT_EQ(ch->data_elements().size(), 1);
  EXPECT_THAT(
      ch->data_elements().front().initial_values,
      ElementsAre(Value::Tuple({Value(UBits(123, 32)), Value(UBits(1, 1))}),
                  Value::Tuple({Value(UBits(42, 32)), Value(UBits(0, 1))})));
}

TEST(IrParserTest, ParseMultipleDataSendReceiveChannelWithInitialValues) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch,
                           Parser::ParseChannel(
                               R"(chan foo(foo_data: bits[32] = {2, 4},
                      bar_data: bits[8] = {0, 42},
                      id=42, kind=streaming, ops=send_receive,
                      metadata="module_port { flopped: true }"))",
                               &p));
  EXPECT_EQ(ch->name(), "foo");
  EXPECT_EQ(ch->id(), 42);
  EXPECT_EQ(ch->supported_ops(), Channel::SupportedOps::kSendReceive);
  EXPECT_EQ(ch->data_elements().size(), 2);

  EXPECT_EQ(ch->data_element(0).name, "foo_data");
  EXPECT_EQ(ch->data_element(0).type, p.GetBitsType(32));
  EXPECT_THAT(ch->data_element(0).initial_values,
              ElementsAre(Value(UBits(2, 32)), Value(UBits(4, 32))));

  EXPECT_EQ(ch->data_element(1).name, "bar_data");
  EXPECT_EQ(ch->data_element(1).type, p.GetBitsType(8));
  EXPECT_THAT(ch->data_element(1).initial_values,
              ElementsAre(Value(UBits(0, 8)), Value(UBits(42, 8))));

  EXPECT_EQ(ch->metadata().channel_oneof_case(),
            ChannelMetadataProto::kModulePort);
  EXPECT_TRUE(ch->metadata().module_port().flopped());
}

TEST(IrParserTest, ParseSendOnlyChannel) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch, Parser::ParseChannel(
                        R"(chan bar(baz: (bits[32], bits[1]), qux: bits[123],
                         id=7, kind=port, ops=send_only,
                         metadata="module_port { flopped: false }"))",
                        &p));
  EXPECT_EQ(ch->name(), "bar");
  EXPECT_EQ(ch->id(), 7);
  EXPECT_EQ(ch->supported_ops(), Channel::SupportedOps::kSendOnly);
  EXPECT_EQ(ch->data_elements().size(), 2);
  EXPECT_EQ(ch->data_elements()[0].name, "baz");
  EXPECT_EQ(ch->data_elements()[0].type,
            p.GetTupleType({p.GetBitsType(32), p.GetBitsType(1)}));
  EXPECT_EQ(ch->data_elements()[1].name, "qux");
  EXPECT_EQ(ch->data_elements()[1].type, p.GetBitsType(123));
  EXPECT_EQ(ch->metadata().channel_oneof_case(),
            ChannelMetadataProto::kModulePort);
  EXPECT_FALSE(ch->metadata().module_port().flopped());
}

TEST(IrParserTest, ParseReceiveOnlyChannel) {
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch, Parser::ParseChannel(
                                             R"(chan meh(huh: bits[32][4], id=0,
                         kind=port, ops=receive_only,
                         metadata="module_port { flopped: true }"))",
                                             &p));
  EXPECT_EQ(ch->name(), "meh");
  EXPECT_EQ(ch->id(), 0);
  EXPECT_EQ(ch->supported_ops(), Channel::SupportedOps::kReceiveOnly);
  EXPECT_EQ(ch->data_elements().size(), 1);
  EXPECT_EQ(ch->data_elements().front().name, "huh");
  EXPECT_EQ(ch->data_elements().front().type,
            p.GetArrayType(4, p.GetBitsType(32)));
  EXPECT_EQ(ch->metadata().channel_oneof_case(),
            ChannelMetadataProto::kModulePort);
  EXPECT_TRUE(ch->metadata().module_port().flopped());
}

TEST(IrParserTest, ChannelParsingErrors) {
  Package p("my_package");
  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(huh: bits[32][4], kind=port,
                         ops=receive_only,
                         metadata="module_port { flopped: true }"))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Missing channel id")));

  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(huh: bits[32][4], id=42, ops=receive_only,
                         metadata="module_port { flopped: true }"))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Missing channel kind")));

  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(huh: bits[32][4], id=42, kind=bogus,
                         ops=receive_only,
                         metadata="module_port { flopped: true }"))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid channel kind \"bogus\"")));

  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(huh: bits[32][4], id=7, kind=streaming,
                         metadata="module_port { flopped: true }"))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Missing channel ops")));

  // Unrepresentable initial value.
  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(huh: bits[4] = {128}, kind=streaming,
                         ops=send_receive, id=7,
                         metadata="module_port { flopped: true }"))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Value 128 is not representable in 4 bits")));

  // Wrong initial value type.
  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(huh: bits[4] = {(1, 2)}, kind=streaming,
                         ops=send_receive, id=7
                         metadata="module_port { flopped: true }"))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \"literal\"")));

  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(huh: bits[32][4], id=7, kind=streaming,
                     ops=receive_only))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Missing channel metadata")));

  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(id=44, kind=streaming, ops=receive_only,
                         metadata="module_port { flopped: true }"))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Channel has no data elements")));

  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(foo: bits[32], id=44, kind=streaming,
                         ops=receive_only, bogus="totally!",
                         metadata="module_port { flopped: true }"))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid channel attribute \"bogus\"")));

  // Bad data element name.
  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan meh(123badname: bits[32][4], id=7, kind=streaming,
             ops=receive_only, metadata=""))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \"ident\"")));

  // Bad channel name.
  EXPECT_THAT(Parser::ParseChannel(
                  R"(chan 444meh(foo: bits[32], id=7, kind=streaming,
                         ops=receive_only, metadata=""))",
                  &p)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected token of type \"ident\"")));
}

TEST(IrParserTest, PackageWithSingleDataElementChannels) {
  std::string program = R"(
package test

chan hbo(junk: bits[32], id=0, kind=streaming, ops=receive_only,
            metadata="module_port { flopped: true }")
chan mtv(stuff: bits[32], id=1, kind=streaming, ops=send_only,
            metadata="module_port { flopped: true }")

proc my_proc(my_token: token, my_state: bits[32], init=42) {
  receive.1: (token, bits[32]) = receive(my_token, channel_id=0)
  tuple_index.2: token = tuple_index(receive.1, index=0, id=2)
  tuple_index.3: bits[32] = tuple_index(receive.1, index=1, id=3)
  add.4: bits[32] = add(my_state, tuple_index.3, id=4)
  send.5: token = send(tuple_index.2, data=[add.4], channel_id=1)
  next (send.5, add.4)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  EXPECT_EQ(package->procs().size(), 1);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));
  EXPECT_EQ(proc->name(), "my_proc");
}

TEST(IrParserTest, PackageWithMultipleDataElementChannels) {
  std::string program = R"(
package test

chan hbo(junk: bits[32], garbage: bits[1], kind=streaming, id=0,
         ops=receive_only,
         metadata="module_port { flopped: true }")
chan mtv(stuff: bits[32], zzz: bits[1], kind=streaming, id=1, ops=send_only,
          metadata="module_port { flopped: true }")

proc my_proc(my_token: token, my_state: bits[32], init=42) {
  literal.1: bits[1] = literal(value=1, id=1)
  receive_if.2: (token, bits[32], bits[1]) = receive_if(my_token, literal.1, channel_id=0)
  tuple_index.3: token = tuple_index(receive_if.2, index=0, id=3)
  tuple_index.4: bits[32] = tuple_index(receive_if.2, index=1, id=4)
  tuple_index.5: bits[1] = tuple_index(receive_if.2, index=2, id=5)
  send_if.6: token = send_if(tuple_index.3, literal.1, data=[tuple_index.4, tuple_index.5], channel_id=1)
  next (send_if.6, my_state)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  EXPECT_EQ(package->procs().size(), 1);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, package->GetProc("my_proc"));
  EXPECT_EQ(proc->name(), "my_proc");
}

TEST(IrParserTest, ParseTupleIndexWithInvalidBValue) {
  const std::string input = R"(
fn f(x: bits[4], y: bits[4][1]) -> bits[4] {
  onehot.10: bits[16] = decode(y, width=16, id=10)
  ret ind.20: bits[4] = tuple_index(onehot.10, index=0, id=20)
}
)";

  Package p("my_package");
  EXPECT_THAT(Parser::ParseFunction(input, &p).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Decode argument must be of Bits type")));
}

TEST(IrParserTest, NodeNames) {
  std::string program = R"(package test

fn foo(x: bits[32], foobar: bits[32]) -> bits[32] {
  add.1: bits[32] = add(x, foobar, id=1)
  ret qux: bits[32] = not(add.1, id=123)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, package->GetFunction("foo"));
  EXPECT_EQ(package->DumpIr(), program);

  Node* x = f->param(0);
  EXPECT_TRUE(x->HasAssignedName());
  EXPECT_EQ(x->GetName(), "x");
  EXPECT_EQ(x->id(), 1);

  Node* foobar = f->param(1);
  EXPECT_TRUE(foobar->HasAssignedName());
  EXPECT_EQ(foobar->GetName(), "foobar");
  EXPECT_EQ(foobar->id(), 2);

  Node* add = f->return_value()->operand(0);
  EXPECT_FALSE(add->HasAssignedName());
  EXPECT_EQ(add->GetName(), "add.1");
  EXPECT_EQ(add->id(), 1);

  Node* qux = f->return_value();
  EXPECT_TRUE(qux->HasAssignedName());
  EXPECT_EQ(qux->GetName(), "qux");
}

TEST(IrParserTest, InvalidName) {
  const std::string input = R"(
fn f(x: bits[4]) -> bits[4] {
  ret blahblah.30: bits[4] = add(x, x, id=30)
}
)";
  Package p("my_package");
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("The substring 'blahblah' in node name blahblah.30 "
                         "does not match the node op 'add")));
}

TEST(IrParserTest, IdAttributes) {
  const std::string input = R"(
fn f(x: bits[4]) -> bits[4] {
  foo: bits[4] = not(x)
  bar: bits[4] = not(foo, id=42)
  not.123: bits[4] = not(bar, id=123)
  ret not.333: bits[4] = not(not.123, id=333)
}
)";
  Package p("my_package");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, Parser::ParseFunction(input, &p));
  EXPECT_EQ(f->return_value()->id(), 333);
  EXPECT_EQ(f->return_value()->operand(0)->id(), 123);
  EXPECT_EQ(f->return_value()->operand(0)->operand(0)->id(), 42);
  EXPECT_EQ(f->return_value()->operand(0)->operand(0)->operand(0)->id(), 2);
}

TEST(IrParserTest, MismatchedId) {
  const std::string input = R"(
fn f(x: bits[4]) -> bits[4] {
  ret add.30: bits[4] = add(x, x, id=42)
}
)";
  Package p("my_package");
  EXPECT_THAT(
      Parser::ParseFunction(input, &p).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("The id '30' in node name add.30 does not match the "
                         "id '42' specified as an attribute")));
}

}  // namespace xls
