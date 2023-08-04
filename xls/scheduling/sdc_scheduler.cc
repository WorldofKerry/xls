// Copyright 2022 The XLS Authors
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

#include "xls/scheduling/sdc_scheduler.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/proc.h"
#include "xls/scheduling/schedule_bounds.h"
#include "xls/scheduling/scheduling_options.h"
#include "ortools/math_opt/cpp/math_opt.h"

namespace math_opt = ::operations_research::math_opt;

namespace xls {

namespace {

static constexpr double kInfinity = std::numeric_limits<double>::infinity();

using DelayMap = absl::flat_hash_map<Node*, int64_t>;

// A helper function to compute each node's delay by calling the delay estimator
absl::StatusOr<DelayMap> ComputeNodeDelays(
    FunctionBase* f, const DelayEstimator& delay_estimator) {
  DelayMap result;
  for (Node* node : f->nodes()) {
    XLS_ASSIGN_OR_RETURN(result[node],
                         delay_estimator.GetOperationDelayInPs(node));
  }
  return result;
}

// All transitive children (operands, operands of operands, etc.) of the given
// node.
absl::btree_set<Node*, Node::NodeIdLessThan> Descendants(Node* root) {
  std::vector<Node*> stack;
  stack.push_back(root);
  absl::btree_set<Node*, Node::NodeIdLessThan> discovered;
  while (!stack.empty()) {
    Node* popped = stack.back();
    stack.pop_back();
    if (!discovered.contains(popped)) {
      discovered.insert(popped);
      for (Node* child : popped->operands()) {
        stack.push_back(child);
      }
    }
  }
  return discovered;
}

// Returns the minimal set of schedule constraints which ensure that no
// combinational path in the schedule exceeds `clock_period_ps`. The returned
// map has a (potentially empty) vector entry for each node in `f`. The map
// value (vector of nodes) for node `x` is the set of nodes which must be
// scheduled at least one cycle later than `x`. That is, if `return_value[x]` is
// `S` then:
//
//   cycle(i) + 1 >= cycle(x) for i \in S
//
// The set of constraints is a minimal set which guarantees that no
// combinational path violates the clock period timing. Specifically, `(a, b)`
// is in the set of returned constraints (ie., `return_value[a]` contains `b`)
// iff critical-path distance from `a` to `b` including the delay of `a` and `b`
// is greater than `critical_path_period`, but the critical-path distance of the
// path *not* including the delay of `b` is *less than* `critical_path_period`.
absl::flat_hash_map<Node*, std::vector<Node*>>
ComputeCombinationalDelayConstraints(FunctionBase* f, int64_t clock_period_ps,
                                     const DelayMap& delay_map) {
  absl::flat_hash_map<Node*, std::vector<Node*>> result;
  result.reserve(f->node_count());

  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> result_as_set;
  result_as_set.reserve(f->node_count());

  // Compute all-pairs longest distance between all nodes in `f`. The distance
  // from node `a` to node `b` is defined as the length of the longest delay
  // path from `a` to `b` which includes the delay of the path endpoints `a` and
  // `b`. The all-pairs distance is stored in the map of maps
  // `distances_to_node` where `distances_to_node[y][x]` (if present) is the
  // critical-path distance from `x` to `y`.
  //
  // We use absl::btree_map on the inner map to guarantee a deterministic
  // iteration order, since that affects the constraints we will add.
  absl::flat_hash_map<Node*,
                      absl::btree_map<Node*, int64_t, Node::NodeIdLessThan>>
      distances_to_node;
  distances_to_node.reserve(f->node_count());
  for (Node* node : f->nodes()) {
    // Initialize the constraint map entry to an empty vector.
    result[node];
    // Initialize the constraint map record to an empty set.
    result_as_set[node];
    // Initialize the distance map entry to an empty map.
    distances_to_node[node];
  }

  for (Node* node : TopoSort(f)) {
    absl::btree_map<Node*, int64_t, Node::NodeIdLessThan>& distances =
        distances_to_node.at(node);

    // The critical path from `node` to `node` is always `node_delay` long.
    int64_t node_delay = delay_map.at(node);
    distances_to_node.at(node)[node] = node_delay;

    // Compute the critical-path distance from `a` to `node` for all descendants
    // `a` of each operand, extending the critical path from `a` to each operand
    // of `node` by `node_delay`.
    for (Node* operand : node->operands()) {
      for (auto [a, operand_distance] : distances_to_node.at(operand)) {
        auto [it, newly_reachable] =
            distances.try_emplace(a, operand_distance + node_delay);
        if (!newly_reachable) {
          if (it->second >= operand_distance + node_delay) {
            continue;
          }
          it->second = operand_distance + node_delay;
        }
        if (operand_distance > clock_period_ps ||
            operand_distance + node_delay <= clock_period_ps) {
          continue;
        }
        auto [_, inserted] = result_as_set.at(a).insert(node);
        if (inserted) {
          result.at(a).push_back(node);
        }
      }
    }
  }

  if (XLS_VLOG_IS_ON(4)) {
    XLS_VLOG(4) << "All-pairs critical-path distances:";
    for (Node* target : TopoSort(f)) {
      XLS_VLOG(4) << absl::StrFormat("  distances to %s:", target->GetName());
      for (Node* source : TopoSort(f)) {
        XLS_VLOG(4) << absl::StrFormat(
            "    %s -> %s : %s", source->GetName(), target->GetName(),
            distances_to_node.at(target).contains(source)
                ? absl::StrCat(distances_to_node.at(target).at(source))
                : "(none)");
      }
    }
    XLS_VLOG(4) << absl::StrFormat("Constraints (clock period: %dps):",
                                   clock_period_ps);
    for (Node* node : TopoSort(f)) {
      XLS_VLOG(4) << absl::StrFormat("  %s: [%s]", node->GetName(),
                                     absl::StrJoin(result.at(node), ", "));
    }
  }
  return result;
}

class ModelBuilder {
 public:
  ModelBuilder(FunctionBase* func, int64_t pipeline_length,
               int64_t clock_period_ps, const sched::ScheduleBounds& bounds,
               const DelayMap& delay_map, std::string_view model_name = "");

  absl::Status AddDefUseConstraints(Node* node, std::optional<Node*> user);
  absl::Status AddCausalConstraint(Node* node, std::optional<Node*> user);
  absl::Status AddLifetimeConstraint(Node* node, std::optional<Node*> user);
  absl::Status AddBackedgeConstraints(const BackedgeConstraint& constraint);
  absl::Status AddTimingConstraints();
  absl::Status AddSchedulingConstraint(const SchedulingConstraint& constraint);
  absl::Status AddIOConstraint(const IOConstraint& constraint);
  absl::Status AddNodeInCycleConstraint(
      const NodeInCycleConstraint& constraint);
  absl::Status AddDifferenceConstraint(const DifferenceConstraint& constraint);
  absl::Status AddRFSLConstraint(
      const RecvsFirstSendsLastConstraint& constraint);
  absl::Status AddSendThenRecvConstraint(
      const SendThenRecvConstraint& constraint);

  absl::Status SetObjective();

  absl::Status AddSlackVariables();

  math_opt::Model& Build() { return model_; }
  const math_opt::Model& Build() const { return model_; }

  absl::StatusOr<ScheduleCycleMap> ExtractResult(
      const math_opt::VariableMap<double>& variable_values) const;

  absl::Status ExtractError(
      const math_opt::VariableMap<double>& variable_values) const;

  absl::flat_hash_map<Node*, math_opt::Variable> GetCycleVars() const {
    return cycle_var_;
  }

  absl::flat_hash_map<Node*, math_opt::Variable> GetLifetimeVars() const {
    return lifetime_var_;
  }

 private:
  math_opt::LinearConstraint DiffAtMostConstraint(Node* x, Node* y,
                                                  int64_t limit,
                                                  std::string_view name) {
    return model_.AddLinearConstraint(
        cycle_var_.at(x) - cycle_var_.at(y) <= static_cast<double>(limit),
        absl::StrFormat("%s:%s-%s≤%d", name, x->GetName(), y->GetName(),
                        limit));
  }

  math_opt::LinearConstraint DiffLessThanConstraint(Node* x, Node* y,
                                                    int64_t limit,
                                                    std::string_view name) {
    return model_.AddLinearConstraint(
        cycle_var_.at(x) - cycle_var_.at(y) <= static_cast<double>(limit - 1),
        absl::StrFormat("%s:%s-%s<%d", name, x->GetName(), y->GetName(),
                        limit));
  }

  math_opt::LinearConstraint DiffAtLeastConstraint(Node* x, Node* y,
                                                   int64_t limit,
                                                   std::string_view name) {
    return model_.AddLinearConstraint(
        cycle_var_.at(x) - cycle_var_.at(y) >= static_cast<double>(limit),
        absl::StrFormat("%s:%s-%s≥%d", name, x->GetName(), y->GetName(),
                        limit));
  }

  math_opt::LinearConstraint DiffGreaterThanConstraint(Node* x, Node* y,
                                                       int64_t limit,
                                                       std::string_view name) {
    return model_.AddLinearConstraint(
        cycle_var_.at(x) - cycle_var_.at(y) >= static_cast<double>(limit + 1),
        absl::StrFormat("%s:%s-%s≥%d", name, x->GetName(), y->GetName(),
                        limit));
  }

  math_opt::LinearConstraint DiffEqualsConstraint(Node* x, Node* y,
                                                  int64_t diff,
                                                  std::string_view name) {
    if (x == y) {
      XLS_LOG(FATAL) << "DiffEqualsConstraint: " << x->GetName() << " - "
                     << y->GetName() << " = " << diff << " is unsatisfiable";
    }
    return model_.AddLinearConstraint(
        cycle_var_.at(x) - cycle_var_.at(y) == static_cast<double>(diff),
        absl::StrFormat("%s:%s-%s=%d", name, x->GetName(), y->GetName(), diff));
  }

  math_opt::Variable AddUpperBoundSlack(
      math_opt::LinearConstraint c,
      std::optional<math_opt::Variable> slack = std::nullopt) {
    XLS_CHECK_LT(c.upper_bound(), kInfinity)
        << "The constraint " << c.name() << " has no upper bound.";
    if (slack.has_value()) {
      XLS_CHECK_EQ(c.coefficient(*slack), 0.0)
          << "The slack variable " << slack->name()
          << " is already referenced in the constraint " << c.name() << ".";
    } else {
      slack = model_.AddVariable(0.0, kInfinity, /*is_integer=*/false,
                                 absl::StrCat(c.name(), "_ub_slack"));
    }
    model_.set_coefficient(c, *slack, -1.0);
    return *slack;
  }

  math_opt::Variable AddLowerBoundSlack(
      math_opt::LinearConstraint c,
      std::optional<math_opt::Variable> slack = std::nullopt) {
    XLS_CHECK_GT(c.lower_bound(), -kInfinity)
        << "The constraint " << c.name() << " has no lower bound.";
    if (slack.has_value()) {
      XLS_CHECK_EQ(c.coefficient(*slack), 0.0)
          << "The slack variable " << slack->name()
          << " is already referenced in the constraint " << c.name() << ".";
    } else {
      slack = model_.AddVariable(0.0, kInfinity, /*is_integer=*/false,
                                 absl::StrCat(c.name(), "_lb_slack"));
    }
    model_.set_coefficient(c, *slack, 1.0);
    return *slack;
  }

  std::pair<math_opt::Variable, math_opt::LinearConstraint> AddUpperBoundSlack(
      math_opt::Variable v,
      std::optional<math_opt::Variable> slack = std::nullopt) {
    XLS_CHECK_LT(v.upper_bound(), kInfinity)
        << "The variable " << v.name() << " has no fixed upper bound.";
    if (!slack.has_value()) {
      slack = model_.AddVariable(0.0, kInfinity, /*is_integer=*/false,
                                 absl::StrCat(v.name(), "_ub_slack"));
    }
    math_opt::LinearConstraint upper_bound = model_.AddLinearConstraint(
        v - *slack <= v.upper_bound(), absl::StrCat(v.name(), "_ub"));
    model_.set_upper_bound(v, kInfinity);
    return {*slack, upper_bound};
  }

  std::pair<math_opt::Variable, math_opt::LinearConstraint> AddLowerBoundSlack(
      math_opt::Variable v,
      std::optional<math_opt::Variable> slack = std::nullopt) {
    XLS_CHECK_GT(v.lower_bound(), -kInfinity)
        << "The variable " << v.name() << " has no fixed lower bound.";
    if (!slack.has_value()) {
      slack = model_.AddVariable(0.0, kInfinity, /*is_integer=*/false,
                                 absl::StrCat(v.name(), "_lb_slack"));
    }
    math_opt::LinearConstraint lower_bound = model_.AddLinearConstraint(
        v + *slack >= v.lower_bound(), absl::StrCat(v.name(), "_lb"));
    model_.set_lower_bound(v, -kInfinity);
    return {*slack, lower_bound};
  }

  FunctionBase* func_;
  math_opt::Model model_;
  int64_t pipeline_length_;
  int64_t clock_period_ps_;
  const DelayMap& delay_map_;

  // Node's cycle after scheduling
  absl::flat_hash_map<Node*, math_opt::Variable> cycle_var_;

  // Node's lifetime, from when it finishes executing until it is consumed by
  // the last user.
  absl::flat_hash_map<Node*, math_opt::Variable> lifetime_var_;

  // A placeholder node to represent an artificial sink node on the
  // data-dependence graph.
  math_opt::Variable cycle_at_sinknode_;

  // A cache of the delay constraints.
  absl::flat_hash_map<Node*, std::vector<Node*>> delay_constraints_;

  absl::flat_hash_map<std::pair<Node*, Node*>, math_opt::LinearConstraint>
      backedge_constraint_;
  absl::flat_hash_map<Node*, math_opt::LinearConstraint> send_last_constraint_;

  struct ConstraintPair {
    math_opt::LinearConstraint lower;
    math_opt::LinearConstraint upper;
  };
  absl::flat_hash_map<IOConstraint, std::vector<ConstraintPair>>
      io_constraints_;

  std::optional<math_opt::Variable> pipeline_length_slack_;
  std::optional<math_opt::Variable> backedge_slack_;

  struct SlackPair {
    math_opt::Variable min;
    math_opt::Variable max;
  };
  absl::flat_hash_map<IOConstraint, SlackPair> io_slack_;
};

ModelBuilder::ModelBuilder(FunctionBase* func, int64_t pipeline_length,
                           int64_t clock_period_ps,
                           const sched::ScheduleBounds& bounds,
                           const DelayMap& delay_map,
                           std::string_view model_name)
    : func_(func),
      model_(model_name),
      pipeline_length_(pipeline_length),
      clock_period_ps_(clock_period_ps),
      delay_map_(delay_map),
      cycle_at_sinknode_(model_.AddContinuousVariable(-kInfinity, kInfinity,
                                                      "cycle_at_sinknode")) {
  for (Node* node : func_->nodes()) {
    cycle_var_.emplace(
        node, model_.AddContinuousVariable(static_cast<double>(bounds.lb(node)),
                                           static_cast<double>(bounds.ub(node)),
                                           node->GetName()));
    lifetime_var_.emplace(
        node,
        model_.AddContinuousVariable(
            0.0, kInfinity, absl::StrFormat("lifetime_%s", node->GetName())));
  }
}

absl::Status ModelBuilder::AddDefUseConstraints(Node* node,
                                                std::optional<Node*> user) {
  XLS_RETURN_IF_ERROR(AddCausalConstraint(node, user));
  XLS_RETURN_IF_ERROR(AddLifetimeConstraint(node, user));
  return absl::OkStatus();
}

absl::Status ModelBuilder::AddCausalConstraint(Node* node,
                                               std::optional<Node*> user) {
  math_opt::Variable cycle_at_node = cycle_var_.at(node);
  math_opt::Variable cycle_at_user =
      user.has_value() ? cycle_var_.at(user.value()) : cycle_at_sinknode_;

  // Explicit delay nodes must lag their inputs by a certain number of cycles.
  int64_t min_delay = 0;
  if (user.has_value() && user.value()->Is<MinDelay>()) {
    min_delay = user.value()->As<MinDelay>()->delay();
  }

  std::string user_str = user.has_value() ? user.value()->GetName() : "«sink»";

  model_.AddLinearConstraint(
      cycle_at_user - cycle_at_node >= static_cast<double>(min_delay),
      absl::StrFormat("causal_%s_%s", node->GetName(), user_str));
  XLS_VLOG(2) << "Setting causal constraint: "
              << absl::StrFormat("cycle[%s] - cycle[%s] ≥ %d", user_str,
                                 node->GetName(), min_delay);

  return absl::OkStatus();
}

absl::Status ModelBuilder::AddLifetimeConstraint(Node* node,
                                                 std::optional<Node*> user) {
  math_opt::Variable cycle_at_node = cycle_var_.at(node);
  math_opt::Variable lifetime_at_node = lifetime_var_.at(node);
  math_opt::Variable cycle_at_user =
      user.has_value() ? cycle_var_.at(user.value()) : cycle_at_sinknode_;

  std::string user_str = user.has_value() ? user.value()->GetName() : "«sink»";

  model_.AddLinearConstraint(
      lifetime_at_node + cycle_at_node - cycle_at_user >= 0,
      absl::StrFormat("lifetime_%s_%s", node->GetName(), user_str));
  XLS_VLOG(2) << "Setting lifetime constraint: "
              << absl::StrFormat("lifetime[%s] + cycle[%s] - cycle[%s] ≥ 0",
                                 node->GetName(), node->GetName(), user_str);

  return absl::OkStatus();
}

// This ensures that state backedges don't span more than II cycles, which is
// necessary while enforcing a target II.
absl::Status ModelBuilder::AddBackedgeConstraints(
    const BackedgeConstraint& constraint) {
  Proc* proc = dynamic_cast<Proc*>(func_);
  if (proc == nullptr) {
    return absl::OkStatus();
  }
  const int64_t II = proc->GetInitiationInterval().value_or(1);

  using StateIndex = int64_t;
  for (StateIndex i = 0; i < proc->GetStateElementCount(); ++i) {
    Node* const state = proc->GetStateParam(i);
    Node* const next = proc->GetNextStateElement(i);
    if (next == state) {
      continue;
    }
    XLS_VLOG(2) << "Setting backedge constraint (II): "
                << absl::StrFormat("cycle[%s] - cycle[%s] < %d",
                                   next->GetName(), state->GetName(), II);
    backedge_constraint_.emplace(
        std::make_pair(state, next),
        DiffLessThanConstraint(next, state, II, "backedge"));
  }

  return absl::OkStatus();
}

absl::Status ModelBuilder::AddTimingConstraints() {
  if (delay_constraints_.empty()) {
    delay_constraints_ = ComputeCombinationalDelayConstraints(
        func_, clock_period_ps_, delay_map_);
  }

  for (Node* source : func_->nodes()) {
    for (Node* target : delay_constraints_.at(source)) {
      DiffAtLeastConstraint(target, source, 1, "timing");
      XLS_VLOG(2) << "Setting timing constraint: "
                  << absl::StrFormat("1 ≤ %s - %s", target->GetName(),
                                     source->GetName());
    }
  }

  return absl::OkStatus();
}

absl::Status ModelBuilder::AddSchedulingConstraint(
    const SchedulingConstraint& constraint) {
  if (std::holds_alternative<BackedgeConstraint>(constraint)) {
    return AddBackedgeConstraints(std::get<BackedgeConstraint>(constraint));
  }
  if (std::holds_alternative<IOConstraint>(constraint)) {
    return AddIOConstraint(std::get<IOConstraint>(constraint));
  }
  if (std::holds_alternative<NodeInCycleConstraint>(constraint)) {
    return AddNodeInCycleConstraint(
        std::get<NodeInCycleConstraint>(constraint));
  }
  if (std::holds_alternative<DifferenceConstraint>(constraint)) {
    return AddDifferenceConstraint(std::get<DifferenceConstraint>(constraint));
  }
  if (std::holds_alternative<RecvsFirstSendsLastConstraint>(constraint)) {
    return AddRFSLConstraint(
        std::get<RecvsFirstSendsLastConstraint>(constraint));
  }
  if (std::holds_alternative<SendThenRecvConstraint>(constraint)) {
    return AddSendThenRecvConstraint(
        std::get<SendThenRecvConstraint>(constraint));
  }
  return absl::InternalError("Unhandled scheduling constraint type");
}

absl::Status ModelBuilder::AddIOConstraint(const IOConstraint& constraint) {
  // Map from channel name to set of nodes that send/receive on that channel.
  absl::flat_hash_map<std::string, std::vector<Node*>> channel_to_nodes;
  for (Node* node : func_->nodes()) {
    if (node->Is<Receive>() || node->Is<Send>()) {
      XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
      channel_to_nodes[channel->name()].push_back(node);
    }
  }

  // We use `channel_to_nodes[...]` instead of `channel_to_nodes.at(...)`
  // below because we don't want to error out if a constraint is specified
  // that affects a channel with no associated send/receives in this proc.
  for (Node* source : channel_to_nodes[constraint.SourceChannel()]) {
    for (Node* target : channel_to_nodes[constraint.TargetChannel()]) {
      auto node_matches_direction = [](Node* node, IODirection dir) -> bool {
        return (node->Is<Send>() && dir == IODirection::kSend) ||
               (node->Is<Receive>() && dir == IODirection::kReceive);
      };
      if (!node_matches_direction(source, constraint.SourceDirection())) {
        continue;
      }
      if (!node_matches_direction(target, constraint.TargetDirection())) {
        continue;
      }
      if (source == target) {
        continue;
      }

      XLS_VLOG(2) << "Setting IO constraint: "
                  << absl::StrFormat("%d ≤ cycle[%s] - cycle[%s] ≤ %d",
                                     constraint.MinimumLatency(),
                                     target->GetName(), source->GetName(),
                                     constraint.MaximumLatency());
      io_constraints_[constraint].push_back({
          .lower = DiffAtLeastConstraint(target, source,
                                         constraint.MinimumLatency(), "io"),
          .upper = DiffAtMostConstraint(target, source,
                                        constraint.MaximumLatency(), "io"),
      });
    }
  }

  return absl::OkStatus();
}

absl::Status ModelBuilder::AddNodeInCycleConstraint(
    const NodeInCycleConstraint& constraint) {
  Node* node = constraint.GetNode();
  int64_t cycle = constraint.GetCycle();

  model_.AddLinearConstraint(cycle_var_.at(node) == static_cast<double>(cycle),
                             absl::StrFormat("nic_%s", node->GetName()));
  XLS_VLOG(2) << "Setting node-in-cycle constraint: "
              << absl::StrFormat("cycle[%s] = %d", node->GetName(), cycle);

  return absl::OkStatus();
}

absl::Status ModelBuilder::AddDifferenceConstraint(
    const DifferenceConstraint& constraint) {
  Node* a = constraint.GetA();
  Node* b = constraint.GetB();
  int64_t max_difference = constraint.GetMaxDifference();
  DiffAtMostConstraint(a, b, max_difference, "diff");

  XLS_VLOG(2) << "Setting difference constraint: "
              << absl::StrFormat("cycle[%s] - cycle[%s] ≤ %d", a->GetName(),
                                 b->GetName(), max_difference);

  return absl::OkStatus();
}

absl::Status ModelBuilder::AddRFSLConstraint(
    const RecvsFirstSendsLastConstraint& constraint) {
  for (Node* node : func_->nodes()) {
    if (node->Is<Receive>()) {
      XLS_VLOG(2) << "Setting receive-in-first-cycle constraint: "
                  << absl::StrFormat("cycle[%s] ≤ 0", node->GetName());
      model_.AddLinearConstraint(cycle_var_.at(node) <= 0,
                                 absl::StrFormat("recv_%s", node->GetName()));
    } else if (node->Is<Send>()) {
      XLS_VLOG(2) << "Setting send-in-last-cycle constraint: "
                  << absl::StrFormat("%d ≤ cycle[%s]", pipeline_length_ - 1,
                                     node->GetName());
      send_last_constraint_.emplace(
          node,
          model_.AddLinearConstraint(
              cycle_var_.at(node) >= static_cast<double>(pipeline_length_ - 1),
              absl::StrFormat("send_%s", node->GetName())));
    }
  }

  return absl::OkStatus();
}

absl::Status ModelBuilder::AddSendThenRecvConstraint(
    const SendThenRecvConstraint& constraint) {
  XLS_CHECK_GE(constraint.MinimumLatency(), 0);
  if (constraint.MinimumLatency() == 0) {
    return absl::OkStatus();
  }

  for (Node* recv : func_->nodes()) {
    if (!recv->Is<Receive>()) {
      continue;
    }

    // Look for a Send dependency.
    //
    // Technically, we probably don't need to trace back through the predicate
    // operand; the only operation we have today that takes a token and returns
    // data is a Receive (and technically tuple construction, but that just gets
    // weird), so we'd end up terminating our search before reaching a Send
    // anyway. But - just in case we ever add some other operation, we'll trace
    // up both paths to be sure.
    std::vector<Node*> stack(recv->operands().begin(), recv->operands().end());
    absl::flat_hash_set<Node*> seen;
    while (!stack.empty()) {
      Node* node = stack.back();
      stack.pop_back();
      if (seen.contains(node)) {
        continue;
      }
      seen.insert(node);

      if (node->Is<Send>()) {
        // Ensure that this send happens before the receive that depends on it.
        DiffAtLeastConstraint(recv, node, constraint.MinimumLatency(),
                              "send_then_recv");
        // We don't need to trace any further back on this line, since any
        // earlier sends are transitively handled.
        continue;
      }
      if (node->Is<Receive>()) {
        // No need to trace any further back on this line; this node will also
        // be the root of a search, and will get a similar appropriate delay.
        continue;
      }
      stack.insert(stack.end(), node->operands().begin(),
                   node->operands().end());
    }
  }
  return absl::OkStatus();
}

absl::Status ModelBuilder::SetObjective() {
  math_opt::LinearExpression objective;
  for (Node* node : func_->nodes()) {
    // Minimize node lifetimes.
    // The scaling makes the tie-breaker small in comparison, and is a power
    // of two so that there's no imprecision (just add to exponent).
    objective += 1024 *
                 static_cast<double>(node->GetType()->GetFlatBitCount()) *
                 lifetime_var_.at(node);
    // This acts as a tie-breaker for under-constrained problems, favoring ASAP
    // schedules.
    objective += cycle_var_.at(node);
  }
  model_.Minimize(objective);
  return absl::OkStatus();
}

absl::StatusOr<ScheduleCycleMap> ModelBuilder::ExtractResult(
    const math_opt::VariableMap<double>& variable_values) const {
  ScheduleCycleMap cycle_map;
  for (Node* node : func_->nodes()) {
    double cycle = variable_values.at(cycle_var_.at(node));
    if (std::fabs(cycle - std::round(cycle)) > 0.001) {
      return absl::InternalError(
          "The scheduling result is expected to be integer");
    }
    cycle_map[node] = std::round(cycle);
  }
  return cycle_map;
}

absl::Status ModelBuilder::AddSlackVariables() {
  // Add slack variables to all relevant constraints.

  // Drop our objective; we only want to find out the minimum slacks required
  // for feasibility.
  model_.Minimize(0.0);

  // First, add slack to our enforcement of pipeline length, which affects the
  // cycle upper bounds and send-last constraints. We assume users are most
  // willing to relax this; i.e., they care about throughput more than latency.
  pipeline_length_slack_ = model_.AddVariable(
      0.0, kInfinity, /*is_integer=*/false, "pipeline_length_slack");
  model_.AddToObjective(*pipeline_length_slack_);
  for (auto& [node, var] : cycle_var_) {
    AddUpperBoundSlack(var, pipeline_length_slack_);
  }
  for (auto& [node, constraint] : send_last_constraint_) {
    // Every send-last constraint is a lower bound specifying that this node's
    // cycle is at least (pipeline_length - 1); by subtracting the slack
    // variable from the linear expression, we ensure that the node's cycle is
    // at least (pipeline_length + slack - 1), keeping it in the last cycle.
    model_.set_coefficient(constraint, *pipeline_length_slack_, -1.0);
  }

  // Next, if this is a proc, relax the state backedge length restriction. We
  // assume users are reasonably willing to relax this; i.e., they care about
  // throughput, but they care more about the I/O constraints they've specified.
  if (Proc* proc = dynamic_cast<Proc*>(func_); proc != nullptr) {
    backedge_slack_ = model_.AddVariable(0.0, kInfinity, /*is_integer=*/false,
                                         "backedge_slack");
    model_.AddToObjective((1 << 10) * *backedge_slack_);
    for (auto& [nodes, constraint] : backedge_constraint_) {
      AddUpperBoundSlack(constraint, backedge_slack_);
    }
  }

  // Finally, relax the I/O constraints, if nothing else works.
  for (auto& [io_constraint, constraints] : io_constraints_) {
    math_opt::Variable min_slack = model_.AddVariable(
        0, kInfinity, /*is_integer=*/false,
        absl::StrCat("io_min_", io_constraint.SourceChannel(), "→",
                     io_constraint.TargetChannel(), "_slack"));
    math_opt::Variable max_slack = model_.AddVariable(
        0, kInfinity, /*is_integer=*/false,
        absl::StrCat("io_max_", io_constraint.SourceChannel(), "→",
                     io_constraint.TargetChannel(), "_slack"));
    model_.AddToObjective((1 << 20) * min_slack);
    model_.AddToObjective((1 << 20) * max_slack);
    io_slack_.emplace(io_constraint, SlackPair{
                                         .min = min_slack,
                                         .max = max_slack,
                                     });

    for (auto& [min_constraint, max_constraint] : constraints) {
      AddLowerBoundSlack(min_constraint, min_slack);
      AddUpperBoundSlack(max_constraint, max_slack);
    }
  }

  return absl::OkStatus();
}

absl::Status ModelBuilder::ExtractError(
    const math_opt::VariableMap<double>& variable_values) const {
  double pipeline_length_slack = variable_values.at(*pipeline_length_slack_);
  if (pipeline_length_slack > 0.001) {
    int64_t new_pipeline_length =
        pipeline_length_ +
        static_cast<int64_t>(std::ceil(pipeline_length_slack));
    return absl::InvalidArgumentError(absl::StrCat(
        "cannot achieve the specified pipeline length. Try `--pipeline_stages=",
        new_pipeline_length, "`"));
  }

  if (func_->IsProc()) {
    double backedge_slack = variable_values.at(*backedge_slack_);
    if (backedge_slack > 0.001) {
      int64_t new_backedge_length =
          func_->AsProcOrDie()->GetInitiationInterval().value_or(1) +
          static_cast<int64_t>(std::ceil(backedge_slack));
      return absl::InvalidArgumentError(absl::StrCat(
          "cannot achieve full throughput. Try `--worst_case_throughput=",
          static_cast<int64_t>(std::ceil(new_backedge_length)), "`"));
    }
  }

  std::vector<std::string> io_problems;
  for (auto& [io_constraint, slacks] : io_slack_) {
    double min_slack = variable_values.at(slacks.min);
    double max_slack = variable_values.at(slacks.max);

    std::vector<std::string> latency_suggestions;
    if (min_slack > 0.001) {
      int64_t new_min_latency = io_constraint.MinimumLatency() -
                                static_cast<int64_t>(std::ceil(min_slack));
      latency_suggestions.push_back(
          absl::StrCat("minimum latency ≤ ", new_min_latency));
    }
    if (max_slack > 0.001) {
      int64_t new_max_latency = io_constraint.MaximumLatency() +
                                static_cast<int64_t>(std::ceil(max_slack));
      latency_suggestions.push_back(
          absl::StrCat("maximum latency ≥ ", new_max_latency));
    }

    if (latency_suggestions.empty()) {
      continue;
    }
    io_problems.push_back(absl::StrCat(
        io_constraint.SourceChannel(), "→", io_constraint.TargetChannel(),
        " with ", absl::StrJoin(latency_suggestions, ", ")));
  }
  if (!io_problems.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "cannot satisfy the given I/O constraints. Would succeed with: ",
        absl::StrJoin(io_problems, ", ",
                      [](std::string* out, const std::string& entry) {
                        absl::StrAppend(out, "{", entry, "}");
                      })));
  }

  return absl::UnknownError("reason unknown.");
}

}  // namespace

absl::StatusOr<ScheduleCycleMap> SDCScheduler(
    FunctionBase* f, int64_t pipeline_stages, int64_t clock_period_ps,
    const DelayEstimator& delay_estimator, sched::ScheduleBounds* bounds,
    absl::Span<const SchedulingConstraint> constraints, bool check_feasibility,
    bool explain_infeasibility) {
  XLS_VLOG(3) << "SDCScheduler()";
  XLS_VLOG(3) << "  pipeline stages = " << pipeline_stages;
  XLS_VLOG_LINES(4, f->DumpIr());

  XLS_VLOG(4) << "Initial bounds:";
  XLS_VLOG_LINES(4, bounds->ToString());

  XLS_ASSIGN_OR_RETURN(DelayMap delay_map,
                       ComputeNodeDelays(f, delay_estimator));

  ModelBuilder model(f, pipeline_stages, clock_period_ps, *bounds, delay_map,
                     absl::StrCat("sdc_schedule_", f->name()));

  for (const SchedulingConstraint& constraint : constraints) {
    XLS_RETURN_IF_ERROR(model.AddSchedulingConstraint(constraint));
  }

  for (Node* node : f->nodes()) {
    for (Node* user : node->users()) {
      XLS_RETURN_IF_ERROR(model.AddDefUseConstraints(node, user));
    }
    if (f->IsFunction() && f->HasImplicitUse(node)) {
      XLS_RETURN_IF_ERROR(model.AddDefUseConstraints(node, std::nullopt));
    }
    if (f->IsProc()) {
      Proc* proc = f->AsProcOrDie();
      for (int64_t index : proc->GetNextStateIndices(node)) {
        // The next-state element always has lifetime extended to the state
        // param node, since we can't store the new value in the state register
        // until the old value's been used.
        XLS_RETURN_IF_ERROR(model.AddLifetimeConstraint(
            proc->GetNextStateElement(index), proc->GetStateParam(index)));
      }
    }
  }

  XLS_RETURN_IF_ERROR(model.AddTimingConstraints());

  if (!check_feasibility) {
    XLS_RETURN_IF_ERROR(model.SetObjective());
  }

  XLS_ASSIGN_OR_RETURN(
      math_opt::SolveResult result,
      math_opt::Solve(model.Build(), math_opt::SolverType::kGlop));

  if (explain_infeasibility &&
      (result.termination.reason == math_opt::TerminationReason::kInfeasible ||
       result.termination.reason ==
           math_opt::TerminationReason::kInfeasibleOrUnbounded)) {
    XLS_RETURN_IF_ERROR(model.AddSlackVariables());
    XLS_ASSIGN_OR_RETURN(
        math_opt::SolveResult result_with_slack,
        math_opt::Solve(model.Build(), math_opt::SolverType::kGlop));
    if (result_with_slack.termination.reason ==
            math_opt::TerminationReason::kOptimal ||
        result_with_slack.termination.reason ==
            math_opt::TerminationReason::kFeasible) {
      XLS_RETURN_IF_ERROR(
          model.ExtractError(result_with_slack.variable_values()));
    }
  }

  if (result.termination.reason != math_opt::TerminationReason::kOptimal) {
    // We don't know why the solver failed to find an optimal solution to our LP
    // problem; it could be an infeasibility issue (which needs more work to
    // analyze), a timeout, a precision error, or more. For now, just return a
    // simple error hinting at the problem.
    return absl::InternalError(
        absl::StrCat("The problem does not have an optimal solution; solver "
                     "terminated with ",
                     math_opt::EnumToString(result.termination.reason)));
  }

  return model.ExtractResult(result.variable_values());
}

}  // namespace xls
