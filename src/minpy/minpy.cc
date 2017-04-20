/*!
 * Copyright (c) 2017 by Contributors
 * \file minpy.cc
 * \brief MinPy.
 */
#include <mxnet/minpy.h>
#include <cassert>
#include <cstdio>
#include "../c_api/c_api_ndarray.h"

namespace mxnet {
namespace minpy {

namespace {

// Call underlying functin in the old way.
void DoStrictEvaluation(ImperativeRuntime::ComputingRecord record) {
  std::printf("Strict evaluating \"%s\".\n", record.op->name.c_str());
  PushFCompute(record.delayed_function, record.op, record.attrs, record.ctx,
               record.read_vars, record.write_vars, record.requested,
               record.ndinputs, record.ndoutputs);
}

}  // anonymous namespace

ImperativeRuntime* ImperativeRuntime::Get() {
  static ImperativeRuntime r{};
  return &r;
}

void ImperativeRuntime::EnableJIT() {
  assert(!jit_enabled_);
  jit_enabled_ = true;
}

void ImperativeRuntime::DisableJIT() {
  assert(jit_enabled_);
  FlushJITSequence();
  jit_enabled_ = false;
}

// void ImperativeRuntime::EnableAutograd() {
//   assert(!autograd_enabled_);
//   autograd_enabled_ = true;
// }

// void ImperativeRuntime::DisableAutograd() {
//   assert(autograd_enabled_);
//   FlushGradSequence();
//   autograd_enabled_ = false;
// }

void ImperativeRuntime::StrictEvaluate() {
  if (jit_enabled_) {
    FlushJITSequence();
  }
}

void ImperativeRuntime::Invoke(ComputingRecord record) {
  // PushAutogradRecord(record);
  PushJITRecord(record);
}

// void ImperativeRuntime::PushAutogradRecord(ComputingRecord record) {
//   if (autograd_enabled_) {
//     autograd_seqeunce_.emplace_back(std::move(record));
//   }
// }

void ImperativeRuntime::PushJITRecord(ComputingRecord record) {
  if (jit_enabled_) {
    // Save for lazy evaluation.
    std::printf("Save \"%s\" for lazy evaluation.\n", record.op->name.c_str());
    jit_sequence_.emplace_back(std::move(record));
  } else {
    DoStrictEvaluation(std::move(record));
  }
}

void ImperativeRuntime::FlushJITSequence() {
  JitGraph *new_graph = new JitGraph(jit_sequence_);
  bool graph_matched = false;
  for (auto&& graph : jit_graphs_)
    if (graph->EqualGraph(*new_graph)) {
      graph_matched = true;
      break;
    }
  if (!graph_matched) {
    std::printf("Compare Graph Result: Not match with any prev graph\n");
    jit_graphs_.emplace_back(new_graph);
  } else
    std::printf("Compare Graph Result: Match a prev graph :-)\n");

  typedef bool EntryState;
  const EntryState kLeaf  = true;
  const EntryState kInner = false;
  using nnvm::Node;
  using nnvm::NodePtr;
  using nnvm::NodeEntry;

  static int node_count = 0;
  nnvm::NodeEntryMap<EntryState> entry_state;
  for (auto&& record : jit_sequence_) {
    std::vector<NDArray>& ndinputs  = record.ndinputs;
    std::vector<NDArray>& ndoutputs = record.ndoutputs;

    NodePtr nn_node = Node::Create();
    nn_node->attrs = record.attrs;
    nn_node->attrs.name = "agnode_" + std::to_string(node_count++);

    for (size_t i = 0; i < ndoutputs.size(); ++i) {
      NodeEntry& e = ndoutputs[i].entry_;
      e = NodeEntry{nn_node, static_cast<uint32_t>(i), 0};

      if (!entry_state.count(e)) {
        entry_state.emplace(e, kLeaf);
      }
    }

    for (size_t i = 0; i < ndinputs.size(); ++i) {
      NodeEntry& e = ndinputs[i].entry_;
      if (e.node.get() == nullptr) {
        e.node = Node::Create();
      }
      nn_node->inputs.emplace_back(e);
      entry_state[e] = kInner;
    }

    DoStrictEvaluation(std::move(record));
  }

  std::vector<NodeEntry> graph_outputs;
  for (auto& kv : entry_state) {
    if (kv.second == kLeaf) {
      graph_outputs.emplace_back(kv.first);
    }
  }

  nnvm::Symbol sym;
  sym.outputs = graph_outputs;
  sym.Print(std::cout);

  jit_sequence_.clear();
}

void ImperativeRuntime::JitGraph::BuildGraph() {
  static const size_t UNMATCHID = 0;
  size_t output_cnt = UNMATCHID + 1;

  for (auto&& record : jit_sequence_) {

    for (auto&& input : record.ndinputs) {
      auto it = ndoutputs_map_.find(reinterpret_cast<size_t>(input.ptr_.get()));
      ndinputs_id_.emplace_back((it != ndoutputs_map_.end())? it->second : UNMATCHID);
    }

    for (auto&& output : record.ndoutputs) {
      ndoutputs_map_.insert(std::pair<size_t, size_t>(
            reinterpret_cast<size_t>(output.ptr_.get()), output_cnt));
      ++output_cnt;
    }
  }
}

bool ImperativeRuntime::JitGraph::EqualGraph(const JitGraph& rhs)const {
  std::printf("Compare Graph: Start Graph Compare\n");
  auto &lhs = *this;

  // Compare JIT sequence length
  if (lhs.jit_sequence_.size() != rhs.jit_sequence_.size()) {
    std::printf("Reason of Failure: Jit Length Not Equal\n");
    return false;
  }

  size_t input_cnt = 0;
  // Compare Each ComputingRecord
  for (size_t i = 0; i < jit_sequence_.size(); ++i) {
    // Check Op Name
    if (lhs.jit_sequence_[i].op->name.compare(rhs.jit_sequence_[i].op->name) != 0) {
      std::printf("Reason of Failure: Different Op Name\n");
      return false;
    }

    auto &lhs_ndinputs = lhs.jit_sequence_[i].ndinputs;
    auto &lhs_ndoutputs = lhs.jit_sequence_[i].ndoutputs;
    auto &rhs_ndinputs = rhs.jit_sequence_[i].ndinputs;
    auto &rhs_ndoutputs = rhs.jit_sequence_[i].ndoutputs;

    // Compare Inputs & Outputs Size
    if (lhs_ndinputs.size() != rhs_ndinputs.size()
        || lhs_ndoutputs.size() != rhs_ndoutputs.size()) {
      std::printf("Reason of Failure: Diff input or output size \n");
      return false;
    }

    // Compare Inputs shape & ID
    for (size_t j = 0; j < lhs_ndinputs.size(); ++j) {
      if (lhs_ndinputs[j].shape() != rhs_ndinputs[j].shape()) {
        std::printf("Reason of Failure: Mismatch input shape\n");
        return false;
      }
      if (lhs.ndinputs_id_[input_cnt] != rhs.ndinputs_id_[input_cnt]) {
        std::printf("Reason of Failure: Diff input ID\n");
        return false;
      }
      ++input_cnt;
    }

    // Compare Outputs shape
    for (size_t j = 0; j < lhs_ndoutputs.size(); ++j)
      if (lhs_ndoutputs[j].shape() != rhs_ndoutputs[j].shape()) {
        std::printf("Reason of Failure: Mismatch Output shape\n");
        return false;
      }
  }

  return true;
}

// void ImperativeRuntime:: ::FlushAutogradSequence() {
//   for (auto&& i : autograd_component_.Process(std::move(autograd_sequence_)))
//   {
//     PushJITSequence(std::move(i));
//   }
//   autograd_sequence_.clear();
// }

}  // namespace minpy
}  // namespace mxnet
