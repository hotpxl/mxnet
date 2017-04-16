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
      e = NodeEntry{nn_node, i, 0};

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

// void ImperativeRuntime:: ::FlushAutogradSequence() {
//   for (auto&& i : autograd_component_.Process(std::move(autograd_sequence_)))
//   {
//     PushJITSequence(std::move(i));
//   }
//   autograd_sequence_.clear();
// }

}  // namespace minpy
}  // namespace mxnet
