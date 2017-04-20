/*!
 * Copyright (c) 2017 by Contributors
 * \file minpy.cc
 * \brief MinPy.
 */
#include <mxnet/minpy.h>
#include <cassert>
#include <cstdio>
#include <map>
#include <vector>
#include "../c_api/c_api_ndarray.h"

namespace mxnet {
namespace minpy {

namespace {

// Call underlying functin in the old way.
void DoStrictEvaluation(ImperativeRuntime::ComputingRecord record) {
  std::printf("Strict evaluating \"%s\".\n", record.op->name.c_str());
  PushFCompute(record.delayed_function, record.op, record.attrs, record.ctx,
               record.read_vars, record.write_vars, record.requested,
               record.inputs, record.outputs);
}

nnvm::Symbol CompileToSymbol(
    std::vector<ImperativeRuntime::ComputingRecord> const* computing_records) {
  // TODO(ziheng)
}

void RunCompiledSymbol(nnvm::Symbol symbol, std::vector<NDArray>* arrays) {
  // TODO(ziheng)
  // The order of arrays, is the same as the original computing_records. namely
  // {record[0].inputs, record[0].outputs, record[1].inputs, record[1].outputs,
  // ...}
  // For now, just compute output for leave nodes.
  // TODO(yutian): Ignore memory allocation for now. I'm still thinking about
  // the correct way to do it.
}

}  // anonymous namespace

class ImperativeRuntime::JITGraph final {
 public:
  JITGraph(std::vector<ImperativeRuntime::ComputingRecord> const& jit_sequence);
  bool operator==(JITGraph const& other) const;

 private:
  std::unordered_map<std::size_t, TShape> array_shapes_{};
  struct Record {
    nnvm::NodeAttrs attrs;
    std::vector<std::size_t> inputs;
    std::vector<std::size_t> outputs;

    bool operator==(Record const& other) const;
  };  // struct Record
  std::vector<Record> records_{};
};  // class ImperativeRuntime::JITGraph

bool ImperativeRuntime::JITGraph::Record::operator==(Record const& other) const {
  return attrs.op == other.attrs.op && attrs.scalars == other.attrs.scalars &&
         attrs.dict == other.attrs.dict && inputs == other.inputs &&
         outputs == other.outputs;
}

ImperativeRuntime::JITGraph::JITGraph(
    std::vector<ImperativeRuntime::ComputingRecord> const& jit_sequence) {
  std::size_t id_counter = 0;
  std::unordered_map<NDArray::Chunk*, std::size_t> array_to_id{};

  for (auto&& record : jit_sequence) {
    std::vector<std::size_t> inputs;
    std::vector<std::size_t> outputs;
    for (auto&& input : record.inputs) {
      std::size_t id;
      auto ptr = input.ptr_.get();
      auto it = array_to_id.find(ptr);
      if (it == array_to_id.end()) {
        id = id_counter++;
        array_to_id.insert(std::make_pair(ptr, id));
        array_shapes_.insert(std::make_pair(id, input.shape()));
      } else {
        id = it->second;
      }
      inputs.push_back(id);
    }
    for (auto&& output : record.outputs) {
      std::size_t id;
      auto ptr = output.ptr_.get();
      auto it = array_to_id.find(ptr);
      if (it == array_to_id.end()) {
        id = id_counter++;
        array_to_id.insert(std::make_pair(ptr, id));
        array_shapes_.insert(std::make_pair(id, output.shape()));
      } else {
        id = it->second;
      }
      outputs.push_back(id);
    }
    records_.push_back({record.attrs, std::move(inputs), std::move(outputs)});
  }
}

bool ImperativeRuntime::JITGraph::operator==(JITGraph const& rhs) const {
  return array_shapes_ == rhs.array_shapes_ && records_ == rhs.records_;
}

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
  JITGraph* new_graph = new JITGraph(jit_sequence_);
  bool graph_matched = false;
  for (auto&& graph : jit_graphs_)
    if (*graph == *new_graph) {
      graph_matched = true;
      break;
    }
  if (!graph_matched) {
    std::printf("Compare Graph Result: Not match with any prev graph\n");
    jit_graphs_.emplace_back(new_graph);
  } else
    std::printf("Compare Graph Result: Match a prev graph :-)\n");

  typedef bool EntryState;
  const EntryState kLeaf = true;
  const EntryState kInner = false;
  using nnvm::Node;
  using nnvm::NodePtr;
  using nnvm::NodeEntry;

  static int node_count = 0;
  nnvm::NodeEntryMap<EntryState> entry_state;
  std::unordered_map<NDArray::Chunk*, NodeEntry> array_to_entry{};
  for (auto&& record : jit_sequence_) {
    std::vector<NDArray>& inputs = record.inputs;
    std::vector<NDArray>& outputs = record.outputs;

    NodePtr nn_node = Node::Create();
    nn_node->attrs = record.attrs;
    nn_node->attrs.name = "agnode_" + std::to_string(node_count++);

    for (size_t i = 0; i < outputs.size(); ++i) {
      NodeEntry e{nn_node, static_cast<uint32_t>(i), 0};
      array_to_entry.insert({outputs[i].ptr_.get(), e});
      if (!entry_state.count(e)) {
        entry_state.emplace(e, kLeaf);
      }
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
      NodeEntry e;
      auto ptr = inputs[i].ptr_.get();
      if (array_to_entry.count(ptr)) {
        e = array_to_entry.at(ptr);
      } else {
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
