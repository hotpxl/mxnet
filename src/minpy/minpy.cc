/*!
 * Copyright (c) 2017 by Contributors
 * \file minpy.cc
 * \brief MinPy.
 */
#include <mxnet/minpy.h>
#include <nnvm/graph.h>
#include <cassert>
#include <cstdio>
#include <map>
#include <vector>
#include "../c_api/c_api_ndarray.h"
#include "../executor/graph_executor.h"

namespace mxnet {
namespace minpy {

using nnvm::Node;
using nnvm::NodePtr;
using nnvm::NodeEntry;
using nnvm::NodeEntryMap;

namespace {

std::unordered_map<NDArray::Chunk*, std::size_t> AssignRelativeOrderToArrays(
    std::vector<ImperativeRuntime::ComputingRecord> const& sequence) {
  std::size_t id_counter = 0;
  std::unordered_map<NDArray::Chunk*, std::size_t> ret{};
  for (auto&& record : sequence) {
    for (auto&& input : record.inputs) {
      auto ptr = input.ptr_.get();
      auto it = ret.find(ptr);
      if (it == ret.end()) {
        ret.insert(std::make_pair(ptr, id_counter++));
      }
    }
    for (auto&& output : record.outputs) {
      auto ptr = output.ptr_.get();
      auto it = ret.find(ptr);
      if (it == rec.end()) {
        ret.insert(std::make_pair(ptr, id_counter++));
      }
    }
  }
  return ret;
}

// Call underlying functin in the old way.
void DoStrictEvaluation(ImperativeRuntime::ComputingRecord record) {
  std::printf("Strict evaluating \"%s\".\n", record.op->name.c_str());
  PushFCompute(record.delayed_function, record.op, record.attrs, record.ctx,
               record.read_vars, record.write_vars, record.requested,
               record.inputs, record.outputs);
}

Executor* BindSymbol(Symbol symbol, const NodeEntryMap<TShape>& shapes,
                     const NodeEntryMap<Context>& ctxs) {
  std::vector<NodePtr> input_nodes =
      symbol.ListInputs(Symbol::ListInputOption::kAll);

  size_t input_size = input_nodes.size();
  std::vector<NDArray> inputs;
  inputs.reserve(input_size);
  std::vector<NDArray> grads;
  grads.reserve(input_size);
  std::vector<OpReqType> grad_reqs;
  grad_reqs.reserve(input_size);

  // prepare inputs and set grad for every input
  for (size_t i = 0; i < input_size; ++i) {
    NodeEntry e = NodeEntry{input_nodes[i], 0, 0};
    if (shapes.count(e) && ctxs.count(e)) {
      TShape shape = shapes.at(e);
      Context ctx = ctxs.at(e);
      inputs.emplace_back(shape, ctx);
      NDArray grad(shape, ctx);
      grad = static_cast<real_t>(1.0);
      grads.emplace_back(grad);
      grad_reqs.emplace_back(OpReqType::kWriteTo);
    } else {
      LOG(FATAL) << "no corresponding ndarray: " << input_nodes[i]->attrs.name
                 << "(0)";
    }
  }

  // default context, assuming use the same context
  CHECK_GT(ctxs.size(), 0)
      << "The size of context mapping should be greater than zero";
  Context ctx = ctxs.begin()->second;

  std::map<std::string, Context> ctx_map;
  std::vector<NDArray> aux_states;

  return Executor::Bind(symbol, ctx, ctx_map, inputs, grads, grad_reqs,
                        aux_states);
}

nnvm::Symbol CompileToSymbol(
    std::vector<ImperativeRuntime::ComputingRecord>* computing_records) {
  using EntryState = bool;
  constexpr EntryState const kLeaf = true;
  constexpr EntryState const kInner = false;

  static int node_count = 0;
  NodeEntryMap<EntryState> entry_state;
  NodeEntryMap<NDArray> entry_ndarray;
  for (auto&& record : *computing_records) {
    std::vector<NDArray>& inputs = record.inputs;
    std::vector<NDArray>& outputs = record.outputs;

    NodePtr nn_node = Node::Create();
    nn_node->attrs = record.attrs;
    nn_node->attrs.name = "jit_node_" + std::to_string(node_count++);

    for (size_t i = 0; i < outputs.size(); ++i) {
      NodeEntry e{nn_node, static_cast<uint32_t>(i), 0};
      ndarray_entry_.insert({outputs[i].ptr_.get(), e});
      if (!entry_state.count(e)) {
        entry_state.emplace(e, kLeaf);
      }
      entry_ndarray.insert({e, outputs[i]});
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
      NodeEntry e;
      NDArray::Chunk* ptr = inputs[i].ptr_.get();
      if (ndarray_entry_.count(ptr)) {
        e = ndarray_entry_.at(ptr);
      } else {
        e.node = Node::Create();
      }
      nn_node->inputs.emplace_back(e);
      entry_state[e] = kInner;
      entry_ndarray.insert({e, inputs[i]});
    }
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

  NodeEntryMap<TShape> shapes;
  NodeEntryMap<Context> ctxs;
  for (const auto& kv : entry_ndarray) {
    shapes.insert({kv.first, kv.second.shape()});
    ctxs.insert({kv.first, kv.second.ctx()});
  }

  Executor* exec_ = BindSymbol(sym, shapes, ctxs);
  return sym;
}

}  // anonymous namespace

// The order of arrays, is the same as the original computing_records. namely
// {record[0].inputs, record[0].outputs, record[1].inputs, record[1].outputs,
// ...}
// For now, just compute output for leave nodes.
// TODO(yutian): Ignore memory allocation for now. I'm still thinking about
// the correct way to do it.
void ImperativeRuntime::RunCompiledSymbol(Executor* executor,
                                          std::vector<NDArray>* arrays) {
  exec::GraphExecutor* exec = static_cast<exec::GraphExecutor*>(executor);
  const nnvm::IndexedGraph& idx = exec->graph_.indexed_graph();

  for (const NDArray& arr : *arrays) {
    NDArray::Chunk* ptr = arr.ptr_.get();
    if (ndarray_entry_.count(ptr)) {
      NodeEntry e = ndarray_entry_.at(ptr);
      if (idx.exist(e.node.get())) {
        uint32_t entry_id = idx.entry_id(e);
        exec->data_entry_[entry_id] = arr;
      }
    }
  }
  exec->Forward(false);
  // return exec->outputs();
  // or copy it to outputs ndarray
}

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

bool ImperativeRuntime::JITGraph::Record::operator==(
    Record const& other) const {
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
