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

namespace {

std::unordered_map<Engine::VarHandle, std::size_t> AssignRelativeOrderToArrays(
    std::vector<ImperativeRuntime::ComputingRecord> const& sequence) {
  std::size_t id_counter = 0;
  std::unordered_map<Engine::VarHandle, std::size_t> ret{};
  for (auto&& record : sequence) {
    for (auto&& input : record.inputs) {
      auto ptr = input.var();
      auto it = ret.find(ptr);
      if (it == ret.end()) {
        ret.insert(std::make_pair(ptr, id_counter++));
      }
    }
    for (auto&& output : record.outputs) {
      auto ptr = output.var();
      auto it = ret.find(ptr);
      if (it == ret.end()) {
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

Executor* BindSymbol(Symbol symbol, nnvm::NodeEntryMap<TShape> const& shapes,
                     nnvm::NodeEntryMap<Context> const& ctxs) {
  std::vector<nnvm::NodePtr> input_nodes =
      symbol.ListInputs(Symbol::ListInputOption::kAll);

  std::size_t input_size = input_nodes.size();
  std::vector<NDArray> inputs;
  inputs.reserve(input_size);
  std::vector<NDArray> grads;
  grads.reserve(input_size);
  std::vector<OpReqType> grad_reqs;
  grad_reqs.reserve(input_size);

  // prepare inputs and set grad for every input
  for (std::size_t i = 0; i < input_size; ++i) {
    nnvm::NodeEntry e = nnvm::NodeEntry{input_nodes[i], 0, 0};
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

bool ImperativeRuntime::JITGraph::Record::operator==(
    Record const& other) const {
  return attrs.op == other.attrs.op && attrs.scalars == other.attrs.scalars &&
         attrs.dict == other.attrs.dict && inputs == other.inputs &&
         outputs == other.outputs;
}

ImperativeRuntime::JITGraph::JITGraph(
    std::vector<ImperativeRuntime::ComputingRecord> const& jit_sequence) {
  auto array_to_id = AssignRelativeOrderToArrays(jit_sequence);

  for (auto&& record : jit_sequence) {
    std::vector<std::size_t> inputs;
    std::vector<std::size_t> outputs;
    for (auto&& input : record.inputs) {
      std::size_t id = array_to_id.at(input.var());
      array_shapes_[id] = input.shape();
      inputs.push_back(id);
    }
    for (auto&& output : record.outputs) {
      std::size_t id = array_to_id.at(output.var());
      array_shapes_[id] = output.shape();
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

void ImperativeRuntime::StrictEvaluate() {
  printf("strict evaluate %d\n", jit_enabled_);
  if (jit_enabled_) {
    FlushJITSequence();
  }
}

void ImperativeRuntime::Invoke(ComputingRecord record) {
  PushJITRecord(record);
}

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
  auto new_graph = std::make_shared<JITGraph>(jit_sequence_);
  std::shared_ptr<CompiledSymbol> compiled_symbol;
  for (auto&& graph : jit_graphs_) {
    if (*graph.first == *new_graph) {
      compiled_symbol = graph.second;
      break;
    }
  }
  std::printf("Compare graph result: %d.\n",
              static_cast<bool>(compiled_symbol));
  if (static_cast<bool>(compiled_symbol)) {
    RunCompiledSymbol(compiled_symbol.get(), &jit_sequence_);
  } else {
    auto compiled_symbol =
        std::make_shared<CompiledSymbol>(CompileToSymbol(&jit_sequence_));
    jit_graphs_.emplace(new_graph, compiled_symbol);
    RunCompiledSymbol(compiled_symbol.get(), &jit_sequence_);
  }
  jit_sequence_.clear();
}

ImperativeRuntime::CompiledSymbol ImperativeRuntime::CompileToSymbol(
    std::vector<ImperativeRuntime::ComputingRecord>* jit_sequence) {
  using EntryState = bool;
  constexpr EntryState const kLeaf = true;
  constexpr EntryState const kInner = false;
  auto array_to_id = AssignRelativeOrderToArrays(*jit_sequence);
  std::unordered_map<std::size_t, nnvm::NodeEntry> array_id_to_node{};

  static int node_count = 0;
  nnvm::NodeEntryMap<EntryState> entry_state;
  nnvm::NodeEntryMap<NDArray> entry_array;
  for (auto&& record : *jit_sequence) {
    std::vector<NDArray>& inputs = record.inputs;
    std::vector<NDArray>& outputs = record.outputs;

    nnvm::NodePtr nn_node = nnvm::Node::Create();
    nn_node->attrs = record.attrs;
    nn_node->attrs.name = "jit_node_" + std::to_string(node_count++);

    for (size_t i = 0; i < outputs.size(); ++i) {
      nnvm::NodeEntry e{nn_node, static_cast<std::uint32_t>(i), 0};
      array_id_to_node[array_to_id[outputs[i].var()]] = e;
      if (!entry_state.count(e)) {
        entry_state.insert({e, kLeaf});
      }
      entry_array.insert({e, outputs[i]});
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
      nnvm::NodeEntry e;
      auto it = array_id_to_node.find(array_to_id[inputs[i].var()]);
      if (it == array_id_to_node.end()) {
        e.node = nnvm::Node::Create();
      } else {
        e = it->second;
      }
      nn_node->inputs.push_back(e);
      entry_state[e] = kInner;
      entry_array.insert({e, inputs[i]});
    }
  }

  std::vector<nnvm::NodeEntry> graph_outputs;
  for (auto& kv : entry_state) {
    if (kv.second == kLeaf) {
      graph_outputs.push_back(kv.first);
    }
  }

  nnvm::Symbol symbol;
  symbol.outputs = graph_outputs;
  // TODO(yutian): Debug.
  symbol.Print(std::cout);

  nnvm::NodeEntryMap<TShape> shapes;
  nnvm::NodeEntryMap<Context> ctxs;
  for (auto const& kv : entry_array) {
    shapes.insert({kv.first, kv.second.shape()});
    ctxs.insert({kv.first, kv.second.ctx()});
  }

  Executor* exec = BindSymbol(symbol, shapes, ctxs);
  return {symbol, exec, std::move(array_id_to_node)};
}

void ImperativeRuntime::RunCompiledSymbol(
    CompiledSymbol* compiled_symbol,
    std::vector<ComputingRecord>* jit_sequence) {
  exec::GraphExecutor* exec =
      static_cast<exec::GraphExecutor*>(compiled_symbol->executor);
  nnvm::IndexedGraph const& idx = exec->graph_.indexed_graph();
  auto array_id_to_node = AssignRelativeOrderToArrays(*jit_sequence);

  for (auto&& record : *jit_sequence) {
    for (auto&& input : record.inputs) {
      auto id = array_id_to_node[input.var()];
      auto it = compiled_symbol->array_id_to_node.find(id);
      if (it != compiled_symbol->array_id_to_node.end()) {
        auto entry = it->second;
        if (idx.exist(entry.node.get())) {
          auto entry_id = idx.entry_id(entry);
          exec->data_entry_[entry_id] = input;
        }
      }
    }
    for (auto&& output : record.outputs) {
      auto id = array_id_to_node[output.var()];
      auto it = compiled_symbol->array_id_to_node.find(id);
      if (it != compiled_symbol->array_id_to_node.end()) {
        auto entry = it->second;
        if (idx.exist(entry.node.get())) {
          auto entry_id = idx.entry_id(entry);
          exec->data_entry_[entry_id] = output;
        }
      }
    }

    // TODO(yutian)
    for (auto&& input : record.inputs) {
      input.CheckAndAlloc();
    }
    for (auto&& output : record.outputs) {
      output.CheckAndAlloc();
    }
  }
  std::printf("running symbol\n");
  exec->Forward(false);
  std::printf("running symbol complete\n");
}

}  // namespace minpy
}  // namespace mxnet
