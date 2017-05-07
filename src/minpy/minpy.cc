/*!
 * Copyright (c) 2017 by Contributors
 * \file minpy.cc
 * \brief MinPy.
 */
#include <mxnet/minpy.h>
#include <nnvm/graph.h>
#include <cassert>
#include <cstdio>
#include <functional>
#include <map>
#include <vector>
#include "../c_api/c_api_ndarray.h"
#include "../executor/graph_executor.h"

namespace mxnet {
namespace minpy {

namespace {

// Empty NDArray is explicitly not handled here. They should not be used and
// will cause an immediate segmentation fault.
struct NDArrayHash {
  std::size_t operator()(NDArray const& a) const {
    return std::hash<Engine::VarHandle>()(a.var());
  }
};  // struct NDArrayHash

struct NDArrayEqual {
  bool operator()(NDArray const& a, NDArray const& b) const {
    return a.var() == b.var();
  }
};  // struct NDArrayEqual

std::unordered_map<NDArray, std::size_t, NDArrayHash, NDArrayEqual>
AssignRelativeOrderToArrays(
    std::vector<ImperativeRuntime::ComputingRecord> const& sequence) {
  std::size_t id_counter = 0;
  std::unordered_map<NDArray, std::size_t, NDArrayHash, NDArrayEqual> ret{};
  for (auto&& record : sequence) {
    for (auto&& input : record.inputs) {
      auto it = ret.find(input);
      if (it == ret.end()) {
        ret.insert(std::make_pair(input, id_counter++));
      }
    }
    for (auto&& output : record.outputs) {
      auto it = ret.find(output);
      if (it == ret.end()) {
        ret.insert(std::make_pair(output, id_counter++));
      }
    }
  }
  return ret;
}

// Call underlying functin in the old way.
void DoStrictEvaluation(ImperativeRuntime::ComputingRecord record) {
  // std::fprintf(stderr, "Strict evaluating \"%s\".\n", record.op->name.c_str());
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

  // Prepare inputs and set grad for every input.
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
      LOG(FATAL) << "No corresponding NDArray: " << input_nodes[i]->attrs.name
                 << "(0).";
    }
  }

  // default context, assuming use the same context
  CHECK_GT(ctxs.size(), 0)
      << "The size of context mapping should be greater than zero.";
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
      std::size_t id = array_to_id.at(input);
      array_shapes_[id] = input.shape();
      inputs.push_back(id);
    }
    for (auto&& output : record.outputs) {
      std::size_t id = array_to_id.at(output);
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
  if (jit_enabled_) {
    FlushJITSequence();
  }
}

void ImperativeRuntime::MarkAsOutput(NDArray const& array) {
  assert(jit_enabled_);
  extra_outputs_.push_back(array);
}

void ImperativeRuntime::SetContext(int dev_type, int dev_id) {
  assert(jit_enabled_);
  default_context_ = std::make_shared<Context>(
      Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id));
}

void ImperativeRuntime::Invoke(ComputingRecord record) {
  PushJITRecord(record);
}

void ImperativeRuntime::PushJITRecord(ComputingRecord record) {
  if (jit_enabled_) {
    // Save for lazy evaluation.
    // std::fprintf(stderr, "Save \"%s\" for lazy evaluation.\n",
    //              record.op->name.c_str());
    jit_sequence_.emplace_back(std::move(record));
  } else {
    DoStrictEvaluation(std::move(record));
  }
}

void ImperativeRuntime::FlushJITSequence() {
  if (jit_sequence_.empty()) {
    extra_outputs_.clear();
    default_context_.reset();
    return;
  }
  auto new_graph = std::make_shared<JITGraph>(jit_sequence_);
  std::shared_ptr<CompiledSymbol> compiled_symbol;
  for (auto&& graph : jit_graphs_) {
    if (*graph.first == *new_graph) {
      compiled_symbol = graph.second;
      break;
    }
  }
  // std::fprintf(stderr, "Compare graph result: %d.\n",
  //              static_cast<bool>(compiled_symbol));
  if (static_cast<bool>(compiled_symbol)) {
    RunCompiledSymbol(compiled_symbol, &jit_sequence_);
  } else {
    auto compiled_symbol = std::make_shared<CompiledSymbol>(
        CompileToSymbol(&jit_sequence_, extra_outputs_, default_context_));
    jit_graphs_.emplace(new_graph, compiled_symbol);
    RunCompiledSymbol(compiled_symbol, &jit_sequence_);
  }
  extra_outputs_.clear();
  jit_sequence_.clear();
  default_context_.reset();
}

ImperativeRuntime::CompiledSymbol ImperativeRuntime::CompileToSymbol(
    std::vector<ImperativeRuntime::ComputingRecord>* jit_sequence,
    std::vector<NDArray> const& extra_outputs,
    std::shared_ptr<Context> default_context) {
  auto array_to_id = AssignRelativeOrderToArrays(*jit_sequence);

  std::unordered_map<std::size_t, nnvm::NodeEntry> array_id_to_node;
  static int node_count = 0;
  std::set<std::size_t> input_array_ids;
  std::set<std::size_t> output_array_ids;
  nnvm::NodeEntryMap<NDArray> node_to_array;
  for (auto&& record : *jit_sequence) {
    auto&& inputs = record.inputs;
    auto&& outputs = record.outputs;

    nnvm::NodePtr nn_node = nnvm::Node::Create();
    nn_node->attrs = record.attrs;
    nn_node->attrs.name = "jit_node_" + std::to_string(node_count++);

    for (std::size_t i = 0; i < inputs.size(); ++i) {
      nnvm::NodeEntry e;
      auto id = array_to_id.at(inputs[i]);
      auto it = array_id_to_node.find(id);
      if (it == array_id_to_node.end()) {
        e.node = nnvm::Node::Create();
        e.index = 0;
        e.version = 0;
        input_array_ids.emplace(id);
        node_to_array.emplace(e, inputs[i]);
        array_id_to_node.emplace(id, e);
      } else {
        e = it->second;
      }
      nn_node->inputs.emplace_back(e);
      output_array_ids.erase(id);
    }

    for (std::size_t i = 0; i < outputs.size(); ++i) {
      nnvm::NodeEntry e{nn_node, static_cast<std::uint32_t>(i), 0};
      auto id = array_to_id.at(outputs[i]);
      assert(array_id_to_node.count(id) == 0);
      array_id_to_node.emplace(id, e);
      output_array_ids.emplace(id);
      node_to_array.emplace(e, outputs[i]);
    }
  }
  for (auto&& array : extra_outputs) {
    auto id = array_to_id.at(array);
    output_array_ids.emplace(id);
  }
  std::vector<nnvm::NodeEntry> graph_outputs;
  for (auto&& id : output_array_ids) {
    graph_outputs.emplace_back(array_id_to_node.at(id));
  }

  nnvm::Symbol symbol;
  symbol.outputs = graph_outputs;
  // TODO(yutian): Debug.
  // symbol.Print(std::cout);

  nnvm::NodeEntryMap<TShape> shapes;
  nnvm::NodeEntryMap<Context> ctxs;
  for (auto&& kv : node_to_array) {
    shapes.emplace(kv.first, kv.second.shape());
    ctxs.emplace(kv.first,
                 default_context ? *default_context : kv.second.ctx());
  }

  Executor* exec = BindSymbol(symbol, shapes, ctxs);
  return {symbol, exec, std::move(array_id_to_node), std::move(input_array_ids),
          std::move(output_array_ids)};
}

void ImperativeRuntime::RunCompiledSymbol(
    std::shared_ptr<CompiledSymbol> compiled_symbol,
    std::vector<ComputingRecord>* jit_sequence) {
  exec::GraphExecutor* exec =
      static_cast<exec::GraphExecutor*>(compiled_symbol->executor);
  nnvm::IndexedGraph const& idx = exec->graph_.indexed_graph();
  auto array_to_id = AssignRelativeOrderToArrays(*jit_sequence);

  for (auto&& p : array_to_id) {
    auto id = p.second;
    auto&& array = p.first;
    if (compiled_symbol->input_array_ids.count(id) != 0) {
      auto&& node = compiled_symbol->array_id_to_node.at(id);
      CopyFromTo(array, &exec->data_entry_.at(idx.entry_id(node)));
    }
  }

  exec->Forward(false);

  for (auto&& p : array_to_id) {
    auto id = p.second;
    auto array = p.first;
    if (compiled_symbol->output_array_ids.count(id) != 0) {
      array.CheckAndAlloc();
      auto&& node = compiled_symbol->array_id_to_node.at(id);
      auto&& arr = exec->data_entry_.at(idx.entry_id(node));
      CopyFromTo(arr, &array);
      // TODO(yutian): optional?
      array.WaitToRead();
    }
  }
}

}  // namespace minpy
}  // namespace mxnet
