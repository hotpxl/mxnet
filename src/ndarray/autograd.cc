/*!
 *  Copyright (c) 2017 by Contributors
 * \file autograd.cc
 * \brief Implementation of AutogradRuntime module.
 */

#include <mxnet/operator.h>
#include <mxnet/executor.h>
#include <nnvm/pass_functions.h>
#include <unordered_set>
#include <iostream>
#include "../executor/graph_executor.h"
#include "./autograd.h"

using namespace std;

namespace mxnet {
namespace autograd {

using nnvm::Symbol;
using nnvm::Node;
using nnvm::NodePtr;
using nnvm::NodeEntry;
using nnvm::NodeEntryMap;
using exec::GraphExecutor;

#if DMLC_CXX11_THREAD_LOCAL
thread_local bool AutogradRuntime::is_train_;
#else
MX_TREAD_LOCAL bool AutogradRuntime::is_train_;
#endif

template<typename FVisit>
inline void AGDFSVisit(const std::vector<AGNodeEntry>& heads,
                       FVisit fvisit) {
  typedef const AGNodePtr* GNode;
  std::vector<GNode> head_nodes(heads.size());
  std::transform(heads.begin(), heads.end(), head_nodes.begin(),
                 [](const AGNodeEntry& e)->GNode {
                   return &e.ag_node;
                 });
  nnvm::PostOrderDFSVisit<GNode, AGNode*>(
      head_nodes,
      [fvisit](GNode n) { fvisit(*n); },  // FVisit
      [](GNode n)->AGNode* { return n->get(); },  // HashFunc
      [](GNode n)->uint32_t { return (*n)->inputs.size(); },
      [](GNode n, uint32_t index)->GNode { return &(*n)->inputs.at(index).ag_node; });
}

nnvm::NodeEntry AGNodeEntry::nn_entry() const {
  return nnvm::NodeEntry{ag_node->nn_node, index, version};
}

AutogradRuntime::AutogradRuntime() {}

void AutogradRuntime::MarkVariables(
    const std::vector<NDArray*>& variables,
    const std::vector<mx_uint>& grad_reqs,
    const std::vector<NDArray*>& gradients) {
  for (uint32_t i = 0; i < variables.size(); ++i) {
    AGNodeEntry e{AGNode::Create(Node::Create()), 0, 0};
    variables[i]->entry_.clear();
    e.ag_node->outputs.push_back(*variables[i]);
    gradients[i]->entry_.clear();
    e.ag_node->out_grads.push_back(*gradients[i]);
    e.ag_node->grad_req = static_cast<OpReqType>(grad_reqs[i]);
    e.ag_node->nn_node->attrs.name = "agvar" + std::to_string(variable_count_++);
    variables[i]->entry_ = std::move(e);  // assign last to prevent cyclic reference
  }
}

void AutogradRuntime::RecordImperativeFCompute(FCompute fn,
                                               const nnvm::Op* op,
                                               const nnvm::NodeAttrs& attrs,
                                               std::vector<NDArray> *p_inputs,
                                               std::vector<NDArray> *p_outputs) {
  RecordOp(op, attrs, p_inputs, p_outputs, nullptr);
}

void AutogradRuntime::RecordImperativeOperator(const std::shared_ptr<Operator>& opr,
                                               const nnvm::Op* op,
                                               const nnvm::NodeAttrs& attrs,
                                               std::vector<NDArray> *p_inputs,
                                               std::vector<NDArray> *p_outputs) {
  RecordOp(op, attrs, p_inputs, p_outputs, opr);
}

std::shared_ptr<AutogradRuntime> AutogradRuntime::_GetSharedRef() {
  static std::shared_ptr<AutogradRuntime> inst(new AutogradRuntime());
  return inst;
}

AutogradRuntime* AutogradRuntime::Get() {
  static AutogradRuntime *ptr = _GetSharedRef().get();
  return ptr;
}

AGNodePtr AutogradRuntime::RecordOp(const nnvm::Op* op,
                                    const nnvm::NodeAttrs& attrs,
                                    std::vector<NDArray> *p_inputs,
                                    std::vector<NDArray> *p_outputs,
                                    const std::shared_ptr<Operator>& opr) {
  LOG(INFO) << "Record op: " << ((op)? op->name : "null");
  std::vector<NDArray>& inputs  = *p_inputs;
  std::vector<NDArray>& outputs = *p_outputs;

  NodePtr nn_node = Node::Create();
  nn_node->attrs = attrs;
  nn_node->attrs.name = "agnode_" + std::to_string(node_count_++);

  AGNodePtr ag_node = AGNode::Create(nn_node);
  ag_node->opr = opr;

  for (uint32_t i = 0; i < outputs.size(); ++i) {
    outputs[i].entry_.clear();
    ag_node->outputs.push_back(outputs[i]);
    outputs[i].entry_ = AGNodeEntry{ag_node, i, 0};
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].entry_.ag_node.get() == nullptr) {
      AGNodeEntry e{AGNode::Create(Node::Create()), 0, 0};
      e.ag_node->outputs.emplace_back(inputs[i]);
      e.ag_node->out_grads.emplace_back();
      e.ag_node->nn_node->attrs.name = "agvar_" + std::to_string(variable_count_++);
      inputs[i].entry_ = std::move(e);  // assign last to prevent cyclic reference
    }
    nn_node->inputs.push_back(inputs[i].entry_.nn_entry());
    ag_node->inputs.push_back(inputs[i].entry_);
  }

  return ag_node;
}

void AutogradRuntime::ComputeGradient(
    const std::vector<NDArray>& outputs,
    const std::vector<NDArray>& grad_outputs) {
  std::vector<AGNodeEntry> heads;
  Symbol sym;
  NodeEntryMap<NDArray> feed_dict;
  for (const auto& i : outputs) {
    CHECK(i.entry_.ag_node.get() != nullptr)
      << "Cannot differentiate node because it doesn't have "
      << "computation history. Did you forget to set is_training?";
    heads.emplace_back(i.entry_);
    sym.outputs.emplace_back(i.entry_.nn_entry());
    feed_dict.insert({i.entry_.nn_entry(), i});
  }

  std::vector<NDArray> args, args_grad;
  std::vector<OpReqType> grad_reqs;
  std::unordered_map<const nnvm::Node*, std::shared_ptr<Operator>> saved_opr;
  AGDFSVisit(heads, [&](const AGNodePtr& n) {
      if (n->opr != nullptr) {
        saved_opr.insert({n->nn_node.get(), n->opr});
      } else if (n->nn_node->is_variable()) {
        args.push_back(n->outputs[0]);
        args_grad.push_back(n->out_grads[0]);
        grad_reqs.push_back(n->grad_req);
      }
      for (const auto& i : n->inputs) {
        feed_dict.insert({i.nn_entry(), i.ag_node->outputs[i.index]});
      }
    });


  if (args.size()) {
    std::map<std::string, Context> ctx_map;
    std::vector<NDArray> aux_states;
    auto exec = new exec::GraphExecutor();
    // (TODO) too hack here
    exec->saved_opr_ = saved_opr;
    exec->Init(sym, args[0].ctx(), ctx_map,
               args, args_grad, grad_reqs,
               aux_states, nullptr, feed_dict);

    std::vector<NDArray> head_grads = grad_outputs;
    if (head_grads.empty()) {
      // Create head grad arrays based on output arrays. All the values are set to be one.
      head_grads.reserve(exec->head_grad_array_.size());
      for (size_t i = 0; i < exec->output_arrays_.size(); ++i) {
        NDArray grad(exec->output_arrays_[i].shape(), exec->output_arrays_[i].ctx());
        grad = static_cast<real_t>(1.0);
        head_grads.push_back(grad);
      }
    }
    exec->Backward(head_grads);
    delete exec;
  }

  for (auto& i : heads) {
    i.ag_node->clear_history();
  }
}

AutogradRuntime::AutogradGraph AutogradRuntime::CreateGradientGraph(
    const vector<NDArray>& outputs, const vector<NDArray>& grad_outputs) {
  using namespace nnvm;
  const int kUnknownType = -1;
  // TODO(minjie): cached gradient graph.
  
  unordered_set<const Node*> forward_nodes;
  NodeEntryMap<NDArray> feed_dict;
  // Convert the computation history to symbol.
  vector<AGNodeEntry> heads;
  Symbol sym;
  for (const NDArray& out : outputs) {
    CHECK(out.entry_.ag_node.get() != nullptr)
      << "Cannot differentiate node because it doesn't have "
      << "computation history.";
    heads.emplace_back(out.entry_);
    sym.outputs.emplace_back(out.entry_.nn_entry());
    feed_dict.insert({out.entry_.nn_entry(), out});
  }

  // TODO(minjie): Shape hints?
  unordered_map<const Node*, TShape> var_shape_hints;
  unordered_map<const Node*, int> var_dtype_hints;
  vector<AGNodePtr> var_agnodes;
  AGDFSVisit(heads, [&](const AGNodePtr& n) {
      if (n->nn_node->is_variable()) {
        var_agnodes.push_back(n);
        var_shape_hints.insert({n->nn_node.get(), n->outputs[0].shape()});
        var_dtype_hints.insert({n->nn_node.get(), n->outputs[0].dtype()});
      }
      for (const AGNodeEntry& in_ent : n->inputs) {
        feed_dict.insert({in_ent.nn_entry(),
            in_ent.ag_node->outputs[in_ent.index]});
      }
      forward_nodes.insert(n->nn_node.get());
    });

  // Create head grad entries.
  vector<NodeEntry> head_grad_entries;
  for (size_t i = 0; i < outputs.size(); ++i) {
    NodePtr head_grad_node = Node::Create();
    head_grad_node->attrs.name = "__head_grad" + std::to_string(i);
    const NodeEntry grad_entry{head_grad_node, 0, 0};
    // Add attribute hints of the gradient entries of outputs
    // to output entries.
    head_grad_entries.emplace_back(exec::AttrHint(
          grad_entry, outputs[i].entry_.nn_entry()));
    var_shape_hints.insert({head_grad_node.get(), outputs[i].shape()});
    var_dtype_hints.insert({head_grad_node.get(), outputs[i].dtype()});
  }
  if (!grad_outputs.empty()) {
    // Add gradient output arrays to the feed dictionary.
    CHECK_EQ(head_grad_entries.size(), grad_outputs.size());
    for (size_t i = 0; i < head_grad_entries.size(); ++i) {
      feed_dict.insert({head_grad_entries[i], grad_outputs[i]});
    }
  }

  // Extract argument entries.
  vector<NodeEntry> arg_entries;
  vector<AGNodePtr> arg_agnodes;
  for (const AGNodePtr& n : var_agnodes) {
    if (n->grad_req != kNullOp) {
      arg_agnodes.push_back(n);
      arg_entries.emplace_back(NodeEntry{n->nn_node, 0, 0});
    }
  }

  std::vector<const nnvm::Op*> zero_ops;
  zero_ops.push_back(nnvm::Op::Get("zeros_like"));
  zero_ops.push_back(nnvm::Op::Get("_zeros"));

  // Take gradient.
  Graph g;
  g.outputs = sym.outputs;
  g = nnvm::pass::Gradient(
      g,                       // graph
      g.outputs,               // ys
      arg_entries,             // xs
      head_grad_entries,       // ys_out_grad
      exec::AggregateGradient, // aggregator
      nullptr,                 // mirror_fun
      nullptr,                 // attr_hint_fun
      zero_ops);               // zero_ops
  CHECK_EQ(g.outputs.size(), arg_entries.size());

  // Bind output (arg_grad) entries to corresponding ndarrays.
  for (size_t i = 0; i < g.outputs.size(); ++i) {
    feed_dict.insert({g.outputs[i], arg_agnodes[i]->out_grads[0]});
  }

  const auto& idx = g.indexed_graph();
  std::cout << ">>>>>Whole graph after gradient" << std::endl;
  for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
    const nnvm::Node* node = idx[nid].source;
    std::cout << "Node #" << nid << ": " << node->attrs.name << " fwd?" << (forward_nodes.count(node) == 1);
    if (!idx[nid].source->is_variable()) {
      std::cout << " op: " << idx[nid].source->attrs.op->name;
    }
    std::cout << std::endl;
  }
  std::cout << "<<<<<Whole graph after gradient" << std::endl;

  // Shape/Type inference.
  const vector<uint32_t>& input_nodes = idx.input_nodes();
  nnvm::ShapeVector arg_shapes(input_nodes.size(), TShape());
  nnvm::DTypeVector arg_dtypes(input_nodes.size(), kUnknownType);
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    const uint32_t nid = input_nodes[i];
    const Node* node = idx[nid].source;
    if (var_shape_hints.count(node)) {
      arg_shapes[i] = var_shape_hints[node];
    }
    if (var_dtype_hints.count(node)) {
      arg_dtypes[i] = var_dtype_hints[node];
    }
  }
  g = nnvm::pass::InferShape(g, arg_shapes, "__shape__");
  g = nnvm::pass::InferType(g, arg_dtypes, "__dtype__");
  
  return {g, forward_nodes, feed_dict};
}

}  // namespace autograd
}  // namespace mxnet
