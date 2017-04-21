/*!
 * Copyright (c) 2017 by Contributors
 * \file minpy.h
 * \brief MinPy.
 */
#ifndef MXNET_MINPY_H_
#define MXNET_MINPY_H_

#include <map>
#include <memory>
#include <vector>
#include "./executor.h"
#include "./op_attr_types.h"

// extern "C" {
//
//// Entry point. Here we return immediately but save the operation in our own
//// sequence. The output array will have correct shape but invalid `Chunk`.
// int MXImperativeInvoke(AtomicSymbolCreator creator, int num_inputs,
//                       NDArrayHandle* inputs, int* num_outputs,
//                       NDArrayHandle** outputs, int num_params,
//                       char const** param_keys, char const** param_vals);
//
//// When these two functions are called, flush JIT sequence to make sure data
/// is
//// truly ready.
// int MXNDArrayWaitToRead(NDArrayHandle handle);
// int MXNDArrayWaitToWrite(NDArrayHandle handle);
//}

namespace mxnet {
namespace minpy {

class ImperativeRuntime final {
 public:
  static ImperativeRuntime* Get();

  // Python-side utility functions.
  void EnableJIT();
  void DisableJIT();
  // TODO(yutian): Reenable autograd interface.
  // void EnableAutograd();
  // void DisableAutograd();
  void StrictEvaluate();
  // TODO(yutian): Mark as output.
  void MarkAsOutput(NDArray*) {}

  struct ComputingRecord {
    using DelayedFunction = FCompute;
    DelayedFunction delayed_function;
    nnvm::Op const* op;
    nnvm::NodeAttrs attrs;
    Context ctx;
    std::vector<engine::VarHandle> read_vars;
    std::vector<engine::VarHandle> write_vars;
    std::vector<Resource> requested;
    std::vector<NDArray> inputs;
    std::vector<NDArray> outputs;
  };

  void Invoke(ComputingRecord record);

 private:
  ImperativeRuntime() = default;
  virtual ~ImperativeRuntime() = default;
  ImperativeRuntime(ImperativeRuntime const&) = delete;
  ImperativeRuntime(ImperativeRuntime&&) = delete;
  ImperativeRuntime& operator=(ImperativeRuntime const&) = delete;
  ImperativeRuntime& operator=(ImperativeRuntime&&) = delete;

  // void PushAutogradRecord(ComputingRecord record);
  void PushJITRecord(ComputingRecord record);
  void FlushJITSequence();

  // void FlushAutogradSequence();

  // We own the following two components.
  // AutogradComponent autograd_component_{};
  // JITComponent jit_component_{};
  // std::vector<ComputingRecord> autograd_sequence_{};
  std::vector<ComputingRecord> jit_sequence_{};

  class JITGraph;
  std::vector<std::shared_ptr<JITGraph>> jit_graphs_{};

  bool jit_enabled_ = false;

  nnvm::Symbol symbol_;
  Executor *exec_;
  std::unordered_map<NDArray::Chunk*, nnvm::NodeEntry> ndarray_entry_;
  nnvm::Symbol CompileToSymbol(std::vector<ComputingRecord> *computing_records);
  void RunCompiledSymbol(Executor *exec, std::vector<NDArray> *arrays);
  // bool autograd_enabled_ = false;
};  // class ImperativeRuntime

}  // namespace minpy
}  // namespace mxnet

#endif  // MXNET_MINPY_H_
