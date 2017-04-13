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
  // DisableAutograd();
  // EnableAutograd();
  // TODO(yutian): Call `MXNDArrayWaitToRead`
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
  // for (auto&& i : jit_component_.Process(std::move(jit_sequence_))) {
  for (auto&& i : jit_sequence_) {
    DoStrictEvaluation(std::move(i));
  }
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
