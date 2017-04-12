/*!
 * Copyright (c) 2017 by Contributors
 * \file c_api_ndarray.h
 * \brief C API of MXNet.
 */
#ifndef MXNET_C_API_C_API_NDARRAY_H_
#define MXNET_C_API_C_API_NDARRAY_H_
#include <mxnet/engine.h>
#include <mxnet/op_attr_types.h>

void PushFCompute(mxnet::FCompute const& fn, nnvm::Op const* op,
                  nnvm::NodeAttrs const& attrs, mxnet::Context const& ctx,
                  std::vector<mxnet::engine::VarHandle> const& read_vars,
                  std::vector<mxnet::engine::VarHandle> const& write_vars,
                  std::vector<mxnet::Resource> const& requested,
                  std::vector<mxnet::NDArray> const& ndinputs,
                  std::vector<mxnet::NDArray> const& ndoutputs);
#endif  // MXNET_C_API_C_API_NDARRAY_H_
