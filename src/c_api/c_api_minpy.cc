/*!
 * Copyright (c) 2017 by Contributors
 * \file c_api_minpy.cc
 * \brief C API of MinPy.
 */
#include <mxnet/c_api.h>
#include <mxnet/minpy.h>
#include "./c_api_common.h"

MXNET_DLL int MXEnableJIT() {
  API_BEGIN();
  mxnet::minpy::ImperativeRuntime::Get()->EnableJIT();
  API_END();
}

MXNET_DLL int MXDisableJIT() {
  API_BEGIN();
  mxnet::minpy::ImperativeRuntime::Get()->DisableJIT();
  API_END();
}
