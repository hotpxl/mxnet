/*!
 *  Copyright (c) 2017 by Contributors
 * \file c_api_impera.cc
 * \brief C API of mxnet ImperativeRunTime
 */

#include <mxnet/c_api.h>
#include <mxnet/minpy.h>
#include "./c_api_common.h"

using namespace mxnet;

MXNET_DLL int MXEnableJIT() {
  API_BEGIN();
  minpy::ImperativeRuntime::Get()->EnableJIT();
  API_END();
}

MXNET_DLL int MXDisableJIT() {
  API_BEGIN();
  minpy::ImperativeRuntime::Get()->DisableJIT();
  API_END();
}
