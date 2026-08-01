// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum OrtErrorCode {
  ORT_OK = 0,
  ORT_INVALID_ARGUMENT = 1,
  ORT_NO_SUCHFILE = 2,
  ORT_NO_MODEL = 3,
  ORT_ENGINE_ERROR = 4,
  ORT_RUNTIME_EXCEPTION = 5,
  ORT_INVALID_PROTOBUF = 6,
  ORT_MODEL_LOADED = 7,
  ORT_NOT_IMPLEMENTED = 8,
  ORT_INVALID_GRAPH = 9,
  ORT_ORT_FATAL = 10,
  ORT_NULL_POINTER = 11,
  ORT_IO_BINDING_ERROR = 12,
} OrtErrorCode;

#ifdef __cplusplus
}
#endif
