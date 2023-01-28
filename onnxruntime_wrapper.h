#ifndef ONNXRUNTIME_WRAPPER_H
#define ONNXRUNTIME_WRAPPER_H

// We want to always use the unix-like onnxruntime C APIs, even on Windows, so
// we need to undefine _WIN32 before including onnxruntime_c_api.h. However,
// this requires a careful song-and-dance.

// First, include these common headers, as they get transitively included by
// onnxruntime_c_api.h. We need to include them ourselves, first, so that the
// preprocessor will skip then while _WIN32 is undefined.
#include <stdio.h>
#include <stdlib.h>

// Next, we actually include the header.
#undef _WIN32
#include "onnxruntime_c_api.h"

// ... However, mingw will complain if _WIN32 is *not* defined! So redefine it.
#define _WIN32

#ifdef __cplusplus
extern "C" {
#endif

// Takes a pointer to the api_base struct in order to obtain the OrtApi
// pointer. Intended to be called from Go. Returns nonzero on error.
int SetAPIFromBase(OrtApiBase *api_base);

// Wraps calling ort_api->ReleaseStatus(status)
void ReleaseOrtStatus(OrtStatus *status);

// Wraps calling ort_api->CreateEnv. Returns a non-NULL status on error.
OrtStatus *CreateOrtEnv(char *name, OrtEnv **env);

// Releases the given OrtEnv.
void ReleaseOrtEnv(OrtEnv *env);

// Returns the message associated with the given ORT status.
const char *GetErrorMessage(OrtStatus *status);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // ONNXRUNTIME_WRAPPER_H
