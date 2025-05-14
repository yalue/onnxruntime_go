#ifndef ONNXRUNTIME_WRAPPER_H
#define ONNXRUNTIME_WRAPPER_H

// We want to always use the unix-like onnxruntime C APIs, even on Windows, so
// we need to undefine _WIN32 before including onnxruntime_c_api.h. However,
// this requires a careful song-and-dance.

// First, include these common headers, as they get transitively included by
// onnxruntime_c_api.h. We need to include them ourselves, first, so that the
// preprocessor will skip them while _WIN32 is undefined.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Next, we actually include the header.
#undef _WIN32
#include "onnxruntime_c_api.h"

// ... However, mingw will complain if _WIN32 is *not* defined! So redefine it.
#define _WIN32

#ifdef __cplusplus
extern "C" {
#endif

// Used for the OrtSessionOptionsAppendExecutionProvider_CoreML function
// pointer on supported systems. Must match the signature in
// coreml_provider_factory.h provided along with the onnxruntime releases for
// Apple platforms.
typedef OrtStatus* (*AppendCoreMLProviderFn)(OrtSessionOptions*, uint32_t);

// Takes a pointer to the api_base struct in order to obtain the OrtApi
// pointer. Intended to be called from Go. Returns nonzero on error.
int SetAPIFromBase(OrtApiBase *api_base);

// Get the version of the Onnxruntime library for logging.
const char *GetVersion();

// OrtSessionOptionsAppendExecutionProvider_CoreML is exported directly from
// the Apple .dylib, so we call this function on Apple platforms to set the
// function pointer to the correct address. On other platforms, the function
// pointer should remain NULL.
void SetCoreMLProviderFunctionPointer(void *ptr);

// Wraps ort_api->ReleaseStatus(status)
void ReleaseOrtStatus(OrtStatus *status);

// Wraps calling ort_api->CreateEnv. Returns a non-NULL status on error.
OrtStatus *CreateOrtEnv(char *name, OrtEnv **env);

// Wraps calling ort_api->UpdateEnvWithCustomLogLevel. Return a non-NULL status on error.
OrtStatus *UpdateEnvWithCustomLogLevel(OrtEnv *ort_env, OrtLoggingLevel log_severity_level);

// Wraps ort_api->DisableTelemetryEvents. Returns a non-NULL status on error.
OrtStatus *DisableTelemetry(OrtEnv *env);

// Wraps ort_api->EnableTelemetryEvents. Returns a non-NULL status on error.
OrtStatus *EnableTelemetry(OrtEnv *env);

// Wraps ort_api->ReleaseEnv
void ReleaseOrtEnv(OrtEnv *env);

// Wraps ort_api->CreateCpuMemoryInfo with some basic, default settings.
OrtStatus *CreateOrtMemoryInfo(OrtMemoryInfo **mem_info);

// Wraps ort_api->ReleaseMemoryInfo
void ReleaseOrtMemoryInfo(OrtMemoryInfo *info);

// Returns the message associated with the given ORT status.
const char *GetErrorMessage(OrtStatus *status);

// Wraps ort_api->CreateSessionOptions
OrtStatus *CreateSessionOptions(OrtSessionOptions **o);

// Wraps ort_api->ReleaseSessionOptions
void ReleaseSessionOptions(OrtSessionOptions *o);

// Wraps ort_api->SetSessionExecutionMode
OrtStatus *SetSessionExecutionMode(OrtSessionOptions *o, int new_mode);

// Wraps ort_api->SetSessionGraphOptimizationLevel
OrtStatus *SetSessionGraphOptimizationLevel(OrtSessionOptions *o, int level);

// Wraps ort_api->AddSessionConfigEntry
OrtStatus *AddSessionConfigEntry(OrtSessionOptions *o, char *key, char *value);

// Wraps ort_api->HasSessionConfigEntry
OrtStatus *HasSessionConfigEntry(OrtSessionOptions *o, char *key, int *result);

// Wraps ort_api->GetSessionConfigEntry
OrtStatus *GetSessionConfigEntry(OrtSessionOptions *o, char *key, char *result,
    size_t *required_size);

// Wraps ort_api->SetIntraOpNumThreads
OrtStatus *SetIntraOpNumThreads(OrtSessionOptions *o, int n);

// Wraps ort_api->SetInterOpNumThreads
OrtStatus *SetInterOpNumThreads(OrtSessionOptions *o, int n);

// Wraps ort_api->EnableCpuMemArena & ort_api->DisableCpuMemArena
OrtStatus *SetCpuMemArena(OrtSessionOptions *o, int use_arena);

// Wraps ort_api->EnableMemPattern & ort_api->DisableMemPattern
OrtStatus *SetMemPattern(OrtSessionOptions *o, int use_mem_pattern);

// Wraps ort_api->SessionOptionsAppendExecutionProvider_CUDA_V2
OrtStatus *AppendExecutionProviderCUDAV2(OrtSessionOptions *o,
  OrtCUDAProviderOptionsV2 *cuda_options);

// Wraps ort_api->CreateCUDAProviderOptions
OrtStatus *CreateCUDAProviderOptions(OrtCUDAProviderOptionsV2 **o);

// Wraps ort_api->ReleaseCUDAProviderOptions
void ReleaseCUDAProviderOptions(OrtCUDAProviderOptionsV2 *o);

// Wraps ort_api->UpdateCUDAProviderOptions
OrtStatus *UpdateCUDAProviderOptions(OrtCUDAProviderOptionsV2 *o,
  const char **keys, const char **values, int num_keys);

// Wraps ort_api->CreateTensorRTProviderOptions
OrtStatus *CreateTensorRTProviderOptions(OrtTensorRTProviderOptionsV2 **o);

// Wraps ort_api->ReleaseTensorRTProviderOptions
void ReleaseTensorRTProviderOptions(OrtTensorRTProviderOptionsV2 *o);

// Wraps ort_api->UpdateTensorRTProviderOptions
OrtStatus *UpdateTensorRTProviderOptions(OrtTensorRTProviderOptionsV2 *o,
  const char **keys, const char **values, int num_keys);

// Wraps ort_api->SessionOptionsAppendExecutionProvider_TensorRT_V2
OrtStatus *AppendExecutionProviderTensorRTV2(OrtSessionOptions *o,
  OrtTensorRTProviderOptionsV2 *tensor_rt_options);

// Wraps OrtSessionOptionsAppendExecutionProvider_CoreML, exported from the
// dylib on Apple devices. Safely returns a non-NULL status on other platforms.
OrtStatus *AppendExecutionProviderCoreML(OrtSessionOptions *o,
  uint32_t flags);

// Wraps OrtApi method SessionOptionsAppendExecutionProvider specifically for CoreML
// with the new options-based API while including the check that coreml is supported.
OrtStatus *AppendExecutionProviderCoreMLV2(OrtSessionOptions *o,
  const char **keys, const char **values, size_t num_options);

// Wraps getting the OrtDmlApi struct and calling
// dml_api->SessionOptionsAppendExecutionProvider_DML.
OrtStatus *AppendExecutionProviderDirectML(OrtSessionOptions *o,
  int device_id);

// Wraps ort_api->AppendExecutionProvider_OpenVINO_V2
OrtStatus *AppendExecutionProviderOpenVINOV2(OrtSessionOptions *o,
  const char **keys, const char **values, int num_keys);

// Wraps ort_api->AppendExecutionProvider
OrtStatus *AppendExecutionProvider(OrtSessionOptions *o, const char *provider_name,
  const char **keys, const char **values, int num_keys);

// Creates an ORT session using the given model. The given options pointer may
// be NULL; if it is, then we'll use default options.
OrtStatus *CreateSession(void *model_data, size_t model_data_length,
  OrtEnv *env, OrtSession **out, OrtSessionOptions *options);

// Like the CreateSession function, but takes a path to a model rather than a
// buffer containing it.
OrtStatus *CreateSessionFromFile(char *model_path, OrtEnv *env,
  OrtSession **out, OrtSessionOptions *options);

// Runs an ORT session with the given input and output tensors, along with
// their names.
OrtStatus *RunOrtSession(OrtSession *session,
  OrtValue **inputs, char **input_names, int input_count,
  OrtValue **outputs, char **output_names, int output_count);

// Wraps ort_api->RunWithBinding.
OrtStatus *RunSessionWithBinding(OrtSession *session, OrtIoBinding *b);

// Wraps ort_api->ReleaseSession
void ReleaseOrtSession(OrtSession *session);

// Wraps ort_api->CreateIoBinding
OrtStatus *CreateIoBinding(OrtSession *session, OrtIoBinding **out);

// Wraps ort_api->ReleaseIoBinding
void ReleaseIoBinding(OrtIoBinding *b);

// Wraps ort_api->BindInput
OrtStatus *BindInput(OrtIoBinding *b, char *name, OrtValue *value);

// Wraps ort_api->BindOutput
OrtStatus *BindOutput(OrtIoBinding *b, char *name, OrtValue *value);

// Wraps ort_api->GetBoundOutputNames. Uses the default allocator. The caller
// must free the buffer and lengths using FreeWithDefaultOrtAllocator after
// this is done.
OrtStatus *GetBoundOutputNames(OrtIoBinding *b, char **buffer,
  size_t **lengths, size_t *count);

// Wraps ort_api->GetBoundOutputNames. Uses the default allocator, and the
// caller must free the buffer of values using FreeWithDefaultOrtAllocator.
OrtStatus *GetBoundOutputValues(OrtIoBinding *b, OrtValue ***buffer,
  size_t *count);

// Wraps ort_api->ClearBoundInputs.
void ClearBoundInputs(OrtIoBinding *b);

// Wraps ort_api->ClearBoundOutputs.
void ClearBoundOutputs(OrtIoBinding *b);

// Wraps ort_api->SessionGetInputCount.
OrtStatus *SessionGetInputCount(OrtSession *session, size_t *result);

// Wraps ort_api->SessionGetOutputCount.
OrtStatus *SessionGetOutputCount(OrtSession *session, size_t *result);

// Used to free OrtValue instances, such as tensors.
void ReleaseOrtValue(OrtValue *value);

// Creates an OrtValue tensor with the given shape, and backed by the user-
// supplied data buffer.
OrtStatus *CreateOrtTensorWithShape(void *data, size_t data_size,
  int64_t *shape, int64_t shape_size, OrtMemoryInfo *mem_info,
  ONNXTensorElementDataType dtype, OrtValue **out);

// Wraps ort_api->GetTensorTypeAndShape
OrtStatus *GetTensorTypeAndShape(const OrtValue *value, OrtTensorTypeAndShapeInfo **out);

// Wraps ort_api->GetDimensionsCount
OrtStatus *GetDimensionsCount(const OrtTensorTypeAndShapeInfo *info, size_t *out);

// Wraps ort_api->GetDimensions
OrtStatus *GetDimensions(const OrtTensorTypeAndShapeInfo *info, int64_t *dim_values, size_t dim_values_length);

// Wraps ort_api->GetTensorElementType
OrtStatus *GetTensorElementType(const OrtTensorTypeAndShapeInfo *info, enum ONNXTensorElementDataType *out);

// Wraps ort_api->ReleaseTensorTypeAndShapeInfo
void ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo *input);

// Wraps ort_api->GetTensorMutableData
OrtStatus *GetTensorMutableData(OrtValue *value, void **out);

// Wraps ort_api->SessionGetInputName, using the default allocator.
OrtStatus *SessionGetInputName(OrtSession *session, size_t i, char **name);

// Wraps ort_api->SessionGetOutputName, using the default allocator.
OrtStatus *SessionGetOutputName(OrtSession *session, size_t i, char **name);

// Frees anything that was allocated using the default ORT allocator.
OrtStatus *FreeWithDefaultORTAllocator(void *to_free);

// Wraps ort_api->SessionGetInputTypeInfo.
OrtStatus *SessionGetInputTypeInfo(OrtSession *session, size_t i,
  OrtTypeInfo **out);

// Wraps ort_api->SessionGetOutputTypeInfo.
OrtStatus *SessionGetOutputTypeInfo(OrtSession *session, size_t i,
  OrtTypeInfo **out);

// If the type_info is for a tensor, sets out to the a pointer to the tensor's
// NameAndTypeInfo. Do _not_ free the out pointer; it will be freed when
// type_info is released.
//
// Wraps ort_api->CastTypeInfoToTensorInfo.
OrtStatus *CastTypeInfoToTensorInfo(OrtTypeInfo *type_info,
  OrtTensorTypeAndShapeInfo **out);

// Wraps ort_api->GetOnnxTypeFromTypeInfo.
OrtStatus *GetONNXTypeFromTypeInfo(OrtTypeInfo *info, enum ONNXType *out);

// Wraps ort_api->FreeTypeInfo.
void ReleaseTypeInfo(OrtTypeInfo *o);

// Wraps ort_spi->SessionGetModelMetadata.
OrtStatus *SessionGetModelMetadata(OrtSession *s, OrtModelMetadata **out);

// Wraps ort_api->ReleaseModelMetadata.
void ReleaseModelMetadata(OrtModelMetadata *m);

// Wraps ort_api->ModelMetadataGetProducerName, using the default allocator.
OrtStatus *ModelMetadataGetProducerName(OrtModelMetadata *m, char **name);

// Wraps ort_api->ModelMetadataGetGraphName, using the default allocator.
OrtStatus *ModelMetadataGetGraphName(OrtModelMetadata *m, char **name);

// Wraps ort_api->ModelMetadataGetDomain, using the default allocator.
OrtStatus *ModelMetadataGetDomain(OrtModelMetadata *m, char **domain);

// Wraps ort_api->ModelMetadataGetDescription, using the default allocator.
OrtStatus *ModelMetadataGetDescription(OrtModelMetadata *m, char **desc);

// Wraps ort_api->ModelMetadataLookupCustomMetadataMap, using the default
// allocator.
OrtStatus *ModelMetadataLookupCustomMetadataMap(OrtModelMetadata *m, char *key,
  char **value);

// Wraps ort_api->ModelMetadataGetCustomMetadataMapKeys, using the default
// allocator.
OrtStatus *ModelMetadataGetCustomMetadataMapKeys(OrtModelMetadata *m,
  char ***keys, int64_t *num_keys);

// Wraps ort_api->ModelMetadataGetVersion.
OrtStatus *ModelMetadataGetVersion(OrtModelMetadata *m, int64_t *version);

// Wraps ort_api->GetValue. Uses the default allocator.
OrtStatus *GetValue(OrtValue *container, int index, OrtValue **dst);

// Wraps ort_api->GetValueType.
OrtStatus *GetValueType(OrtValue *v, enum ONNXType *out);

// Wraps ort_api->GetValueCount.
OrtStatus *GetValueCount(OrtValue *v, size_t *out);

// Wraps ort_api->CreateValue to create a map or a sequence.
OrtStatus *CreateOrtValue(OrtValue **in, size_t num_values,
  enum ONNXType value_type, OrtValue **out);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // ONNXRUNTIME_WRAPPER_H
