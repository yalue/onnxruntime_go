#include "onnxruntime_wrapper.h"

static const OrtApi *ort_api = NULL;

static AppendCoreMLProviderFn append_coreml_provider_fn = NULL;

// The dml_provider_factory.h header for using DirectML is annoying to include
// here for a couple reasons:
//  - It contains C++
//  - It includes d3d12.h and DirectML.h, both of which may be hard to set up
//    under mingw
// Fortunately, the basic AppendExecutionProvider_DML function from the
// OrtDmlApi struct does not rely on any of these things, but we still need the
// struct definition itself. Obviously, copying it here is not perfect, and
// we'll need to keep an eye on it to make sure it doesn't change between
// updates. Most importantly, we need to make sure that the one function we
// care about remains at the same place in the struct.  Since it's first,
// hopefully it's unlikely to change.
typedef OrtStatus* (*AppendDirectMLProviderFn)(OrtSessionOptions*, int);
typedef struct {
  AppendDirectMLProviderFn SessionOptionsAppendExecutionProvider_DML;
  // All of these functions pointers should be irrelevant (and they depend on
  // other definitions from dml_provider_factory.h), but I'll copy them here
  // regardless as plain void*s. GetExecutionProviderApi shouldn't write to
  // this struct anyway, as it only provides a const pointer to it.
  void *SessionOptionsAppendExecutionProvider_DML1;
  void *CreateGPUAllocationFromD3DResource;
  void *FreeGPUAllocation;
  void *GetD3D12ResourceFromAllocation;
  void *SessionOptionsAppendExecutionProvider_DML2;
} DummyOrtDMLAPI;

int SetAPIFromBase(OrtApiBase *api_base) {
  if (!api_base) return 1;
  ort_api = api_base->GetApi(ORT_API_VERSION);
  if (!ort_api) return 2;
  return 0;
}

void SetCoreMLProviderFunctionPointer(void *ptr) {
  append_coreml_provider_fn = (AppendCoreMLProviderFn) ptr;
}

void ReleaseOrtStatus(OrtStatus *status) {
  ort_api->ReleaseStatus(status);
}

OrtStatus *CreateOrtEnv(char *name, OrtEnv **env) {
  return ort_api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, name, env);
}

OrtStatus *DisableTelemetry(OrtEnv *env) {
  return ort_api->DisableTelemetryEvents(env);
}

OrtStatus *EnableTelemetry(OrtEnv *env) {
  return ort_api->EnableTelemetryEvents(env);
}

void ReleaseOrtEnv(OrtEnv *env) {
  ort_api->ReleaseEnv(env);
}

OrtStatus *CreateOrtMemoryInfo(OrtMemoryInfo **mem_info) {
  return ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
    mem_info);
}

void ReleaseOrtMemoryInfo(OrtMemoryInfo *info) {
  ort_api->ReleaseMemoryInfo(info);
}

const char *GetErrorMessage(OrtStatus *status) {
  if (!status) return "No error (NULL status)";
  return ort_api->GetErrorMessage(status);
}

OrtStatus *CreateSessionOptions(OrtSessionOptions **o) {
  return ort_api->CreateSessionOptions(o);
}

void ReleaseSessionOptions(OrtSessionOptions *o) {
  ort_api->ReleaseSessionOptions(o);
}

OrtStatus *SetIntraOpNumThreads(OrtSessionOptions *o, int n) {
  return ort_api->SetIntraOpNumThreads(o, n);
}

OrtStatus *SetInterOpNumThreads(OrtSessionOptions *o, int n) {
  return ort_api->SetInterOpNumThreads(o, n);
}

OrtStatus *SetCpuMemArena(OrtSessionOptions *o, int use_arena){
  if (use_arena)
    return ort_api->EnableCpuMemArena(o);
  return ort_api->DisableCpuMemArena(o);
}

OrtStatus *SetMemPattern(OrtSessionOptions *o, int use_mem_pattern){
  if (use_mem_pattern)
    return ort_api->EnableMemPattern(o);
  return ort_api->DisableMemPattern(o);
}

OrtStatus *AppendExecutionProviderCUDAV2(OrtSessionOptions *o,
  OrtCUDAProviderOptionsV2 *cuda_options) {
  return ort_api->SessionOptionsAppendExecutionProvider_CUDA_V2(o,
    cuda_options);
}

OrtStatus *CreateCUDAProviderOptions(OrtCUDAProviderOptionsV2 **o) {
  return ort_api->CreateCUDAProviderOptions(o);
}

void ReleaseCUDAProviderOptions(OrtCUDAProviderOptionsV2 *o) {
  ort_api->ReleaseCUDAProviderOptions(o);
}

OrtStatus *UpdateCUDAProviderOptions(OrtCUDAProviderOptionsV2 *o,
  const char **keys, const char **values, int num_keys) {
  return ort_api->UpdateCUDAProviderOptions(o, keys, values, num_keys);
}

OrtStatus *CreateTensorRTProviderOptions(OrtTensorRTProviderOptionsV2 **o) {
  return ort_api->CreateTensorRTProviderOptions(o);
}

void ReleaseTensorRTProviderOptions(OrtTensorRTProviderOptionsV2 *o) {
  ort_api->ReleaseTensorRTProviderOptions(o);
}

OrtStatus *UpdateTensorRTProviderOptions(OrtTensorRTProviderOptionsV2 *o,
  const char **keys, const char **values, int num_keys) {
  return ort_api->UpdateTensorRTProviderOptions(o, keys, values, num_keys);
}

OrtStatus *AppendExecutionProviderTensorRTV2(OrtSessionOptions *o,
  OrtTensorRTProviderOptionsV2 *tensor_rt_options) {
  return ort_api->SessionOptionsAppendExecutionProvider_TensorRT_V2(o,
    tensor_rt_options);
}

OrtStatus *AppendExecutionProviderCoreML(OrtSessionOptions *o,
  uint32_t flags) {
  if (!append_coreml_provider_fn) {
    return ort_api->CreateStatus(ORT_NOT_IMPLEMENTED, "Your platform or "
      "onnxruntime library does not support CoreML");
  }
  return append_coreml_provider_fn(o, flags);
}

OrtStatus *AppendExecutionProviderDirectML(OrtSessionOptions *o,
  int device_id) {
  DummyOrtDMLAPI *dml_api = NULL;
  OrtStatus *status = NULL;
  status = ort_api->GetExecutionProviderApi("DML", ORT_API_VERSION,
    (const void **) (&dml_api));
  if (status) return status;
  status = dml_api->SessionOptionsAppendExecutionProvider_DML(o, device_id);
  return status;
}

OrtStatus *CreateSession(void *model_data, size_t model_data_length,
    OrtEnv *env, OrtSession **out, OrtSessionOptions *options) {
  OrtStatus *status = NULL;
  int default_options = 0;
  if (!options) {
    default_options = 1;
    status = ort_api->CreateSessionOptions(&options);
    if (status) return status;
  }
  status = ort_api->CreateSessionFromArray(env, model_data, model_data_length,
    options, out);
  if (default_options) {
    // If we created a default, empty, options struct, we don't need to keep it
    // after creating the session.
    ort_api->ReleaseSessionOptions(options);
  }
  return status;
}

OrtStatus *RunOrtSession(OrtSession *session,
  OrtValue **inputs, char **input_names, int input_count,
  OrtValue **outputs, char **output_names, int output_count) {
  OrtStatus *status = NULL;
  status = ort_api->Run(session, NULL, (const char* const*) input_names,
    (const OrtValue* const*) inputs, input_count,
    (const char* const*) output_names, output_count, outputs);
  return status;
}

void ReleaseOrtSession(OrtSession *session) {
  ort_api->ReleaseSession(session);
}

void ReleaseOrtValue(OrtValue *value) {
  ort_api->ReleaseValue(value);
}

OrtStatus *CreateOrtTensorWithShape(void *data, size_t data_size,
  int64_t *shape, int64_t shape_size, OrtMemoryInfo *mem_info,
  ONNXTensorElementDataType dtype, OrtValue **out) {
  OrtStatus *status = NULL;
  status = ort_api->CreateTensorWithDataAsOrtValue(mem_info, data, data_size,
    shape, shape_size, dtype, out);
  return status;
}

OrtStatus *GetTensorTypeAndShape(const OrtValue *value, OrtTensorTypeAndShapeInfo **out) {
  return ort_api->GetTensorTypeAndShape(value, out);
}

OrtStatus *GetDimensionsCount(const OrtTensorTypeAndShapeInfo *info, size_t *out) {
  return ort_api->GetDimensionsCount(info, out);
}

OrtStatus *GetDimensions(const OrtTensorTypeAndShapeInfo *info, int64_t *dim_values, size_t dim_values_length) {
  return ort_api->GetDimensions(info, dim_values, dim_values_length);
}

OrtStatus *GetTensorElementType(const OrtTensorTypeAndShapeInfo *info, enum ONNXTensorElementDataType *out) {
  return ort_api->GetTensorElementType(info, out);
}

void ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo *input) {
  ort_api->ReleaseTensorTypeAndShapeInfo(input);
}

OrtStatus *GetTensorMutableData(OrtValue *value, void **out) {
  return ort_api->GetTensorMutableData(value, out);
}
