#include "onnxruntime_wrapper.h"

static const OrtApi *ort_api = NULL;
static const char *ORT_VERSION = NULL;

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
  ORT_VERSION = api_base->GetVersionString();
  if (!ort_api) return 2;
  return 0;
}

const char *GetVersion() {
  return ORT_VERSION;
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

OrtStatus *AppendExecutionProviderOpenVINOV2(OrtSessionOptions *o,
  const char **keys, const char **values, int num_keys) {
  return ort_api->SessionOptionsAppendExecutionProvider_OpenVINO_V2(o, keys,
    values, num_keys);
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

OrtStatus *CreateSessionFromFile(char *model_path, OrtEnv *env,
  OrtSession **out, OrtSessionOptions *options) {
  // Nearly identical to CreateSession, except invokes ort_api->CreateSession
  // rather than ort_api->CreateSessionFromArray.
  OrtStatus *status = NULL;
  int default_options = 0;
  if (!options) {
    default_options = 1;
    status = ort_api->CreateSessionOptions(&options);
    if (status) return status;
  }
  status = ort_api->CreateSession(env, (const ORTCHAR_T*) model_path, options,
    out);
  if (default_options) ort_api->ReleaseSessionOptions(options);
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

OrtStatus *SessionGetInputCount(OrtSession *session, size_t *result) {
  return ort_api->SessionGetInputCount(session, result);
}

OrtStatus *SessionGetOutputCount(OrtSession *session, size_t *result) {
  return ort_api->SessionGetOutputCount(session, result);
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

OrtStatus *SessionGetInputName(OrtSession *session, size_t i, char **name) {
  OrtAllocator *allocator = NULL;
  OrtStatus *status = NULL;
  status = ort_api->GetAllocatorWithDefaultOptions(&allocator);
  if (status) return status;
  return ort_api->SessionGetInputName(session, i, allocator, name);
}

OrtStatus *SessionGetOutputName(OrtSession *session, size_t i, char **name) {
  OrtAllocator *allocator = NULL;
  OrtStatus *status = NULL;
  status = ort_api->GetAllocatorWithDefaultOptions(&allocator);
  if (status) return status;
  return ort_api->SessionGetOutputName(session, i, allocator, name);
}

OrtStatus *FreeWithDefaultORTAllocator(void *to_free) {
  OrtAllocator *allocator = NULL;
  OrtStatus *status = NULL;
  status = ort_api->GetAllocatorWithDefaultOptions(&allocator);
  if (status) return status;
  return ort_api->AllocatorFree(allocator, to_free);
}

OrtStatus *SessionGetInputTypeInfo(OrtSession *session, size_t i,
  OrtTypeInfo **out) {
  return ort_api->SessionGetInputTypeInfo(session, i, out);
}

OrtStatus *SessionGetOutputTypeInfo(OrtSession *session, size_t i,
  OrtTypeInfo **out) {
  return ort_api->SessionGetOutputTypeInfo(session, i, out);
}

void ReleaseTypeInfo(OrtTypeInfo *o) {
  ort_api->ReleaseTypeInfo(o);
}

OrtStatus *GetONNXTypeFromTypeInfo(OrtTypeInfo *info, enum ONNXType *out) {
  return ort_api->GetOnnxTypeFromTypeInfo(info, out);
}

OrtStatus *CastTypeInfoToTensorInfo(OrtTypeInfo *type_info,
  OrtTensorTypeAndShapeInfo **out) {
  return ort_api->CastTypeInfoToTensorInfo(type_info,
    (const OrtTensorTypeAndShapeInfo **) out);
}

OrtStatus *SessionGetModelMetadata(OrtSession *s, OrtModelMetadata **m) {
  return ort_api->SessionGetModelMetadata(s, m);
}

void ReleaseModelMetadata(OrtModelMetadata *m) {
  return ort_api->ReleaseModelMetadata(m);
}

OrtStatus *ModelMetadataGetProducerName(OrtModelMetadata *m, char **name) {
  OrtAllocator *allocator = NULL;
  OrtStatus *status = NULL;
  status = ort_api->GetAllocatorWithDefaultOptions(&allocator);
  if (status) return status;
  return ort_api->ModelMetadataGetProducerName(m, allocator, name);
}

OrtStatus *ModelMetadataGetGraphName(OrtModelMetadata *m, char **name) {
  OrtAllocator *allocator = NULL;
  OrtStatus *status = NULL;
  status = ort_api->GetAllocatorWithDefaultOptions(&allocator);
  if (status) return status;
  return ort_api->ModelMetadataGetGraphName(m, allocator, name);
}

OrtStatus *ModelMetadataGetDomain(OrtModelMetadata *m, char **domain) {
  OrtAllocator *allocator = NULL;
  OrtStatus *status = NULL;
  status = ort_api->GetAllocatorWithDefaultOptions(&allocator);
  if (status) return status;
  return ort_api->ModelMetadataGetDomain(m, allocator, domain);
}

OrtStatus *ModelMetadataGetDescription(OrtModelMetadata *m, char **desc) {
  OrtAllocator *allocator = NULL;
  OrtStatus *status = NULL;
  status = ort_api->GetAllocatorWithDefaultOptions(&allocator);
  if (status) return status;
  return ort_api->ModelMetadataGetDescription(m, allocator, desc);
}

OrtStatus *ModelMetadataLookupCustomMetadataMap(OrtModelMetadata *m, char *key,
  char **value) {
  OrtAllocator *allocator = NULL;
  OrtStatus *status = NULL;
  status = ort_api->GetAllocatorWithDefaultOptions(&allocator);
  if (status) return status;
  return ort_api->ModelMetadataLookupCustomMetadataMap(m, allocator, key,
    value);
}

OrtStatus *ModelMetadataGetCustomMetadataMapKeys(OrtModelMetadata *m,
  char ***keys, int64_t *num_keys) {
  OrtAllocator *allocator = NULL;
  OrtStatus *status = NULL;
  status = ort_api->GetAllocatorWithDefaultOptions(&allocator);
  if (status) return status;
  return ort_api->ModelMetadataGetCustomMetadataMapKeys(m, allocator, keys,
    num_keys);
}

OrtStatus *ModelMetadataGetVersion(OrtModelMetadata *m, int64_t *version) {
  return ort_api->ModelMetadataGetVersion(m, version);
}

OrtStatus *GetValue(OrtValue *container, int index, OrtValue **dst) {
  OrtAllocator *allocator = NULL;
  OrtStatus *status = NULL;
  status = ort_api->GetAllocatorWithDefaultOptions(&allocator);
  if (status) return status;
  return ort_api->GetValue(container, index, allocator, dst);
}

OrtStatus *GetValueType(OrtValue *v, enum ONNXType *out) {
  return ort_api->GetValueType(v, out);
}

OrtStatus *GetValueCount(OrtValue *v, size_t *out) {
  return ort_api->GetValueCount(v, out);
}

OrtStatus *CreateOrtValue(OrtValue **in, size_t num_values,
  enum ONNXType value_type, OrtValue **out) {
  return ort_api->CreateValue((const OrtValue* const*) in, num_values,
    value_type, out);
}

// TRAINING API WRAPPER

static const OrtTrainingApi *ort_training_api = NULL;

void SetTrainingApi() {
  ort_training_api = ort_api->GetTrainingApi(ORT_API_VERSION);
}

int IsTrainingApiSupported() {
  return ort_training_api != NULL;
}

OrtStatus *CreateCheckpoint(void *checkpoint_data, size_t checkpoint_data_length, OrtCheckpointState **out) {
  OrtStatus *status = NULL;
  status = ort_training_api->LoadCheckpointFromBuffer(checkpoint_data, checkpoint_data_length, out);
  return status;
}

OrtStatus *CreateTrainingSessionFromBuffer(OrtCheckpointState *checkpoint_state,
    void *training_model_data, size_t training_model_data_length,
    void *eval_model_data, size_t eval_model_data_length,
    void *optim_model_data, size_t optim_model_data_length,
    OrtEnv *env, OrtTrainingSession **out, OrtSessionOptions *options) {
  OrtStatus *status = NULL;
  int default_options = 0;
  if (!options) {
    default_options = 1;
    status = ort_api->CreateSessionOptions(&options);
    if (status) return status;
  }
  status = ort_training_api->CreateTrainingSessionFromBuffer(env, options, checkpoint_state,
  training_model_data, training_model_data_length, eval_model_data, eval_model_data_length,
  optim_model_data, optim_model_data_length, out);
  if (default_options) {
    ort_api->ReleaseSessionOptions(options);
  }
  return status;
}

OrtStatus *CreateTrainingSessionFromPaths(OrtCheckpointState *checkpoint_state,
    char *training_model_path, char *eval_model_path, char *optim_model_path, 
    OrtEnv *env, OrtTrainingSession **out, OrtSessionOptions *options) {
  OrtStatus *status = NULL;
  int default_options = 0;
  if (!options) {
    default_options = 1;
    status = ort_api->CreateSessionOptions(&options);
    if (status) return status;
  }
  status = ort_training_api->CreateTrainingSession(env, options, checkpoint_state,
  training_model_path, eval_model_path, optim_model_path, out);
  if (default_options) {
    ort_api->ReleaseSessionOptions(options);
  }
  return status;
}

OrtStatus *TrainingSessionGetInputCount(OrtTrainingSession *training_session, size_t *result_training, size_t *result_eval) {
  OrtStatus *status = NULL;
  status = ort_training_api->TrainingSessionGetTrainingModelInputCount(training_session, result_training);
  if (status) return status;
  status = ort_training_api->TrainingSessionGetEvalModelInputCount(training_session, result_eval);
  return status;
}

OrtStatus *TrainingSessionGetOutputCount(OrtTrainingSession *training_session, size_t *result_training, size_t *result_eval) {
  OrtStatus *status = NULL;
  status = ort_training_api->TrainingSessionGetTrainingModelOutputCount(training_session, result_training);
  if (status) return status;
  status = ort_training_api->TrainingSessionGetEvalModelOutputCount(training_session, result_eval);
  return status;
}

OrtStatus *TrainingSessionGetTrainingInputName(OrtTrainingSession *training_session, size_t i, char **name) {
  OrtAllocator *allocator = NULL;
  OrtStatus *status = NULL;
  status = ort_api->GetAllocatorWithDefaultOptions(&allocator);
  if (status) return status;
  return ort_training_api->TrainingSessionGetTrainingModelInputName(training_session, i, allocator, name);
}

OrtStatus *TrainingSessionGetTrainingOutputName(OrtTrainingSession *training_session, size_t i, char **name) {
  OrtAllocator *allocator = NULL;
  OrtStatus *status = NULL;
  status = ort_api->GetAllocatorWithDefaultOptions(&allocator);
  if (status) return status;
  return ort_training_api->TrainingSessionGetTrainingModelOutputName(training_session, i, allocator, name);
}

OrtStatus *TrainingSessionGetEvalInputName(OrtTrainingSession *training_session, size_t i, char **name) {
  OrtAllocator *allocator = NULL;
  OrtStatus *status = NULL;
  status = ort_api->GetAllocatorWithDefaultOptions(&allocator);
  if (status) return status;
  return ort_training_api->TrainingSessionGetEvalModelInputName(training_session, i, allocator, name);
}

OrtStatus *TrainingSessionGetEvalOutputName(OrtTrainingSession *training_session, size_t i, char **name) {
  OrtAllocator *allocator = NULL;
  OrtStatus *status = NULL;
  status = ort_api->GetAllocatorWithDefaultOptions(&allocator);
  if (status) return status;
  return ort_training_api->TrainingSessionGetEvalModelOutputName(training_session, i, allocator, name);
}

OrtStatus *TrainStep(OrtTrainingSession *training_session, size_t inputs_len, OrtValue **inputs, size_t output_len, OrtValue **outputs) {
    OrtStatus *status = NULL;
    status = ort_training_api->TrainStep(training_session, NULL, inputs_len, (const OrtValue* const*) inputs, output_len, outputs);
    return status;
}

OrtStatus *OptimizerStep(OrtTrainingSession *training_session) {
    OrtStatus *status = NULL;
    status = ort_training_api->OptimizerStep(training_session, NULL);
    return status;
}

OrtStatus *LazyResetGrad(OrtTrainingSession *training_session) {
    OrtStatus *status = NULL;
    status = ort_training_api->LazyResetGrad(training_session);
    return status;
}

OrtStatus *SaveCheckpoint(OrtCheckpointState *checkpoint, char *path, size_t include_optimizer) {
  OrtStatus *status = NULL;
  status = ort_training_api->SaveCheckpoint(checkpoint, path, include_optimizer);
  return status;
}

OrtStatus *ExportModel(OrtTrainingSession *training_session, char *path, size_t outputs_len, char **output_names) {
    OrtStatus *status = NULL;
    status = ort_training_api->ExportModelForInferencing(training_session, path, outputs_len, (const char* const*) output_names);
    return status;
}

void ReleaseOrtTrainingSession(OrtTrainingSession *session) {
  ort_training_api->ReleaseTrainingSession(session);
}

void ReleaseCheckpointState(OrtCheckpointState *checkpoint) {
  ort_training_api->ReleaseCheckpointState(checkpoint);
}

