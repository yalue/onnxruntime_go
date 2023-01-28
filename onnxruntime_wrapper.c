#include "onnxruntime_wrapper.h"

static const OrtApi *ort_api = NULL;

int SetAPIFromBase(OrtApiBase *api_base) {
  if (!api_base) return 1;
  ort_api = api_base->GetApi(ORT_API_VERSION);
  if (!ort_api) return 2;
  return 0;
}

void ReleaseOrtStatus(OrtStatus *status) {
  ort_api->ReleaseStatus(status);
}

void ReleaseOrtEnv(OrtEnv *env) {
  ort_api->ReleaseEnv(env);
}

OrtStatus *CreateOrtEnv(char *name, OrtEnv **env) {
  return ort_api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, name, env);
}

const char *GetErrorMessage(OrtStatus *status) {
  if (!status) return "No error (NULL status)";
  return ort_api->GetErrorMessage(status);
}
