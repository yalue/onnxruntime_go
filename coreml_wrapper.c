#include "coreml_provider_factory.h"

int CreateSessionWithCoreML(const char* model_path, uint32_t coreml_flags, OrtSession** session) {
  OrtStatus* status = NULL;
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtSessionOptions* session_options;
  status = api->CreateSessionOptions(&session_options); // Update this line if needed

  // Register CoreML Execution Provider
  status = OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, coreml_flags);
  if (status != NULL) {
    api->ReleaseStatus(status);
    return 1;
  }

  OrtEnv* env;
  status = api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "Default", &env);
  if (status != NULL) {
    api->ReleaseStatus(status);
    return 2;
  }

  // Create session
  status = api->CreateSession(env, model_path, session_options, session);
  if (status != NULL) {
    api->ReleaseStatus(status);
    return 3;
  }

  return 0;
}


int RegisterCoreMLExecutionProvider(OrtSessionOptions *session_options, uint32_t coreml_flags) {
  if (!session_options) return 1;
  OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, coreml_flags);
  if (status != NULL) {
    const OrtApi *ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ort_api->ReleaseStatus(status);
    return 2;
  }
  return 0;
}