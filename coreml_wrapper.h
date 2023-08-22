#include "coreml_provider_factory.h" // Include the CoreML header

// Function to register the CoreML Execution Provider
int RegisterCoreMLExecutionProvider(OrtSessionOptions *session_options, uint32_t coreml_flags);
int CreateSessionWithCoreML(const char* model_path, uint32_t coreml_flags, OrtSession** session);
