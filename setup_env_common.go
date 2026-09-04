//go:build !windows

package onnxruntime_go

/*
#cgo LDFLAGS: -ldl

#include <dlfcn.h>
#include "onnxruntime_wrapper.h"

typedef OrtApiBase* (*GetOrtApiBaseFunction)(void);

// Since Go can't call C function pointers directly, we just use this helper
// when calling GetApiBase
OrtApiBase *CallGetAPIBaseFunction(void *fn) {
	OrtApiBase *to_return = ((GetOrtApiBaseFunction) fn)();
	return to_return;
}
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

// Converts the given path to an ORTCHAR_T string, pointed to by a *C.char. The
// returned string must be freed using C.free when no longer needed. This
// wrapper is used for source compatibility with onnxruntime API functions
// requiring paths, which must be UTF-16 on Windows but UTF-8 elsewhere.
//
// It is defined here (rather than in setup_env.go) so that it is available for
// both the default build and the "static" build tag.
func createOrtCharString(str string) (*C.char, error) {
	return C.CString(str), nil
}

// This will contain the handle to the onnxruntime library (shared object in the
// default build, or the main executable's own symbol table in the static
// build) once it has been loaded successfully.
var libraryHandle unsafe.Pointer

// loadORTHandle obtains a handle to the onnxruntime symbols. The exact
// mechanism is build-tag specific: setup_env.go (default) loads a shared
// object from disk, while setup_env_static.go (the "static" tag) resolves the
// symbols that are already linked into the running executable via dlopen(NULL).
// Keeping only this small function tag-specific avoids duplicating the rest of
// the environment-setup logic between the two builds. Its declaration lives in
// the tag-specific file (setup_env.go / setup_env_static.go).
func platformCleanup() error {
	v, e := C.dlclose(libraryHandle)
	if v != 0 {
		return fmt.Errorf("Error closing the library: %w", e)
	}
	return nil
}

// Should only be called on Apple systems; looks up the CoreML provider
// function which should only be exported on apple onnxruntime dylib files.
func setAppendCoreMLFunctionPointer(libraryHandle unsafe.Pointer) error {
	// This function name must match the name in coreml_provider_factory.h,
	// which is provided in the onnxruntime release's include/ directory for
	// Apple platforms.
	fnName := "OrtSessionOptionsAppendExecutionProvider_CoreML"
	cFunctionName := C.CString(fnName)
	defer C.free(unsafe.Pointer(cFunctionName))
	appendCoreMLProviderProc := C.dlsym(libraryHandle, cFunctionName)
	if appendCoreMLProviderProc == nil {
		msg := C.GoString(C.dlerror())
		return fmt.Errorf("Error looking up %s: %s", fnName, msg)
	}
	C.SetCoreMLProviderFunctionPointer(appendCoreMLProviderProc)
	return nil
}

// platformInitializeEnvironment is shared by the default and "static" builds.
// It resolves the ORT entry points through the handle returned by
// loadORTHandle() (which differs per build tag) and registers them with the
// package's internal API base.
func platformInitializeEnvironment() error {
	handle, err := loadORTHandle()
	if err != nil {
		return err
	}
	cFunctionName := C.CString("OrtGetApiBase")
	defer C.free(unsafe.Pointer(cFunctionName))
	getAPIBaseProc := C.dlsym(handle, cFunctionName)
	if getAPIBaseProc == nil {
		C.dlclose(handle)
		msg := C.GoString(C.dlerror())
		return fmt.Errorf("Error looking up OrtGetApiBase: %s", msg)
	}
	ortAPIBase := C.CallGetAPIBaseFunction(getAPIBaseProc)
	tmp := C.SetAPIFromBase((*C.OrtApiBase)(unsafe.Pointer(ortAPIBase)))
	if tmp != 0 {
		C.dlclose(handle)
		return fmt.Errorf("Error setting ORT API base: %d", tmp)
	}
	if (runtime.GOOS == "darwin") || (runtime.GOOS == "ios") {
		setAppendCoreMLFunctionPointer(handle)
		// We'll silently ignore potential errors returned by
		// setAppendCoreMLFunctionPointer (for now at least). Even though we're
		// on Apple hardware, it's possible that the user will have compiled
		// the onnxruntime library from source without CoreML support.
		// A failure here will only leave the coreml function pointer as NULL
		// in our C code, which will be detected and result in an error at
		// runtime.
	}
	libraryHandle = handle
	return nil
}
