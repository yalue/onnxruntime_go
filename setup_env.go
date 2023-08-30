//go:build !windows

package onnxruntime_go

import (
	"fmt"
	"runtime"
	"unsafe"
)

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

// This file includes the code for loading the onnxruntime and setting up the
// environment on non-Windows systems. For now, it has been tested on Linux and
// arm64 OSX.

// This will contain the handle to the onnxruntime shared library if it has
// been loaded successfully.
var libraryHandle unsafe.Pointer

func platformCleanup() error {
	v, e := C.dlclose(libraryHandle)
	if v != 0 {
		return fmt.Errorf("Error closing the library: %w", e)
	}
	return nil
}

// Should only be called on Apple systems; looks up the CoreML provider
// function which should only be exported on apple onnxruntime dylib files.
// For now, this will never return an error; instead it will silently set the
// coreml provider function to a null pointer which will be detected at
// runtime.
func setAppendCoreMLFunctionPointer(libraryHandle unsafe.Pointer) error {
	// This function name must match the name in coreml_provider_factory.h,
	// which is provided in the onnxruntime release's include/ directory on for
	// Apple platforms.
	cFunctionName := C.CString("OrtSessionOptionsAppendExecutionProvider_CoreML")
	defer C.free(unsafe.Pointer(cFunctionName))
	appendCoreMLProviderProc := C.dlsym(libraryHandle, cFunctionName)
	C.SetCoreMLProviderFunctionPointer(appendCoreMLProviderProc)
	return nil
}

func platformInitializeEnvironment() error {
	if onnxSharedLibraryPath == "" {
		onnxSharedLibraryPath = "onnxruntime.so"
	}
	cName := C.CString(onnxSharedLibraryPath)
	defer C.free(unsafe.Pointer(cName))
	handle := C.dlopen(cName, C.RTLD_LAZY)
	if handle == nil {
		msg := C.GoString(C.dlerror())
		return fmt.Errorf("Error loading ONNX shared library \"%s\": %s",
			onnxSharedLibraryPath, msg)
	}
	cFunctionName := C.CString("OrtGetApiBase")
	defer C.free(unsafe.Pointer(cFunctionName))
	getAPIBaseProc := C.dlsym(handle, cFunctionName)
	if getAPIBaseProc == nil {
		C.dlclose(handle)
		msg := C.GoString(C.dlerror())
		return fmt.Errorf("Error looking up OrtGetApiBase in \"%s\": %s",
			onnxSharedLibraryPath, msg)
	}
	ortAPIBase := C.CallGetAPIBaseFunction(getAPIBaseProc)
	tmp := C.SetAPIFromBase((*C.OrtApiBase)(unsafe.Pointer(ortAPIBase)))
	if tmp != 0 {
		C.dlclose(handle)
		return fmt.Errorf("Error setting ORT API base: %d", tmp)
	}
	if (runtime.GOOS == "darwin") || (runtime.GOOS == "ios") {
		e = setAppendCoreMLFunctionPointer(handle)
		if e != nil {
			// This shouldn't actually ever happen in the current
			// implementation; instead the coreml function should remain NULL,
			// which will cause AppendExecutionProviderCoreML to return an
			// error at runtime.
			C.dlclose(handle)
			return fmt.Errorf("Error finding CoreML functionality: %w", e)
		}
	}
	libraryHandle = handle
	return nil
}
