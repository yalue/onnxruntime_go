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
func setAppendCoreMLFunctionPointer(libraryHandle unsafe.Pointer) error {
	// This function name must match the name in coreml_provider_factory.h,
	// which is provided in the onnxruntime release's include/ directory on for
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

// Converts the given path to an ORTCHAR_T string, pointed to by a *C.char. The
// returned string must be freed using C.free when no longer needed. This
// wrapper is used for source compatibility with onnxruntime API functions
// requiring paths, which must be UTF-16 on Windows but UTF-8 elsewhere.
func createOrtCharString(str string) (*C.char, error) {
	return C.CString(str), nil
}
