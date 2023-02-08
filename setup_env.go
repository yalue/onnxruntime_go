//go:build !windows

package onnxruntime_go

import (
	"fmt"
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
// environment on non-Windows systems. For now, it has only been tested on
// Linux.

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
	libraryHandle = handle
	return nil
}
